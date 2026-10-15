"""Small-sample notebook runners using the paper's probe and VisABench implementations."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import base64
import gc
import hashlib
import html
import json
import os

PROBE_CONDITIONS = {
    'full': 'Full image; target is the full canvas.',
    'black': 'Pixels outside the selected window are black.',
    'grey': 'Pixels outside the selected window are grey.',
    'zero': 'Outside patch embeddings are zeroed; their positions remain.',
    'mean': 'Outside patch embeddings are replaced by their mean; positions remain.',
    'crop_reset': 'Only crop pixels, with crop-local positions.',
    'crop_restored': 'Only crop pixels, with full-image positions (LOOK).',
}


def load_samples(root):
    directory = Path(root) / 'code/examples/visabench'
    rows = json.loads((directory / 'samples.json').read_text())['samples']
    result = []
    for row in rows:
        image = (directory / row['image']).resolve()
        if not image.is_file() or hashlib.sha256(image.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError(f'Missing or changed sample image: {image}')
        result.append({**row, 'image': str(image)})
    return result


def display_table(rows, columns=None):
    from IPython.display import HTML, display
    if not rows:
        print('No inference results yet.')
        return
    columns = columns or list(rows[0])
    def fmt(value):
        if value is None:
            return '—'
        if isinstance(value, float):
            return f'{value:.4f}'
        return html.escape(str(value))
    head = ''.join(f'<th style="padding:8px;text-align:left">{html.escape(c)}</th>' for c in columns)
    body = ''.join('<tr>' + ''.join(f'<td style="padding:8px;border-top:1px solid #ddd">{fmt(r.get(c))}</td>'
                                   for c in columns) + '</tr>' for r in rows)
    display(HTML(f'<div style="overflow-x:auto"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'))


def display_samples(samples):
    from IPython.display import HTML, display
    cards = []
    for row in samples:
        data = base64.b64encode(Path(row['image']).read_bytes()).decode()
        cards.append(f'<figure style="width:240px;margin:10px"><img src="data:image/jpeg;base64,{data}" '
                     f'style="width:240px;height:180px;object-fit:contain" alt="{html.escape(row["expr"])}">'
                     f'<figcaption><b>{html.escape(row["expr"])}</b><br>'
                     f'<small>{html.escape(row["id"])}</small></figcaption></figure>')
    display(HTML('<div style="display:flex;flex-wrap:wrap">' + ''.join(cards) + '</div>'))


def find_adapter(work, arm, step, env_name):
    configured = os.environ.get(env_name)
    if configured:
        return str(Path(configured).expanduser().resolve())
    matches = sorted((Path(work) / 'runs' / arm).glob(f'**/checkpoint-{step}'))
    if len(matches) == 1:
        return str(matches[0].resolve())
    if len(matches) > 1:
        raise ValueError(f'Multiple {arm} checkpoints found. Set {env_name} explicitly.')
    return None


def default_models(work):
    qwen = 'Qwen/Qwen3-VL-8B-Instruct'
    return [
        dict(key='qwen3', label='Qwen3-VL-8B-Instruct', model=qwen, adapter=None, enabled=True, report_blind=True),
        dict(key='sft', label='Qwen3-VL-8B + SFT', model=qwen,
             adapter=find_adapter(work, 'arm1', 2848, 'LOOK_SFT_ADAPTER'), enabled=True,
             requires_adapter=True, report_blind=True),
        dict(key='look', label='Qwen3-VL-8B + LOOK (10:10)', model=qwen,
             adapter=find_adapter(work, 'arm2', 5676, 'LOOK_BEST_ADAPTER'), enabled=True,
             requires_adapter=True, report_blind=True),
        dict(key='qwen25', label='Qwen2.5-VL-7B-Instruct', model='Qwen/Qwen2.5-VL-7B-Instruct',
             adapter=None, enabled=True, report_blind=False),
        dict(key='llava', label='LLaVA-1.5-7B', model='llava-hf/llava-1.5-7b-hf',
             adapter=None, enabled=True, report_blind=False),
        dict(key='gemma4', label='Gemma-4-12B', model='google/gemma-4-12B-it',
             adapter=None, enabled=True, report_blind=True),
    ]


def validate_models(models):
    active = [m for m in models if m['enabled']]
    if not active:
        raise ValueError('Enable at least one model.')
    for spec in active:
        path = spec.get('adapter')
        if spec.get('requires_adapter') and not path:
            raise ValueError(f'{spec["label"]}: set its adapter path in MODELS or set '
                             'LOOK_SFT_ADAPTER / LOOK_BEST_ADAPTER before starting Jupyter. '
                             'Train ARM=1 / ARM=2 if you do not have the adapters.')
        if path:
            directory = Path(path)
            if not (directory / 'adapter_config.json').is_file():
                raise ValueError(f'Adapter configuration not found: {directory}/adapter_config.json')
            if not any(directory.glob('adapter_model*.safetensors')) and not any(directory.glob('adapter_model*.bin')):
                raise ValueError(f'Adapter weights not found: {directory}')
    return active


def require_gpu():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Model inference requires a working CUDA GPU. Use LOOK_NOTEBOOK_PREVIEW=1 '
                           'for sample/configuration preview without model loading.')
    return torch


def save_run(directory, rows, summary, config):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'predictions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (directory / 'config.json').write_text(json.dumps(config, indent=2) + '\n')


def summarize_probes(rows):
    from looktask.metrics import iou
    groups = defaultdict(list)
    for row in rows:
        groups[row['condition']].append(row)
    out = []
    for condition, rs in groups.items():
        parsed = [r for r in rs if r['pred'] is not None]
        out.append(dict(condition=condition, n=len(rs), parse_rate=len(parsed)/len(rs),
                        iou_all=sum(r['iou'] for r in rs)/len(rs),
                        iou_parsed=sum(r['iou'] for r in parsed)/len(parsed) if parsed else None,
                        success_at_05=sum(r['iou'] >= .5 for r in rs)/len(rs),
                        whole_canvas_iou=sum(iou(r['gt'], [0,0,1000,1000]) for r in rs)/len(rs)))
    return out


def display_probe_boxes(rows):
    """Compare predictions for the first image in the original canvas coordinates."""
    if not rows:
        return
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Patch
    from PIL import Image
    selected = [r for r in rows if r['id'] == rows[0]['id']]
    fig, axes = plt.subplots((len(selected)+2)//3, 3, figsize=(12, 3.5*((len(selected)+2)//3)),
                             squeeze=False)
    for ax in axes.flat:
        ax.axis('off')
    for ax, row in zip(axes.flat, selected):
        image = Image.open(row['image']).convert('RGB')
        ax.imshow(image)
        for field, color in [('gt','#147d83'), ('pred','#d68436')]:
            if row[field] is None:
                continue
            x0,y0,x1,y1 = [v*s/1000 for v,s in zip(row[field], [image.width,image.height]*2)]
            ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor=color,linewidth=2))
        ax.set_title(f'{row["condition"]}: IoU {row["iou"]:.3f}')
    fig.legend(handles=[Patch(facecolor='#147d83',label='Target'),Patch(facecolor='#d68436',label='Prediction')],
               loc='lower center',ncol=2)
    fig.suptitle('Boxes in the original image coordinates')
    fig.tight_layout(rect=(0,.04,1,.96))
    plt.show()


def run_visual_probes(samples, output, model_id='Qwen/Qwen3-VL-8B-Instruct',
                      adapter=None, seed=0, conditions=None, max_new_tokens=48):
    if 'qwen3' not in model_id.lower():
        raise ValueError('The position-restoration probes require a Qwen3-VL model.')
    torch = require_gpu()
    import numpy as np
    from contextlib import ExitStack
    from PIL import Image
    from bench.vlm import load_model
    from looktask import make_look_sample
    from looktask.metrics import iou, parse_box
    from looktask.positions import PosOverride, LlmPosOffset, find_visual, assert_rows_in_full
    from looktask.probe import PatchEmbedMask, outside_rows, painted

    conditions = list(conditions or PROBE_CONDITIONS)
    if set(conditions) - set(PROBE_CONDITIONS):
        raise ValueError('Unknown probe condition.')
    model, processor, geo = load_model(model_id, adapter)
    try:
        visual = find_visual(model)
        num_grid = int(getattr(model.config.vision_config, 'num_position_embeddings', 0) ** .5) or visual.num_grid_per_side
        rng = np.random.default_rng(seed)
        rows = []
        for example in samples:
            with Image.open(example['image']) as image:
                sample = make_look_sample(image, geo, rng)
            gw, gh, _, _ = geo.grids(*sample.canvas)
            cgw, cgh, _, _ = geo.grids(*sample.crop.size)
            mapping = geo.crop_rows_in_full(*sample.offset, cgw, cgh, gw)
            grid = torch.tensor([[1, gh, gw]])
            assert_rows_in_full(geo, grid, cgw, cgh, mapping, sample.offset)
            prompt = 'Which region of the full image can you see? Answer with a bounding box.'
            for cond in conditions:
                contexts = []
                gt = [0,0,1000,1000] if cond == 'full' else sample.gt_norm1000
                if cond == 'full':
                    img = sample.canon
                elif cond in ('black','grey'):
                    img = painted(sample, (0,0,0) if cond == 'black' else (128,128,128))
                elif cond in ('zero','mean'):
                    img = sample.canon
                    contexts = [PatchEmbedMask(visual, outside_rows(geo, sample), cond)]
                else:
                    img = sample.crop
                    if cond == 'crop_restored':
                        contexts = [PosOverride(visual, sample.offset, geo.merge, grid, mapping, num_grid),
                                    LlmPosOffset(model, sample.offset)]
                messages = [{'role':'user','content':[{'type':'image'},{'type':'text','text':prompt}]}]
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[text], images=[img], return_tensors='pt', do_resize=False).to(model.device)
                with ExitStack() as stack, torch.inference_mode():
                    for context in contexts:
                        stack.enter_context(context)
                    generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
                answer = processor.tokenizer.decode(generated[0, inputs['input_ids'].shape[1]:], skip_special_tokens=False)
                for term in ('<|im_end|>','<|endoftext|>'):
                    answer = answer.replace(term,'')
                pred = parse_box(answer)
                rows.append(dict(id=example['id'], image=example['image'], condition=cond,
                                 gt=gt, pred=pred, iou=iou(gt,pred), answer=answer.strip(), prompt=prompt))
                print(example['id'], cond, f'IoU={rows[-1]["iou"]:.3f}')
        save_run(output, rows, summarize_probes(rows),
                 dict(model=model_id, adapter=adapter, seed=seed, conditions=conditions,
                      sample_ids=[s['id'] for s in samples], max_pixels=os.environ.get('MAX_PIXELS')))
        return rows
    finally:
        del model, processor
        gc.collect()
        torch.cuda.empty_cache()


def summarize_visabench(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['model'], row['condition']].append(row)
    result = []
    for (model, condition), rs in groups.items():
        forced = sum(r['p_yes'] > .5 for r in rs)/len(rs)
        mean_p = sum(r['p_yes'] for r in rs)/len(rs)
        report = all(r['reportable'] for r in rs)
        result.append(dict(model=model, condition=condition, n=len(rs),
                           yes_rate=forced if report else None,
                           mean_p_yes=mean_p if report else None,
                           top_token_yes_no=sum(r['top_token'].strip().lower() in ('yes','no') for r in rs)/len(rs),
                           forced_yes_rate=forced, forced_mean_p_yes=mean_p))
    for row in result:
        peers = {r['condition']:r for r in result if r['model']==row['model']}
        real, blind = peers.get('real'), peers.get('blind')
        row['gap'] = real['yes_rate']-blind['yes_rate'] if real and blind and blind['yes_rate'] is not None else None
    return result


def run_visabench(samples, models, output):
    active = validate_models(models)
    torch = require_gpu()
    from PIL import Image
    from bench.blind import load_any, yn_ids, Blind, p_yes, PROMPT
    from bench.vlm import canonical

    all_rows = []
    for spec in active:
        print('Loading', spec['label'])
        model, processor, geo = load_any(spec['model'], spec['adapter'])
        blind = None
        try:
            tok = processor.tokenizer
            yes_ids, no_ids = yn_ids(tok)
            pad_id = next(getattr(model.config,k) for k in ('image_token_id','image_token_index')
                          if getattr(model.config,k,None) is not None)
            blind = Blind(model, pad_id)
            rows = []
            for example in samples:
                with Image.open(example['image']) as image:
                    img = canonical(image,geo) if geo is not None else image.convert('RGB')
                for cond in ('blind','real','mismatch'):
                    expr = example['mismatch_expr'] if cond=='mismatch' else example['expr']
                    prob, top = p_yes(model,processor,img,PROMPT.format(expr=expr),yes_ids,no_ids,
                                      blind if cond=='blind' else None,native_geo=geo is not None)
                    rows.append(dict(model=spec['label'], key=spec['key'], id=example['id'],
                                     condition=cond, expr=expr, image=example['image'],
                                     expected='Yes' if cond=='real' else 'No', p_yes=prob,
                                     forced_answer='Yes' if prob>.5 else 'No', top_token=tok.decode([top]),
                                     reportable=cond!='blind' or spec['report_blind']))
            save_run(Path(output)/spec['key'], rows, summarize_visabench(rows),
                     dict(spec, sample_ids=[s['id'] for s in samples]))
            all_rows.extend(rows)
        finally:
            del blind, model, processor
            gc.collect()
            torch.cuda.empty_cache()
    save_run(output, all_rows, summarize_visabench(all_rows), dict(models=active))
    return all_rows
