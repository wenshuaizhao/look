"""Shared model loading + single-image generation for the custom benchmarks (CHAIR, ROPE).

Runs in the TRAINING env (transformers 5.x) so a LoRA checkpoint is loaded exactly as the
look-task sweep loads it (`swift.get_model_processor` + `Swift.from_pretrained`). Images are
resized to the same canonical geometry the training template uses (MAX_PIXELS, cell-aligned), so
the model sees the same token budget at eval as during SFT.
"""
from __future__ import annotations
import json
import os
import sys
os.environ.setdefault('WORK', os.path.abspath(os.environ.get('LOOK_WORK', 'work')))
os.environ.setdefault('MODELSCOPE_CACHE', os.environ['WORK'] + '/cache/modelscope')
os.environ.setdefault('HF_HOME', os.environ['WORK'] + '/cache/huggingface')
os.environ.setdefault('NLTK_DATA', os.environ['WORK'] + '/cache/nltk_data')
os.environ.setdefault('MAX_PIXELS', '1003520')
os.environ.setdefault('PYTHONNOUSERSITE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import torch                                                          # noqa: E402
from PIL import Image                                                 # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from looktask.geometry import Geometry                                # noqa: E402

WORK = os.environ['WORK']


def load_model(model_id='Qwen/Qwen3-VL-8B-Instruct', adapters=None):
    from swift import get_model_processor
    model, processor = get_model_processor(model_id, torch_dtype=torch.bfloat16, device_map='cuda:0')
    if adapters:
        from swift.tuners import Swift
        model = Swift.from_pretrained(model, adapters)
        print('loaded adapters:', adapters, flush=True)
    model.eval()
    geo = Geometry(processor.image_processor, int(os.environ['MAX_PIXELS']))
    return model, processor, geo


def canonical(img: Image.Image, geo: Geometry) -> Image.Image:
    img = img.convert('RGB')
    rw, rh = geo.canonical_size(*img.size)
    return img if (rw, rh) == img.size else img.resize((rw, rh), Image.LANCZOS)


@torch.no_grad()
def generate(model, processor, img: Image.Image, prompt: str, max_new_tokens: int) -> str:
    msgs = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': prompt}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], return_tensors='pt', do_resize=False).to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    txt = processor.tokenizer.decode(out[0, inputs['input_ids'].shape[1]:], skip_special_tokens=False)
    for term in ('<|im_end|>', '<|endoftext|>'):
        txt = txt.replace(term, '')
    return txt.strip()


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:      # torn line from two appenders (set-A and set-B jobs share a file)
                continue
    return rows


def append_jsonl(path, row):
    with open(path, 'a') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')
        f.flush()
