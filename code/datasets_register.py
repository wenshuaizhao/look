"""Register the training-mix datasets that ms-swift does not ship.

Load with  --custom_register_path scripts/vlm_grounding/datasets_register.py
(swift/arguments/base_args/base_args.py:98), so nothing inside swift/ needs editing.

All splits (train AND validation) go into the mix: evaluation is on separate benchmarks
(MMMU, MMMU-Pro, MMStar, POPE, CHAIR, ROPE), none of which overlap these sources.

AI2D is deliberately absent: no usable train split exists on the hub. lmms-lab/ai2d and
lmms-lab-encoder/ai2d are test-only (3,088 = the benchmark split), lhndzn/AI2D relabels that
same 3,088 as "train", and tejas21private/ai2d_dataset is image+description, not QA.

Every preprocessor returns a dict containing BOTH `messages` and `images`. The return value
*replaces* the row (swift/dataset/preprocessor/core.py:211), so omitting `images` silently
turns a multimodal dataset into a text one -- the bug that was live in ScienceQA.
"""
from typing import Any, Dict, List, Optional

from swift.dataset import DatasetMeta, RowPreprocessor, SubsetDataset, register_dataset


def _img(v):
    """Normalise one image cell to a list ms-swift accepts."""
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


def _qa(query: str, answer: str, image) -> Dict[str, Any]:
    return {'messages': [{'role': 'user', 'content': query},
                         {'role': 'assistant', 'content': answer}],
            'images': _img(image)}


class VQAv2Preprocessor(RowPreprocessor):
    def preprocess(self, row):
        ans = row.get('multiple_choice_answer')
        if not ans:
            answers = row.get('answers') or []
            ans = max(set(answers), key=answers.count) if answers else None
        if not ans or not row.get('question'):
            return None
        return _qa(row['question'], str(ans), row.get('image'))


class TextVQAPreprocessor(RowPreprocessor):
    def preprocess(self, row):
        answers = row.get('answers') or []
        if not answers or not row.get('question'):
            return None
        return _qa(row['question'], max(set(answers), key=answers.count), row.get('image'))


class ChartQAPreprocessor(RowPreprocessor):
    def preprocess(self, row):
        label = row.get('label')
        label = label[0] if isinstance(label, list) and label else label
        if not label or not row.get('query'):
            return None
        return _qa(row['query'], str(label), row.get('image'))


class MMWrappedPreprocessor(RowPreprocessor):
    """nimapourjafar/mm_* layout: images=[{bytes,path}], data=[{data, modality, role}].

    An image turn carries the image's *index* as a string; text turns carry the text. Turns
    from the same consecutive role are merged into one message.
    """

    def preprocess(self, row):
        turns: List[Dict[str, Any]] = row.get('data') or []
        images = row.get('images') or []
        messages, used = [], 0
        for t in turns:
            role = 'assistant' if t.get('role') == 'assistant' else 'user'
            if t.get('modality') == 'image':
                piece, used = '<image>', used + 1
            else:
                piece = (t.get('data') or '').strip()
            if not piece:
                continue
            if messages and messages[-1]['role'] == role:
                messages[-1]['content'] = (messages[-1]['content'] + '\n' + piece).strip()
            else:
                messages.append({'role': role, 'content': piece})
        if len(messages) < 2 or messages[-1]['role'] != 'assistant':
            return None
        return {'messages': messages, 'images': images[:used] if used else images}


def _hf(name: str, hf_id: str, pre: RowPreprocessor, splits: List[str], tags: List[str]):
    register_dataset(DatasetMeta(dataset_name=name, hf_dataset_id=hf_id, split=splits,
                                 preprocess_func=pre, tags=tags))


# train + validation both feed the mix; benchmarks are separate datasets entirely.
_hf('vqav2',     'Multimodal-Fatima/VQAv2_train', VQAv2Preprocessor(),   ['train'],
    ['multi-modal', 'vqa', 'en'])
_hf('textvqa',   'lmms-lab/textvqa',              TextVQAPreprocessor(), ['train', 'validation'],
    ['multi-modal', 'ocr', 'en'])
_hf('chartqa',   'HuggingFaceM4/ChartQA',         ChartQAPreprocessor(), ['train', 'val'],
    ['multi-modal', 'chart', 'en'])
_hf('visual7w',  'nimapourjafar/mm_visual7w',     MMWrappedPreprocessor(), ['train'],
    ['multi-modal', 'vqa', 'spatial', 'en'])
_hf('geomverse', 'nimapourjafar/mm_geomverse',    MMWrappedPreprocessor(), ['train'],
    ['multi-modal', 'geometry', 'reasoning', 'en'])

# Already in ms-swift, listed here for reference:
#   swift/A-OKVQA   swift/TextCaps   swift/ScienceQA (image-drop bug fixed in mllm.py)
