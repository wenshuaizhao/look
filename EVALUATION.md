# Evaluation

Run from the repository root with the training environment installed:

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export WORK="$PWD/work"
export HF_HOME="$WORK/cache/huggingface"
export MODELSCOPE_CACHE="$WORK/cache/modelscope"
```

All model generation needs a GPU and separately downloaded model weights. Add `--adapters /path/to/checkpoint` for a trained LoRA adapter. Base-model evaluation omits it. Default model: `Qwen/Qwen3-VL-8B-Instruct`.

## Dataset layout

The original evaluation loaders expect these layouts under `$WORK`:

| Evaluation | External files |
| --- | --- |
| LOOK sweep / position probe | `data/refcoco_val.jsonl`: native rows with an `images` list containing an absolute local image path; sampled once per unique image |
| VisABench | `cache/refcoco_hf/`: jxu124 RefCOCO parquet files; `cache/modelscope/media_resources/coco2014/train2014/`: COCO images |
| RefCOCO / RefCOCO+ / RefCOCOg | Same parquet directory and COCO train2014 images; see `bench/ground.py` split definitions |
| CHAIR | `cache/coco/val2014/` and `cache/coco/annotations/{instances,captions}_val2014.json`; NLTK WordNet resources |
| ROPE | `cache/rope/validation/`: sled-umich/ROPE validation release |
| Other grounding benchmarks | `cache/groundingme/`, `cache/refl4/`, `cache/refadv/`, `cache/finecops/`, `cache/sorec/`; see constants and loader schemas in `bench/ground.py` |

The release does not redistribute these benchmark datasets or silently download them. Consult each dataset's release for access and use terms. Paths in input JSONL files must point to your own images. Inspect a module's `--help` and loader for the precise split filenames before launching expensive inference.

## Commands

```bash
# Position-preserving crop localization; same-image reset-position controls are included.
python -m looktask.sweep --n 50 --out "$WORK/runs/eval/base/sweep.json"
python -m looktask.probe run --n 1000 --out "$WORK/runs/eval/base/probe"

# VisABench: real, no-evidence, and mismatched-image conditions.
python -m bench.blind run --full --out "$WORK/runs/eval/base/blind_full"
python -m bench.blind score --out "$WORK/runs/eval/base/blind_full"

# Downstream benchmarks: generation and scoring are separate, resumable operations.
python -m bench.ground gen --sets refcoco_testA --out "$WORK/runs/eval/base/ground"
python -m bench.ground score --out "$WORK/runs/eval/base/ground"
python -m bench.chair gen --n 500 --out "$WORK/runs/eval/base/chair"
python -m bench.chair score --out "$WORK/runs/eval/base/chair"
python -m bench.rope gen --out "$WORK/runs/eval/base/rope"
python -m bench.rope score --out "$WORK/runs/eval/base/rope"
```

`bench/blind.py` is the original VisABench implementation, retaining its experimental filename. It preserves the input sequence layout while replacing visual features and DeepStack contributions to remove evidence. This differs from passing a black image. Report no-evidence and real-image behavior together to detect indiscriminate refusal.

LOOK reports should include parse rate, IoU over all samples with parse failures scored zero, and IoU over parsed samples. Output-format gains can otherwise appear to improve localization. The paper documents image-pool overlap between LOOK construction and RefCOCO; use disjoint image pools when evaluating generalization. Gains are not uniform across downstream benchmarks.

Each evaluator saves per-example predictions and aggregate scores to its output directory. Use `bench.blind score`, `looktask.probe score`, or the corresponding downstream scorer to summarize a completed run.
