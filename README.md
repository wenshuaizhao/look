# Learning to Look: A Foundation for Visual Understanding

Wenshuai Zhao, Yihao Wang, Benjamin Robson, Juho Kannala, Arno Solin

[Project page](https://wenshuaizhao.github.io/look/) · [Probe notebook](notebooks/01_visual_awareness_probes.ipynb) · [VisABench notebook](notebooks/02_visabench_model_comparison.ipynb)

LOOK trains a vision-language model to identify which region of its visual input is available. Image crops retain their full-image positional information, and their coordinates provide self-supervised labels. VisABench tests visual-input awareness with real, withheld, and mismatched visual evidence.

## Installation

Use Python 3.12 and a CUDA GPU for model inference and training. The pinned environment uses Transformers 5.16.1 and PyTorch 2.14.0; select a CUDA build of PyTorch compatible with your driver.

```bash
git clone https://github.com/wenshuaizhao/look.git
cd look
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-notebooks.txt
python -m pip install -e vendor/ms-swift
python -m ipykernel install --user --name look --display-name 'Python (LOOK)'
export PYTHONPATH="$PWD/code${PYTHONPATH:+:$PYTHONPATH}"
export WORK="$PWD/work"
export HF_HOME="$WORK/cache/huggingface"
export MODELSCOPE_CACHE="$WORK/cache/modelscope"
export USE_HF=1
```

The local ms-swift package contains the framework version used by the experiments. LOOK's restored-position probes require the recorded Transformers vision-position APIs. Pretrained weights are downloaded on first use. SFT and LOOK adapters must be supplied locally or trained with the recipe below.

## Run the notebooks

```bash
export LOOK_SFT_ADAPTER=/path/to/sft/checkpoint-2848
export LOOK_BEST_ADAPTER=/path/to/look/checkpoint-5676
jupyter lab notebooks
```

Select the **Python (LOOK)** kernel and run these notebooks in order:

| Notebook | Experiment |
| --- | --- |
| [01_visual_awareness_probes.ipynb](notebooks/01_visual_awareness_probes.ipynb) | Compare full-image, black/grey pixel masks, zero/mean patch-embedding masks, crop-reset, and crop-restored probes. Inspect bounding-box answers, parse rate, IoU, and a constant-box baseline. |
| [02_visabench_model_comparison.ipynb](notebooks/02_visabench_model_comparison.ipynb) | Compare the pretrained Qwen3-VL model, SFT, LOOK (10:10), Qwen2.5-VL, LLaVA, and Gemma on the same real/no-evidence/mismatched conditions. Inspect probabilities and individual predictions. |

Four RefCOCO test samples and their original COCO photographs are bundled in `examples/visabench/`; the notebooks do not need the full dataset. The second notebook enables all six model configurations. Set adapter paths in its configuration cell or through the environment variables above. It also discovers the original final checkpoint names under `$WORK/runs/arm1/` and `$WORK/runs/arm2/`. Set a model's `enabled` flag to `False` to run a smaller comparison. Missing adapters raise an error rather than silently using pretrained weights.

Each model is loaded and released before the next one. Outputs, including raw answers/logits-derived probabilities, summaries, and run configuration, are saved under `$WORK/notebooks/`. The notebooks contain no precomputed model predictions. To inspect samples and configuration on CPU, start Jupyter with `LOOK_NOTEBOOK_PREVIEW=1`; this disables inference.

## Models and checkpoints

| Configuration | Pretrained model | Adapter |
| --- | --- | --- |
| Qwen3-VL-8B-Instruct | `Qwen/Qwen3-VL-8B-Instruct` | None |
| SFT | `Qwen/Qwen3-VL-8B-Instruct` | Instruction-only SFT, original final step 2,848 |
| LOOK (10:10) | `Qwen/Qwen3-VL-8B-Instruct` | 1:1 instruction/LOOK mixture, original final step 5,676 |
| Qwen2.5-VL-7B-Instruct | `Qwen/Qwen2.5-VL-7B-Instruct` | None |
| LLaVA-1.5-7B | `llava-hf/llava-1.5-7b-hf` | None |
| Gemma-4-12B | `google/gemma-4-12B-it` | None |

LOOK (10:10) is the best configuration for the paper's no-image Yes rate and real-to-no-image gap. Other training ratios are evaluated separately. Access to gated models requires accepting the model host's terms and authenticating with that host.

## Training

Build the instruction mixture and LOOK examples with a fixed seed:

```bash
python build_mix.py --out "$WORK/data/mix" --seed 0
python build_look.py --src "$WORK/data/mix" --out "$WORK/data/look" --seed 0
```

The instruction mixture contains 46,000 examples from VQAv2, TextVQA, Visual7W, A-OKVQA, ChartQA, TextCaps, ScienceQA, and GeomVerse. LOOK reuses these images; the crop coordinates supply the labels. These builders download the source datasets.

```bash
ARM=1 bash scripts/train.sh                         # Instruction-only SFT
ARM=2 bash scripts/train.sh                         # LOOK 10:10 (1:1)
ARM=3 LOOK_N=4600 RUN=arm3 bash scripts/train.sh      # LOOK 1:10
ARM=3 LOOK_N=9200 RUN=arm3b bash scripts/train.sh     # LOOK 2:10
ARM=2 RUN=locpos bash scripts/train.sh --template qwen3_vl  # Crop-reset control
```

The default recipe uses four GPUs, bfloat16, LoRA rank 16 and alpha 32, two epochs, learning rate 1e-4, a frozen vision encoder and aligner, batch size 1 per GPU, eight gradient-accumulation steps, sequence length 4,096, and maximum image area 1,003,520 pixels. `NPROC_PER_NODE`, `LR`, `EPOCHS`, and `MODEL` can be overridden. Keep the effective batch size at 32 when changing the GPU count.

The default `qwen3_vl_look` template preserves `objects.look_offset` and applies the full-image vision/LLM position overrides. The LocPos command uses the standard Qwen3-VL template on the same crop data, leaving positions crop-local. The training split holds out 1% for validation and is not image-disjoint. LOOK examples are generated once, rather than resampled each epoch.

## Full VisABench evaluation

The complete test uses 10,752 RefCOCO testA/testB expressions. Download the [RefCOCO parquet release](https://huggingface.co/datasets/jxu124/refcoco) and [COCO train2014 photographs](https://cocodataset.org/#download), then arrange them as described in [EVALUATION.md](EVALUATION.md). The four notebook examples are only a small-sample demonstration.

```bash
python -m bench.blind run --full --out "$WORK/runs/eval/arm0-base/blind_full"
python -m bench.blind run --full --adapters "$LOOK_SFT_ADAPTER" --out "$WORK/runs/eval/arm1-final/blind_full"
python -m bench.blind run --full --adapters "$LOOK_BEST_ADAPTER" --out "$WORK/runs/eval/arm2-final/blind_full"
python -m bench.blind score --out "$WORK/runs/eval/arm0-base/blind_full"
python -m bench.blind score --out "$WORK/runs/eval/arm1-final/blind_full"
python -m bench.blind score --out "$WORK/runs/eval/arm2-final/blind_full"
```

For another model family, pass its model ID with `--model` and omit `--adapters`. Generation is resumable from `preds.jsonl`. No-image features are replaced by image-placeholder embeddings; DeepStack contributions are zeroed. Yes/No probabilities come from disjoint single-token vocabulary variants at the first answer position. The mismatched condition uses an expression from another image, so its negative labels are approximate.

The paper's main Qwen3-VL results are:

| Model | No-image Yes (%) ↓ | Real-image Yes (%) ↑ | Different-image Yes (%) ↓ | Gap ↑ |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-VL-8B-Instruct | 2.5 | 93.3 | 15.6 | 90.7 |
| SFT | 31.2 | 94.9 | 17.0 | 63.7 |
| LOOK (10:10) | **0.8** | 94.1 | 15.1 | **93.2** |

Gap is the real-image Yes rate minus the no-image Yes rate, in percentage points. The paper omits no-image rates for Qwen2.5-VL and LLaVA because their untrained placeholder embeddings do not yield Yes/No responses. The comparison notebook retains raw forced-choice scores and top tokens for inspecting these cases.

## Localization and downstream evaluation

Prepare `$WORK/data/refcoco_val.jsonl` with one `images` list per row, pointing to a local validation photograph. Use the same images and seed for each model/checkpoint:

```bash
python -m looktask.probe run --n 1000 --seed 0 --out "$WORK/runs/eval/arm0-base/probe"
python -m looktask.probe score --out "$WORK/runs/eval/arm0-base/probe"
python -m looktask.probe run --n 1000 --seed 0 --adapters "$LOOK_BEST_ADAPTER" --out "$WORK/runs/eval/arm2-final/probe"
python -m looktask.probe score --out "$WORK/runs/eval/arm2-final/probe"
```

Report parse rate, IoU over all samples (parse failures count as zero), and IoU over parsed samples. [EVALUATION.md](EVALUATION.md) lists the grounding, CHAIR, ROPE, and other evaluation commands and their dataset layouts. RefCOCO image pools overlap the instruction/LOOK source images; use a separate image pool when testing image-level generalization.

Each evaluator writes per-example predictions and an aggregate JSON score file to its output directory. Use the same sample IDs, prompts, seed, and preprocessing settings when comparing runs.

## CPU data example and checks

```bash
python -m pip install -r requirements-demo.txt
python examples/quickstart.py
PYTHONPATH=. python -m unittest discover -s tests -v
```

The CPU example saves `context.png`, `crop.png`, and a native training row in `work/quickstart/`. It does not load a VLM.

## Citation

```bibtex
@misc{zhao2026learningtolook,
  title={Learning to Look: A Foundation for Visual Understanding},
  author={Wenshuai Zhao and Yihao Wang and Benjamin Robson and Juho Kannala and Arno Solin},
  year={2026},
  url={https://github.com/wenshuaizhao/look}
}
```

Research code and the vendored framework are licensed under [Apache-2.0](LICENSE). Models, datasets, photographs, and fonts retain their respective terms and attribution.
