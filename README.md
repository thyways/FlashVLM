# FlashVLM

## 🔖Table of Contents

1. [News](#news)
2. [Todo List](#todo-list)
3. [Highlights](#highlights)
4. [Project Structure](#project-structure)
5. [Installation](#installation)
6. [Quickstart](#quickstart)
7. [Evaluation](#evaluation)
8. [Acknowledgement](#acknowledgement)
9. [Citation](#citation)

## 🔥News

- [2026.03.04] Added root-level `pyproject.toml` for reproducible installation.
- [2026.03.04] Added dual optional environments:
  - `qwen` with `transformers==5.2.0`
  - `llava` with `transformers==4.57.3`
- [2026.03.04] Included local `llava/` package and `method/` packages in distributable project metadata.

## 📋Todo List

- [x] Package `llava` and `method/*` into one installable project.
- [x] Provide reproducible environment definitions for Qwen and LLaVA settings.
- [x] Keep Qwen3-VL + multiple compressor scripts under unified `lmms-eval` workflow.
- [ ] Add end-to-end benchmark result tables and logs summary.
- [ ] Add smoke-test CI for both `qwen` and `llava` optional dependencies.

## ✨Highlights


## 🧱Project Structure

```text
FlashVLM/
├── llava/                  # local llava package (imported by lmms-eval llava models)
├── method/                 # compressor implementations
│   ├── flashvid/
│   ├── vidcom2/
│   ├── fastv/
│   ├── visionzip/
│   └── holitom/
├── scripts/                # runnable experiment scripts
│   ├── baseline/
│   ├── flashvid/
│   ├── vidcom2/
│   ├── fastv/
│   ├── visionzip/
│   └── holitom/
├── lmms-eval/              # vendored lmms-eval (v0.7.0-compatible workflow)
└── pyproject.toml          # reproducible package metadata
```

## 📦Installation

> Use **separate virtual environments** for Qwen and LLaVA settings, since their `transformers` versions are different.

1. Clone the repository:

```bash
git clone https://github.com/thyways/FlashVLM.git
cd FlashVLM
```

2. Create and activate a virtual environment:
uv：
```bash
uv venv --python 3.12 --seed --managed-python
source .venv/bin/activate
```
conda：
```bash
conda create -n flashvlm python==3.12
conda activate flashvlm
pip install uv
```

3. Install one of the two optional environments:

- Qwen setup (`transformers==5.2.0`)

```bash
uv pip install -e ".[qwen]"
```

- LLaVA setup (`transformers==4.57.3`)

```bash
uv pip install -e ".[llava]"
```

## 🚀Quickstart

### 1) Baseline Qwen3-VL

```bash
bash scripts/baseline/qwen3_vl.sh
```

### 2) Switch compressor for Qwen3-VL

```bash
bash scripts/flashvid/qwen3_vl.sh
bash scripts/vidcom2/qwen3_vl.sh
bash scripts/fastv/qwen3_vl.sh
bash scripts/visionzip/qwen3_vl.sh
bash scripts/holitom/qwen3_vl.sh
```

### 3) LLaVA baselines

```bash
bash scripts/baseline/llava_ov.sh
bash scripts/baseline/llava_vid.sh
```

## 📊Evaluation

All scripts run through `accelerate launch -m lmms_eval`.

For Qwen3-VL runs, method selection is controlled by `COMPRESSOR` in `lmms_eval/models/simple/qwen3_vl.py`.
Supported values:
- `flashvid`
- `vidcom2`
- `fastv`
- `visionzip`
- `holitom`

Output logs are written under `./logs/<method_name>` by default in each script.

## 👏Acknowledgement

This repository builds on multiple open-source projects and methods, including:
- [LMMs-Eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)
- [LLaVA / LLaVA-OneVision / LLaVA-Video](https://github.com/LLaVA-VL)
- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)
- [FlashVID](https://github.com/Fanziyang-v/FlashVID)
- [VidCom2](https://github.com/THUDM/VidCom2)
- [FastV](https://github.com/pkunlp-icler/FastV)
- [VisionZip](https://github.com/dvlab-research/VisionZip)
- [HoliTom](https://github.com/cokeshao/HoliTom)

## 📜Citation

If this repository helps your work, please cite the relevant original papers/method repositories above.
