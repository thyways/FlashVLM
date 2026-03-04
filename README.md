# FlashVLM

## 🔖Table of Contents

1. [Installation](#installation)
2. [Environment Profiles](#environment-profiles)
3. [Quickstart](#quickstart)
4. [Notes](#notes)

## 📦Installation

This project uses [uv](https://github.com/astral-sh/uv) for environment and dependency management.

```bash
git clone https://github.com/thyways/FlashVLM.git
cd FlashVLM
```

## ⚙️Environment Profiles

Two optional Python environments are defined in `pyproject.toml`:

- `qwen`:
  - `transformers==5.2.0`
- `llava`:
  - `transformers==4.57.3`
  - local editable package `./llava` (published as `flashvlm-llava`)

## 🚀Quickstart

Create and install the `qwen` environment:

```bash
uv venv ".venv-qwen" --python "3.12"
uv pip install --python ".venv-qwen/bin/python" --group "qwen"
```

Create and install the `llava` environment:

```bash
uv venv ".venv-llava" --python "3.12"
uv pip install --python ".venv-llava/bin/python" --group "llava"
```

## 📝Notes

- `qwen` and `llava` are isolated environments with different `transformers` versions.
- `./llava/pyproject.toml` is used so the local `llava` code can be installed in the `llava` environment.
