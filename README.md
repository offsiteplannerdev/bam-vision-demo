# BAM Vision Demo

Binary defect classification prototype for automotive part inspection images. This repository represents an early R&D baseline: simple data loading, albumentations transforms, a lightweight CNN, and a configurable PyTorch training loop.

The intended dataset layout is:

```text
data/
  train/
    defect/
    ok/
  val/
    defect/
    ok/
```

No real inspection images are included in this repository.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Train

```bash
python -m src.train --config configs/default.yaml
```

The default config assumes images are stored under `data/train` and `data/val` relative to the repository root.

## Test

```bash
pytest
```

Current test coverage is intentionally light and mostly checks transform output contracts.

## Current Gaps

- No real manufacturing image samples are committed.
- Dataset tests are missing.
- The baseline CNN is intentionally simple and not production-tuned.
- Notebook exploration has not been filled in yet.
