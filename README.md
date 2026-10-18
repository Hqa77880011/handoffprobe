# HandoffProbe

Implementation of **HandoffProbe: Consumer-Conditioned Semantic Tests for Collaborative Data Analysis**.

## Installation

Use Python `>=3.10`. Run commands from the repository root. Create the environment, activate it with `source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\Activate.ps1` (PowerShell), then install the package.

```bash
python -m venv .venv
```

```bash
python -m pip install -e .
```

## Data preparation

Import [InfiAgent-DABench](https://github.com/InfiAgent/InfiAgent). Preparation groups questions by source table, separates runtime tasks from evaluator labels, and fixes file splits with seed 2027.

```bash
git clone https://github.com/InfiAgent/InfiAgent data/raw/InfiAgent
handoffprobe prepare --source infi --input data/raw/InfiAgent --output data/infi
```

[DSBench](https://github.com/LiqiangJing/DSBench) is also supported: extract its analysis data pack to the cloned `data_analysis/data` directory, then use `handoffprobe prepare --source dsbench --input data/raw/DSBench --output data/dsbench`. Custom manifests contain `id,source_id,split,dataset,request,tables,dictionary`; labels stay in a separate file.

## Declarations and training

Local model commands download Hugging Face weights. Inspect generated source bindings and anchor recipes before using them as clean supervision. The scorer trains only on `train` tasks.

```bash
handoffprobe build-specifications --tasks data/infi/tasks.json --split train --model Qwen/Qwen2.5-7B-Instruct --output data/train_specs.json
handoffprobe make-controls --tasks data/infi/tasks.json --specifications data/train_specs.json --output data/probe_examples.json
handoffprobe train --examples data/probe_examples.json --output checkpoints/scorer.pkl
```

## Execution and evaluation

`configs/default.json` sets probe, executor and token budgets. Use `--revision` to pin model weights and `--seed` for independent runs. The example explicitly uses seed 17.

```bash
handoffprobe run --tasks data/infi/tasks.json --scorer checkpoints/scorer.pkl --model Qwen/Qwen2.5-7B-Instruct --seed 17 --method handoffprobe --output runs/handoffprobe
handoffprobe evaluate --input runs/handoffprobe/traces.json --labels data/infi/labels.json --output runs/handoffprobe/metrics.json
handoffprobe plot --input runs/handoffprobe/metrics.json --output runs/figures
```

## Ablations

Reuse the same scorer, model and tasks with `run --method no_impact`, `no_applicability`, `no_coverage`, `no_witness`, `no_anchor` or `no_repair`, writing separate run directories. `no_consumer` requires separate supervision from `make-controls --no-consumer` and a separately fitted scorer. Controlled held-out metrics require the corresponding audit pack.

```bash
handoffprobe build-specifications --tasks data/infi/tasks.json --split test --model Qwen/Qwen2.5-7B-Instruct --output data/test_specs.json
handoffprobe make-controls --tasks data/infi/tasks.json --specifications data/test_specs.json --split test --labels data/infi/labels.json --export-pack data/controls --output data/controls_examples.json
handoffprobe run --tasks data/controls/tasks.json --specifications data/controls/specifications.json --scorer checkpoints/scorer.pkl --model Qwen/Qwen2.5-7B-Instruct --seed 17 --output runs/controls
handoffprobe evaluate --input runs/controls/traces.json --labels data/controls/labels.json --audit data/controls/audit.json --output runs/controls/metrics.json
```

## Outputs

Preparation writes `tasks.json` and `labels.json`. Control generation writes examples and an audit companion; exported packs contain separate tasks, specifications, labels and audit files. `scorer.pkl` is the fitted scorer. Runs write `traces.json` with checks, witnesses, repairs and token/executor ledgers. Evaluation writes metrics; plots are PNG/SVG.

## Verification

Source checks used Python 3.13.5. Installed dependency metadata: numpy 2.1.3, torch 2.7.1+cu118, scipy 1.15.3, scikit-learn 1.6.1, matplotlib 3.10.0. Python AST, configuration files, CLI parser definitions and source input/output paths were checked without invoking command handlers. Training, full evaluation and CI were not run for this review.
