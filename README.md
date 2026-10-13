# HandoffProbe

Implementation of **HandoffProbe: Consumer-Conditioned Semantic Tests for Collaborative Data Analysis**.

HandoffProbe freezes a source-grounded contract before repair, predicts the answer impact of executable checks, and selects complementary tests under a budget. Failures return clause IDs, affected fields, expected/observed values and source row identities. One producer repair is followed by rechecking the entire selected set.

## Installation

Use Python 3.11 and PyTorch for your hardware. Activate a virtual environment, then install:

```bash
python -m pip install -e .
```

Local model execution uses Hugging Face Transformers. Commands run from the repository root. Model files are downloaded to the Hugging Face cache; no API key is needed.

## Data

Import the public [InfiAgent-DABench](https://github.com/InfiAgent/InfiAgent) questions, CSV tables and separate evaluator labels:

```bash
git clone https://github.com/InfiAgent/InfiAgent data/raw/InfiAgent
handoffprobe prepare --source infi --input data/raw/InfiAgent --output data/infi
```

The adapter preserves original requests, statistical constraints and answer formats. A 52-file release uses the 31/10/11 file partition with seed 2027; other inventories use 60/20/20 proportions. All questions from a file remain together. Runtime `tasks.json` and sealed `labels.json` are separate.

For [DSBench](https://github.com/LiqiangJing/DSBench), extract its analysis pack into `data_analysis/data` after cloning:

```bash
git clone https://github.com/LiqiangJing/DSBench data/raw/DSBench
handoffprobe prepare --source dsbench --input data/raw/DSBench --output data/dsbench
```

DSBench remains an external test grouped by challenge. CSV, XLSX and XLS tables are read directly; original filenames remain their table identifiers. The runtime receives the original question and tables, never the sealed answer.

Curated packs can use `prepare --source manifest --input tasks.json --output data/custom`. Entries contain `id`, `source_id`, `split`, `dataset`, `request`, `tables` (table name → CSV path) and `dictionary` (evidence key → public definition). Paths are relative to the input JSON.

## Declarations and training

```bash
handoffprobe build-specifications --tasks data/infi/tasks.json --split train --model Qwen/Qwen2.5-7B-Instruct --output data/train_specs.json
handoffprobe make-controls --tasks data/infi/tasks.json --specifications data/train_specs.json --output data/probe_examples.json
handoffprobe train --examples data/probe_examples.json --output checkpoints/scorer.pkl
```

Inspect source bindings and anchor recipes in generated training declarations before treating them as clean controls. A specification contains `id`, `consumer`, `plan` and `contract`. Each clause has `id`, `type` (`unit`, `join`, `field`, `denominator`, `time`, `aggregation`), `fields`, `required`, executable `params`, and `evidence`. Evidence names a public `source` (`request` or a dictionary key) and an exact `quote`. Unsupported clauses remain unresolved; producer assertions do not authorize them.

An anchor has an independently specified `anchor_plan` over source tables and a clause-specific `anchor_operation`. Keep the latter limited to the affected quantity so a failure localizes the property. Join parameters include dimension `table` and `keys`; denominator parameters include `entity_keys`; time parameters name `field`; field mutations can name dictionary-supported `alternative_fields`.

Plans use a restricted table language: filters, conversion, column derivation and renaming, joins with cardinality, half-open time windows, missing-value handling, deduplication, grouping and sorting. Consumers support sum, mean, median, extrema, standard deviation, count, unique count, entity means, ratios, Pearson/Spearman correlation, Shapiro normality and quantiles. Execution operates on copied DataFrames without arbitrary generated Python or network/filesystem primitives. Unsupported analyses produce an explicit failure trace.

Controlled mutations change one supported property. Answer differences against the supplied clean transformation determine harmfulness; unchanged answers remain negative controls. The audit companion records actual mutations and exclusions. Training uses only `train` tasks. You can instead provide independently audited examples with `source_id`, visible `features`, and seven binary `labels`: six harmful-type detection labels in the order above and one clean-rejection label. Answers, injection names and dataset IDs are excluded from features.

Seven logistic heads use equal total source-task weight and L2 regularization. A single-class head uses its observed constant. Select `--regularization` and the objective penalties on validation files. `configs/default.json` controls four selected probes, eight reserved executor seconds including recheck, one second per probe, the 12,000-token total cap and selection penalties.

## Execution and evaluation

```bash
handoffprobe run --tasks data/infi/tasks.json --scorer checkpoints/scorer.pkl --model Qwen/Qwen2.5-7B-Instruct --method handoffprobe --output runs/handoffprobe
handoffprobe evaluate --input runs/handoffprobe/traces.json --labels data/infi/labels.json --output runs/handoffprobe/metrics.json
handoffprobe plot --input runs/handoffprobe/metrics.json --output runs/figures
```

Use `--specifications` for predeclared contracts and plans, `--split validation` for selection, and `--revision` to pin the model. Seeds are 17, 29 and 43. The generator is frozen.

Source anchors detect absolute semantic errors. Permutation checks require an order-independent consumer. Partition recombination uses weighted sufficient statistics for sum/mean/count. Time fixtures check both half-open boundaries. Duplicate-dimension checks require source-key uniqueness. Unavailable replay, unsupported premises and reservation overruns return `unknown`; they do not establish a defect. Each execution runs in a separate worker process with a wall-time deadline; timeouts stop the worker. The ledger records worker CPU time, with 24 calls and 30 executor seconds per task. Required clauses without a passing check cause abstention.

`traces.json` stores tests, witnesses, repaired plans, unresolved clauses, raw responses, elapsed executor time and prompt/generated tokens by stage. Declaration, production, review, repair and reporting all enter the token ledger. `evaluate` computes task success, detection recall, false positives, precision, F1, localization, repair success, abstention and cost. For controlled metrics, add `--audit audit.json` containing `id`, `answer_impacted`, `type` and `fields`. Localization requires the correct clause type and affected field set.

DABench preserves `@field[value]` parsing and equality with numeric tolerance below 1e−6. Missing or abstained answers fail. DSBench's native evaluator uses a language-model judgment; `--judge-model Qwen/Qwen2.5-7B-Instruct` applies its clear-answer rubric locally. This changes the judge model and should be named with reported DSBench scores.

## Baselines and ablations

`run --method` accepts `none`, `schema`, `rules`, `producer_tests`, `self_check`, `semantic_review`, `fixed`, `random`, `contract_critic` and `full_library`. Model-review controls select from the same library. Full-library checking records its additional cost. Removals are `no_consumer`, `no_impact` (no learning), `no_applicability`, `no_coverage`, `no_witness`, `no_anchor` and `no_repair`. `single_agent` declares the producer and consumer together before selecting executable semantic checks.

The implementation uses an explicit table language for production and analysis, while the local model handles declarations, review controls, repair and reporting. The eligible natural-task cohort is defined by these supported operators. Source anchors remain necessary for constant or consistently wrong transformations because invariance alone does not establish the requested statistic.

## Controlled evaluation and repair accounting

Create evaluation controls from a separately declared held-out cohort. The contract stays fixed while one applicable producer operation changes. The exported runtime tasks contain no defect labels; specifications, scorer supervision and evaluator audit records are separate files.

```bash
handoffprobe build-specifications --tasks data/infi/tasks.json --split test --model Qwen/Qwen2.5-7B-Instruct --output data/test_specs.json
handoffprobe make-controls --tasks data/infi/tasks.json --specifications data/test_specs.json --split test --labels data/infi/labels.json --export-pack data/controls --output data/controls_examples.json
handoffprobe run --tasks data/controls/tasks.json --specifications data/controls/specifications.json --scorer checkpoints/scorer.pkl --model Qwen/Qwen2.5-7B-Instruct --output runs/controls
handoffprobe evaluate --input runs/controls/traces.json --labels data/controls/labels.json --audit data/controls/audit.json --output runs/controls/metrics.json
```

`train` rejects supervision from validation or test splits. For `no_consumer`, create training examples with `make-controls --no-consumer`, fit a separate scorer, then run with `--method no_consumer`. The no-witness control retains detection/localization records but withholds the witness from the repair prompt.

| Analysis | Computed metrics |
| --- | --- |
| Natural tasks, each dataset | Unconditional success, abstention, initial/final successes, recovered/damaged successes, repair attempts |
| Controlled defects | TP/FN/FP/TN, recall, false-positive rate, precision, F1, correct clause/field localization, successful repair |
| Error types | Counts and recall for units, joins, fields, denominators, time and aggregation |
| Cost and probe slots | Prompt/output tokens, worker CPU seconds and per-call deadlines; set `probe_slots` to 1–4 in a copied configuration |

The equation `final = initial + recovered - damaged` uses answers from the same task traces. Repair success requires a detected harmful artifact and a correct final answer, not merely a passing recheck. Full-library execution uses a separate 64-call/60-second allowance and remains a cost reference. Run seeds 17, 29 and 43 into separate directories; plots accept numerical evaluation outputs.
