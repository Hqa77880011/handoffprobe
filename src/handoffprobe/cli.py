import argparse
import copy
import json
import pickle
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .probes import (
    METHODS,
    TYPES,
    Scorer,
    candidates,
    execute,
    features,
    grounded_contract,
    select,
)
from .tables import consume, equal, inject, transform
from .executor import Executor
from .utils import load, plot_results, save


def jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def prepare(args):
    root, output = Path(args.input), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tasks, labels = [], []
    if args.source == "infi":
        data = root / "examples/DA-Agent/data"
        raw = jsonl(data / "da-dev-questions.jsonl")
        names = sorted({r["file_name"] for r in raw})
        random.Random(2027).shuffle(names)
        first, second = (
            (31, 41)
            if len(names) == 52
            else (int(0.6 * len(names)), int(0.8 * len(names)))
        )
        partitions = {
            name: "train" if i < first else "validation" if i < second else "test"
            for i, name in enumerate(names)
        }
        for row in raw:
            identifier = str(row["id"])
            tasks.append(
                {
                    "id": identifier,
                    "source_id": row["file_name"],
                    "split": partitions[row["file_name"]],
                    "dataset": "infi",
                    "request": row["question"]
                    + "\n"
                    + row.get("constraints", "")
                    + "\n"
                    + row.get("format", ""),
                    "tables": {
                        "source": str(
                            (data / "da-dev-tables" / row["file_name"]).resolve()
                        )
                    },
                    "dictionary": {},
                }
            )
        labels = [
            {"id": str(r["id"]), "answers": dict(r["common_answers"])}
            for r in jsonl(data / "da-dev-labels.jsonl")
        ]
    elif args.source == "dsbench":
        for challenge in jsonl(root / "data_analysis/data.json"):
            folder = root / "data_analysis/data" / challenge["id"]
            files = {
                p.name: str(p.resolve())
                for p in sorted(folder.iterdir())
                if p.suffix.lower() in (".csv", ".xlsx", ".xls")
            }
            for question, answer in zip(challenge["questions"], challenge["answers"]):
                text_path = folder / (question + ".txt")
                if not text_path.exists():
                    raise FileNotFoundError(
                        f"Extract the DSBench analysis data pack: {text_path}"
                    )
                identifier = challenge["id"] + ":" + question
                tasks.append(
                    {
                        "id": identifier,
                        "source_id": challenge["id"],
                        "split": "test",
                        "dataset": "dsbench",
                        "request": text_path.read_text(encoding="utf-8"),
                        "tables": files,
                        "dictionary": {},
                    }
                )
                labels.append({"id": identifier, "answer": answer})
    else:
        tasks = load(root)
        for task in tasks:
            if any(k in task for k in ("answers", "harmful", "gold")):
                raise ValueError("Keep evaluator labels in a separate file")
            task["tables"] = {
                name: str((root.parent / path).resolve())
                for name, path in task["tables"].items()
            }
    save(output / "tasks.json", tasks)
    if labels:
        save(output / "labels.json", labels)


def read_tables(task):
    tables = {}
    for name, path in task["tables"].items():
        frame = (
            pd.read_excel(path)
            if Path(path).suffix.lower() in (".xlsx", ".xls")
            else pd.read_csv(path)
        )
        frame["__source_id"] = np.arange(len(frame))
        tables[name] = frame
    if not tables:
        raise ValueError("No CSV tables for this analysis task")
    return tables


def metadata(tables):
    return {
        name: {
            "columns": {k: str(v) for k, v in frame.dtypes.items()},
            "rows": len(frame),
            "sample": frame.head(12).to_json(orient="records"),
        }
        for name, frame in tables.items()
    }


def schema_prompt():
    return "A transformation plan is {source:table_name,operations:[...]}. Allowed operations: filter(conditions with field/operator/value; eq,ne,lt,le,gt,ge,in,notna), convert(field,output,factor,offset), rename(mapping), derive(fields,output,operator:add|subtract|multiply|divide), join(table,keys,how,cardinality), time_window(field,start,end,timezone), drop_missing(fields), fill_missing(values), deduplicate(keys), group(keys,aggregations:{output:{field,operator}}), sort(keys,ascending). A consumer is {outputs:[{name,operator,field,round,conditions,...}]}. Operators: sum,mean,median,min,max,std(ddof),count,unique,entity_mean(entity),ratio(entity),pearson(fields,statistic),spearman(fields,statistic),normality(alpha),quantile(q),value(index). Use only needed keys."


def generate(task, tables, llm, single=False):
    public = {
        "request": task["request"],
        **{str(k): str(v) for k, v in task.get("dictionary", {}).items()},
    }
    details = json.dumps(metadata(tables), ensure_ascii=False)
    if single:
        specification = llm.json(
            "As one data-analysis agent, provide a transformation, consumer operation and source-grounded semantic contract. "
            + schema_prompt()
            + " Return {plan,consumer,contract:{clauses:[{id,type,fields,params,required,evidence:[{source,quote}]}]}}. Clause types: unit,join,field,denominator,time,aggregation. Quote exact request/dictionary text."
            + "\nPUBLIC="
            + json.dumps(public)
            + "\nTABLES="
            + details,
            "single_agent",
            3000,
        )
        return (
            grounded_contract(specification["contract"], public),
            specification["plan"],
            specification["consumer"],
        )
    declared = llm.json(
        "Declare the downstream operation and a semantic contract for this request. "
        + schema_prompt()
        + " Contract: {clauses:[{id,type:unit|join|field|denominator|time|aggregation,fields:[...],required:true,params:{anchor_plan,anchor_operation,...},evidence:[{source,quote}]}]}. Every quote must be an exact nonempty substring of request or a dictionary entry. Only these sources authorize a clause; table samples and producer guesses do not. Ground anchor recipes in these meanings. Include only supported clauses; preserve missing-value rules. Return {consumer,contract}.\nPUBLIC="
        + json.dumps(public)
        + "\nTABLES="
        + details,
        "contract",
        1800,
    )
    producer = llm.json(
        "Produce the table transformation for the declared consumer. "
        + schema_prompt()
        + " Return {plan}.\nPUBLIC="
        + json.dumps(public)
        + "\nTABLES="
        + details
        + "\nCONSUMER="
        + json.dumps(declared["consumer"]),
        "producer",
        1200,
    )
    return (
        grounded_contract(declared["contract"], public),
        producer["plan"],
        declared["consumer"],
    )


def protocol(
    task, tables, contract, plan, operation, scorer, llm, cfg, method, executor
):
    started = time.perf_counter()
    artifact = executor.call("tables", "transform", (tables, plan))
    initial_answer = executor.call("tables", "consume", (artifact, operation))
    initial_response = " ".join(
        "@" + key + "[" + str(value) + "]" for key, value in initial_answer.items()
    )
    library = candidates(contract)
    if method == "no_anchor":
        library = [t for t in library if t["variant"] != "anchor"]
    vectors = [
        features(t, artifact, operation, method != "no_consumer") for t in library
    ]
    prediction = scorer.predict(vectors) if vectors else np.empty((0, 7))
    if method == "none":
        selected = []
    elif method == "schema":
        absent = sorted(
            {f for c in contract["clauses"] for f in c["fields"]}
            - set(artifact.columns)
        )
        selected = []
        if absent:
            return {
                "abstained": True,
                "answer": {},
                "tests": [],
                "repair": False,
                "error": "Missing fields: " + str(absent),
                "cpu_seconds": sum(r["seconds"] for r in executor.ledger),
                "initial_response": initial_response,
                "repair_attempted": False,
            }
    elif method in ("rules", "full_library"):
        selected = (
            library if method == "full_library" else library[: cfg["probe_slots"]]
        )
    elif method == "fixed":
        selected = library[: cfg["probe_slots"]]
    elif method == "random":
        selected = random.Random(cfg["seed"]).sample(
            library, min(len(library), cfg["probe_slots"])
        )
    elif method == "no_impact":
        neutral = np.zeros_like(prediction)
        for i, test in enumerate(library):
            neutral[i, TYPES.index(test["clause"]["type"])] = 1
        selected = select(
            library,
            neutral,
            cfg["probe_slots"],
            cfg["probe_seconds"],
            cfg["lambda"],
            cfg["mu"],
        )
    elif method in (
        "single_agent",
        "producer_tests",
        "self_check",
        "semantic_review",
        "contract_critic",
    ):
        if llm is None:
            raise ValueError("This baseline requires a local language model")
        description = [
            {"id": t["id"], "type": t["clause"]["type"], "variant": t["variant"]}
            for t in library
        ]
        chosen = llm.json(
            "Choose up to "
            + str(cfg["probe_slots"])
            + " executable probes for this downstream task using "
            + method
            + " review. Return {test_ids:[...]}.\nREQUEST="
            + task["request"]
            + "\nCONTRACT="
            + json.dumps(contract)
            + "\nPLAN="
            + json.dumps(plan)
            + "\nCANDIDATES="
            + json.dumps(description),
            "review",
            512,
        )
        selected = [t for t in library if t["id"] in chosen["test_ids"]][
            : cfg["probe_slots"]
        ]
    else:
        selected = select(
            library,
            prediction,
            cfg["probe_slots"],
            cfg["probe_seconds"],
            cfg["lambda"],
            cfg["mu"],
            method != "no_coverage",
        )

    def check(test, candidate, producer):
        try:
            return executor.call(
                "probes",
                "execute",
                (
                    test,
                    tables,
                    candidate,
                    producer,
                    operation,
                    cfg["per_probe_seconds"],
                    method != "no_applicability",
                ),
                cfg["per_probe_seconds"],
            )
        except (TimeoutError, RuntimeError) as error:
            clause = test["clause"]
            return {
                "test_id": test["id"],
                "clause_id": clause["id"],
                "type": clause["type"],
                "fields": clause["fields"],
                "verdict": "unknown",
                "reason": str(error),
                "rows": [],
            }

    outcomes = [check(t, artifact, plan) for t in selected]
    failure = [r for r in outcomes if r["verdict"] == "fail"]
    repaired, repair_error = False, None
    repair_attempted = bool(failure and method != "no_repair" and llm is not None)
    if failure and method != "no_repair" and llm is not None:
        localized = sorted(failure, key=lambda r: (len(r["fields"]), r["clause_id"]))[0]
        try:
            result = llm.json(
                "Repair the producer transformation once using the leading witness. Preserve source data, consumer and frozen contract. "
                + schema_prompt()
                + " Return {plan}.\nREQUEST="
                + task["request"]
                + "\nPLAN="
                + json.dumps(plan)
                + "\nCONTRACT="
                + json.dumps(contract)
                + "\nWITNESS="
                + json.dumps(
                    {"verdict": "fail"} if method == "no_witness" else localized,
                    default=str,
                ),
                "repair",
                1400,
            )
            replacement = result["plan"]
            artifact = executor.call(
                "tables", "transform", (tables, replacement), cfg["repair_seconds"]
            )
            recheck = [check(t, artifact, replacement) for t in selected]
            repaired = not any(r["verdict"] == "fail" for r in recheck)
            outcomes.extend(dict(r, recheck=True) for r in recheck)
            plan = replacement
        except (ValueError, KeyError, TypeError, TimeoutError, RuntimeError) as error:
            repair_error = str(error)
    current = [r for r in outcomes if r.get("recheck")] or outcomes
    required = {c["id"] for c in contract["clauses"] if c.get("required", True)}
    passed = {r["clause_id"] for r in current if r["verdict"] == "pass"}
    uncertain = required - passed
    rejected = any(r["verdict"] == "fail" for r in current)
    abstained = rejected or (bool(uncertain) and method not in ("none", "schema"))
    answer = (
        {} if abstained else executor.call("tables", "consume", (artifact, operation))
    )
    if llm is not None and not abstained:
        report = llm.chat(
            "Report only the requested answer in its original requested format using these computed results. Do not change numbers.\nREQUEST="
            + task["request"]
            + "\nRESULTS="
            + json.dumps(answer, default=str),
            "report",
            512,
            temperature=0,
        )
    else:
        report = " ".join(
            "@" + key + "[" + str(value) + "]" for key, value in answer.items()
        )
    return {
        "abstained": abstained,
        "answer": answer,
        "response": report,
        "tests": outcomes,
        "repair": repaired,
        "repair_error": repair_error,
        "unresolved": sorted(uncertain),
        "plan": plan,
        "cpu_seconds": sum(r["seconds"] for r in executor.ledger),
        "wall_seconds": time.perf_counter() - started,
        "executor_ledger": executor.ledger,
        "initial_response": initial_response,
        "initial_answer": initial_answer,
        "repair_attempted": repair_attempted,
        "selected": [t["id"] for t in selected],
    }


def collect_training(args):
    tasks = [t for t in load(args.tasks) if t["split"] == args.split]
    evaluator_labels = (
        {str(r["id"]): r for r in load(args.labels)} if args.labels else {}
    )
    exported_tasks, exported_specs, exported_labels = [], [], []
    specs = {r["id"]: r for r in load(args.specifications)}
    examples, audit = [], []
    for task in tasks:
        spec = specs[task["id"]]
        tables = read_tables(task)
        public = {"request": task["request"], **task.get("dictionary", {})}
        contract = grounded_contract(spec["contract"], public)
        plan, operation = spec["plan"], spec["consumer"]
        clean = transform(tables, plan)
        reference = consume(clean, operation)
        variants = [("clean", None, plan)]
        for clause in contract["clauses"]:
            if not clause["supported"]:
                continue
            try:
                changed = inject(plan, clause, clause["type"])
            except ValueError:
                continue
            variants.append((clause["id"], clause["type"], changed))
        for name, harmful_type, modified in variants:
            try:
                artifact = transform(tables, modified)
                impacted = not equal(consume(artifact, operation), reference)
            except (ValueError, KeyError, pd.errors.MergeError) as error:
                audit.append(
                    {"id": task["id"], "variant": name, "excluded": str(error)}
                )
                continue
            for test in candidates(contract):
                result = execute(test, tables, artifact, modified, operation)
                fail = result["verdict"] == "fail"
                labels = [
                    int(fail and impacted and harmful_type == kind) for kind in TYPES
                ] + [int(fail and not impacted)]
                examples.append(
                    {
                        "source_id": task["source_id"],
                        "features": features(
                            test, artifact, operation, not args.no_consumer
                        ),
                        "labels": labels,
                        "split": task["split"],
                    }
                )
            identifier = task["id"] + "::" + name
            fields = next(
                (c["fields"] for c in contract["clauses"] if c["id"] == name), []
            )
            audit.append(
                {
                    "id": identifier,
                    "source_id": task["source_id"],
                    "variant": name,
                    "type": harmful_type or "clean",
                    "fields": fields,
                    "answer_impacted": impacted,
                }
            )
            exported_tasks.append(dict(task, id=identifier))
            exported_specs.append(
                {
                    "id": identifier,
                    "contract": contract,
                    "consumer": operation,
                    "plan": modified,
                }
            )
            label = evaluator_labels.get(task["id"], {"answers": reference})
            exported_labels.append(dict(label, id=identifier))
    save(args.output, examples)
    save(Path(args.output).with_suffix(".audit.json"), audit)
    if args.export_pack:
        folder = Path(args.export_pack)
        save(folder / "tasks.json", exported_tasks)
        save(folder / "specifications.json", exported_specs)
        save(folder / "labels.json", exported_labels)
        save(folder / "audit.json", [r for r in audit if "excluded" not in r])


def run(args):
    cfg = load(args.config)
    with open(args.scorer, "rb") as handle:
        scorer = pickle.load(handle)
    llm = None
    if args.model:
        from .llm import LocalModel

        llm = LocalModel(
            args.model,
            args.revision,
            seed=args.seed,
            max_generated=cfg["token_cap"],
            max_total=cfg["token_cap"],
        )
    cfg["seed"] = args.seed
    specs = (
        {r["id"]: r for r in load(args.specifications)} if args.specifications else {}
    )
    results = []
    for task in load(args.tasks):
        if task["split"] != args.split:
            continue
        if llm:
            llm.reset()
        tables = read_tables(task)
        executor = None
        try:
            if task["id"] in specs:
                spec = specs[task["id"]]
                contract = grounded_contract(
                    spec["contract"],
                    {"request": task["request"], **task.get("dictionary", {})},
                )
                plan, operation = spec["plan"], spec["consumer"]
            elif llm:
                contract, plan, operation = generate(
                    task, tables, llm, args.method == "single_agent"
                )
            else:
                raise ValueError("Supply --specifications or --model")
            calls = cfg["tool_cap"] if args.method != "full_library" else 64
            seconds = cfg["executor_seconds"] if args.method != "full_library" else 60
            with Executor(seconds, calls) as executor:
                result = protocol(
                    task,
                    tables,
                    contract,
                    plan,
                    operation,
                    scorer,
                    llm,
                    cfg,
                    args.method,
                    executor,
                )
        except (
            ValueError,
            KeyError,
            pd.errors.MergeError,
            RuntimeError,
            TimeoutError,
        ) as error:
            result = {
                "abstained": True,
                "answer": {},
                "response": "",
                "tests": [],
                "repair": False,
                "error": str(error),
                "cpu_seconds": sum(r["seconds"] for r in executor.ledger)
                if executor
                else 0,
                "executor_ledger": executor.ledger if executor else [],
            }
        results.append(
            dict(
                result,
                id=task["id"],
                source_id=task["source_id"],
                dataset=task["dataset"],
                method=args.method,
                seed=args.seed,
                ledger=copy.deepcopy(llm.ledger) if llm else [],
            )
        )
        save(Path(args.output) / "traces.json", results)


def answer_equal(predicted, reference):
    if not reference:
        return False
    for field, value in reference.items():
        actual = predicted.get(field)
        if str(actual) == str(value):
            continue
        try:
            if abs(float(actual) - float(value)) < 1e-6:
                continue
        except (ValueError, TypeError):
            return False
        return False
    return True


def evaluate(args):
    traces = load(args.input)
    labels = {str(r["id"]): r for r in load(args.labels)}
    audit = {str(r["id"]): r for r in load(args.audit)} if args.audit else {}
    judge = None
    if args.judge_model:
        from .llm import LocalModel

        judge = LocalModel(args.judge_model, max_generated=512, max_total=16384)
    summary = []
    for method in sorted({r["method"] for r in traces}):
        rows = [r for r in traces if r["method"] == method]
        scores, confusion, localized, repaired = [], np.zeros(4, dtype=int), [], []
        initial_scores = []
        datasets = defaultdict(list)
        categories = defaultdict(lambda: np.zeros(4, dtype=int))
        for row in rows:
            label = labels[row["id"]]
            predicted = dict(re.findall(r"@(\w+)\[(.*?)\]", row.get("response", "")))
            if "answers" in label:
                correctness = []
                for field, reference in label["answers"].items():
                    value = predicted.get(field)
                    same = value == reference
                    if not same:
                        try:
                            same = abs(float(value) - float(reference)) < 1e-6
                        except (ValueError, TypeError):
                            same = False
                    correctness.append(same)
                score = bool(correctness) and all(correctness) and not row["abstained"]
            elif judge:
                judge.reset()
                response = judge.chat(
                    "Judge whether the generated answer gives a clear correct answer rather than only a calculation process. Return True or False only.\nREFERENCE="
                    + str(label["answer"])
                    + "\nPREDICTION="
                    + row.get("response", ""),
                    "sealed_evaluation",
                    8,
                    temperature=0,
                )
                score = response.strip().lower() == "true" and not row["abstained"]
            else:
                raise ValueError(
                    "DSBench evaluation requires --judge-model using the native answer-judgment rubric"
                )
            scores.append(score)
            datasets[row["dataset"]].append(score)
            if "initial_response" in row:
                initial = dict(re.findall(r"@(\w+)\[(.*?)\]", row["initial_response"]))
                if "answers" in label:
                    initial_scores.append(answer_equal(initial, label["answers"]))
                elif judge:
                    judge.reset()
                    response = judge.chat(
                        "Return True if the candidate correctly answers the reference, else False.\nREFERENCE="
                        + str(label["answer"])
                        + "\nCANDIDATE="
                        + row["initial_response"],
                        "sealed_initial_evaluation",
                        8,
                        temperature=0,
                    )
                    initial_scores.append(response.strip().lower() == "true")
            else:
                initial_scores.append(False)
            if row["id"] in audit:
                gold = audit[row["id"]]
                detected = any(
                    t["verdict"] == "fail" and not t.get("recheck")
                    for t in row["tests"]
                )
                harmful = bool(gold["answer_impacted"])
                index = (
                    0
                    if harmful and detected
                    else 1
                    if harmful
                    else 2
                    if detected
                    else 3
                )
                confusion[index] += 1
                categories[gold.get("type", "clean")][index] += 1
                if index == 0:
                    failures = [
                        t
                        for t in row["tests"]
                        if t["verdict"] == "fail" and not t.get("recheck")
                    ]
                    leading = min(
                        failures, key=lambda t: (len(t["fields"]), t["clause_id"])
                    )
                    localized.append(
                        leading["type"] == gold["type"]
                        and set(leading["fields"]) == set(gold["fields"])
                    )
                    repaired.append(row["repair"] and score)
        tp, fn, fp, tn = map(int, confusion)
        summary.append(
            {
                "method": method,
                "task_success_percent": 100 * float(np.mean(scores)),
                "initial_successes": sum(initial_scores),
                "recovered_successes": sum(
                    not a and b for a, b in zip(initial_scores, scores)
                ),
                "damaged_successes": sum(
                    a and not b for a, b in zip(initial_scores, scores)
                ),
                "final_successes": sum(scores),
                "repair_attempts": sum(r.get("repair_attempted", False) for r in rows),
                "category_recall": {
                    k: float(v[0] / max(v[0] + v[1], 1)) for k, v in categories.items()
                },
                **{
                    k + "_success_percent": 100 * float(np.mean(v))
                    for k, v in datasets.items()
                },
                "detection_recall": tp / max(tp + fn, 1) if audit else None,
                "false_positive_rate": fp / max(fp + tn, 1) if audit else None,
                "precision": tp / max(tp + fp, 1) if audit else None,
                "f1": 2 * tp / max(2 * tp + fp + fn, 1) if audit else None,
                "localization_accuracy": float(np.mean(localized))
                if localized
                else None,
                "repair_success": float(np.mean(repaired)) if repaired else None,
                "abstention_fraction": float(np.mean([r["abstained"] for r in rows])),
                "tokens": float(
                    np.mean(
                        [
                            sum(
                                t["prompt_tokens"] + t["generated_tokens"]
                                for t in r["ledger"]
                            )
                            for r in rows
                        ]
                    )
                ),
                "executor_seconds": float(np.mean([r["cpu_seconds"] for r in rows])),
                "category_counts": {k: v.tolist() for k, v in categories.items()},
                "counts": {"tp": tp, "fn": fn, "fp": fp, "tn": tn} if audit else None,
            }
        )
    save(args.output, {"summary": summary})


def main():
    parser = argparse.ArgumentParser(prog="handoffprobe")
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument(
        "--source", choices=["infi", "dsbench", "manifest"], required=True
    )
    prep.add_argument("--input", required=True)
    prep.add_argument("--output", required=True)
    controls = sub.add_parser("make-controls")
    controls.add_argument("--tasks", required=True)
    controls.add_argument("--specifications", required=True)
    controls.add_argument("--output", required=True)
    controls.add_argument(
        "--split", choices=["train", "validation", "test"], default="train"
    )
    controls.add_argument("--labels")
    controls.add_argument("--export-pack")
    controls.add_argument("--no-consumer", action="store_true")
    specifications = sub.add_parser("build-specifications")
    specifications.add_argument("--tasks", required=True)
    specifications.add_argument(
        "--split", choices=["train", "validation", "test"], default="train"
    )
    specifications.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    specifications.add_argument("--revision", default="main")
    specifications.add_argument("--output", required=True)
    train = sub.add_parser("train")
    train.add_argument("--examples", required=True)
    train.add_argument("--regularization", type=float, default=1)
    train.add_argument("--output", required=True)
    execution = sub.add_parser("run")
    execution.add_argument("--tasks", required=True)
    execution.add_argument("--scorer", required=True)
    execution.add_argument("--specifications")
    execution.add_argument("--model")
    execution.add_argument("--revision", default="main")
    execution.add_argument("--seed", type=int, default=17)
    execution.add_argument(
        "--split", choices=["train", "validation", "test"], default="test"
    )
    execution.add_argument("--method", choices=METHODS, default="handoffprobe")
    execution.add_argument("--config", default="configs/default.json")
    execution.add_argument("--output", required=True)
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--input", required=True)
    evaluation.add_argument("--labels", required=True)
    evaluation.add_argument("--audit")
    evaluation.add_argument("--judge-model")
    evaluation.add_argument("--output", required=True)
    plot = sub.add_parser("plot")
    plot.add_argument("--input", required=True)
    plot.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "make-controls":
        collect_training(args)
    elif args.command == "build-specifications":
        from .llm import LocalModel

        model = LocalModel(
            args.model, args.revision, max_generated=4000, max_total=12000
        )
        specifications = []
        for task in load(args.tasks):
            if task["split"] != args.split:
                continue
            model.reset()
            contract, plan, consumer = generate(task, read_tables(task), model)
            specifications.append(
                {
                    "id": task["id"],
                    "contract": contract,
                    "plan": plan,
                    "consumer": consumer,
                }
            )
            save(args.output, specifications)
    elif args.command == "train":
        examples = load(args.examples)
        if not examples:
            raise ValueError("No supervised probe examples")
        if any(r.get("split", "train") != "train" for r in examples):
            raise ValueError("Only training examples may fit the scorer")
        scorer = Scorer().fit(examples, args.regularization)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "wb") as handle:
            pickle.dump(scorer, handle)
    elif args.command == "run":
        run(args)
    elif args.command == "evaluate":
        evaluate(args)
    else:
        plot_results(args.input, args.output)


if __name__ == "__main__":
    main()
