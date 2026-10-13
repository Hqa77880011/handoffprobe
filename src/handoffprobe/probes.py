import copy
import time

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from .tables import consume, equal, transform


TYPES = ("unit", "join", "field", "denominator", "time", "aggregation")
METHODS = (
    "handoffprobe",
    "none",
    "schema",
    "rules",
    "producer_tests",
    "self_check",
    "semantic_review",
    "fixed",
    "random",
    "contract_critic",
    "full_library",
    "no_impact",
    "no_coverage",
    "no_anchor",
    "no_repair",
    "no_consumer",
    "no_applicability",
    "no_witness",
    "single_agent",
)


def grounded_contract(contract, public):
    validated = copy.deepcopy(contract)
    seen = set()
    for clause in validated["clauses"]:
        if clause["id"] in seen or clause["type"] not in TYPES:
            raise ValueError("Invalid or duplicated semantic clause")
        seen.add(clause["id"])
        evidence = clause.get("evidence", [])
        supported = bool(evidence) and all(
            e.get("source") in public
            and e.get("quote")
            and e["quote"] in public[e["source"]]
            for e in evidence
        )
        clause["supported"] = supported
    return validated


def candidates(contract):
    result = []
    for clause in contract["clauses"]:
        if not clause["supported"]:
            continue
        variants = (
            ["anchor", "permutation", "partition"]
            if clause["type"] == "aggregation"
            else ["anchor", "permutation"]
        )
        if clause["type"] == "time":
            variants.append("boundary")
        if clause["type"] == "join":
            variants.append("duplicate_dimension")
        for variant in variants[:4]:
            result.append(
                {
                    "id": clause["id"] + ":" + variant,
                    "clause": clause,
                    "variant": variant,
                    "cost": 2.0,
                }
            )
    counts = {kind: 0 for kind in TYPES}
    bounded = []
    for test in result:
        kind = test["clause"]["type"]
        if counts[kind] < 4:
            bounded.append(test)
            counts[kind] += 1
    return bounded


def features(test, artifact, operation, consumer=True):
    clause = test["clause"]
    values = {
        "type_" + clause["type"]: 1.0,
        "variant_" + test["variant"]: 1.0,
        "fields": len(clause["fields"]),
        "field_overlap": len(set(clause["fields"]) & set(artifact.columns))
        / max(len(clause["fields"]), 1),
        "missing_fraction": float(artifact.isna().mean().mean())
        if len(artifact.columns)
        else 0.0,
        "log_rows": float(np.log1p(len(artifact))),
        "required": float(clause.get("required", True)),
    }
    for output in operation["outputs"] if consumer else []:
        values["consumer_" + output["operator"]] = 1.0
    keys = clause["params"].get("keys", clause["params"].get("entity_keys", []))
    values["key_multiplicity"] = (
        float(artifact.duplicated(keys).mean())
        if keys and set(keys) <= set(artifact.columns) and len(artifact)
        else 0.0
    )
    return values


def execute(test, tables, artifact, plan, operation, timeout=1, applicability=True):
    start = time.perf_counter()
    clause, variant = test["clause"], test["variant"]
    result = {
        "test_id": test["id"],
        "clause_id": clause["id"],
        "type": clause["type"],
        "fields": clause["fields"],
        "verdict": "unknown",
        "reason": None,
        "rows": [],
    }
    params = clause["params"]
    try:
        if not set(clause["fields"]) <= set(artifact.columns):
            result.update(
                verdict="fail",
                reason="Required field missing",
                expected=clause["fields"],
                observed=list(artifact.columns),
            )
        elif variant == "anchor":
            anchor_plan = params.get("anchor_plan")
            if anchor_plan is None:
                result["reason"] = "No source-grounded anchor recipe"
            else:
                expected_frame = transform(tables, anchor_plan)
                anchor_operation = params.get("anchor_operation", operation)
                expected = consume(expected_frame, anchor_operation)
                observed = consume(artifact, anchor_operation)
                result.update(
                    verdict="pass" if equal(expected, observed) else "fail",
                    expected=expected,
                    observed=observed,
                    reason="Independent source anchor",
                    rows=artifact.get("__source_id", artifact.index.to_series())
                    .head(32)
                    .tolist(),
                )
        elif plan is None:
            result["reason"] = "Replay unavailable"
        elif variant == "permutation":
            independent = all(
                o["operator"] not in ("value",) for o in operation["outputs"]
            )
            if applicability and (
                not independent
                or any(
                    op["op"] in ("sort", "deduplicate")
                    for op in plan.get("operations", [])
                )
            ):
                result["reason"] = "Order independence premise unsupported"
            else:
                permuted = {
                    key: frame.sample(frac=1, random_state=17)
                    for key, frame in tables.items()
                }
                expected, observed = (
                    consume(artifact, operation),
                    consume(transform(permuted, plan), operation),
                )
                result.update(
                    verdict="pass" if equal(expected, observed) else "fail",
                    expected=expected,
                    observed=observed,
                    reason="Permutation invariance",
                )
        elif variant == "partition":
            if applicability and any(
                o["operator"] not in ("sum", "mean", "count")
                or o.get("conditions")
                or not o.get("skipna", True)
                for o in operation["outputs"]
            ):
                result["reason"] = "Sufficient-statistic recombination not applicable"
            else:
                observed = consume(artifact, operation)
                expected = {}
                partitions = [artifact.iloc[::2], artifact.iloc[1::2]]
                for item in operation["outputs"]:
                    field, kind = item.get("field"), item["operator"]
                    totals = [
                        float(part[field].sum()) if field else float(len(part))
                        for part in partitions
                    ]
                    number = sum(
                        int(part[field].count()) if field else len(part)
                        for part in partitions
                    )
                    expected[item["name"]] = (
                        sum(totals) / max(number, 1)
                        if kind == "mean"
                        else number
                        if kind == "count"
                        else sum(totals)
                    )
                    if "round" in item:
                        expected[item["name"]] = round(
                            expected[item["name"]], item["round"]
                        )
                result.update(
                    verdict="pass" if equal(expected, observed) else "fail",
                    expected=expected,
                    observed=observed,
                    reason="Weighted sufficient-statistic recombination",
                )
        elif variant == "duplicate_dimension":
            table = params.get("table")
            if table not in tables or not params.get("keys") or not len(tables[table]):
                result["reason"] = "Unique dimension premise unavailable"
            else:
                import pandas as pd

                if applicability and tables[table].duplicated(params["keys"]).any():
                    result["reason"] = "Dimension is not unique on declared keys"
                else:
                    altered = {k: v.copy() for k, v in tables.items()}
                    altered[table] = pd.concat(
                        [altered[table], altered[table].iloc[:1]], ignore_index=True
                    )
                    try:
                        transform(altered, plan)
                        result.update(
                            verdict="fail",
                            reason="Invalid dimension multiplicity was accepted",
                        )
                    except pd.errors.MergeError:
                        result.update(
                            verdict="pass",
                            reason="Invalid dimension multiplicity rejected",
                        )
        elif variant == "boundary":
            import pandas as pd

            field = params.get("field")
            window = next(
                (
                    o
                    for o in plan["operations"]
                    if o["op"] == "time_window" and o["field"] == field
                ),
                None,
            )
            if window is None or not len(tables[plan["source"]]):
                result["reason"] = "No declared temporal replay"
            else:
                fixtures = tables[plan["source"]].iloc[[0, 0, 0, 0]].copy()
                begin, end = pd.Timestamp(window["start"]), pd.Timestamp(window["end"])
                fixtures[field] = [
                    str(begin - pd.Timedelta(seconds=1)),
                    str(begin),
                    str(end - pd.Timedelta(seconds=1)),
                    str(end),
                ]
                fixtures["__fixture_id"] = [0, 1, 2, 3]
                fixture_tables = {k: v.copy() for k, v in tables.items()}
                fixture_tables[plan["source"]] = fixtures
                actual = transform(fixture_tables, plan)
                if "__fixture_id" not in actual:
                    result["reason"] = "Transformation removes fixture identity"
                else:
                    found = sorted(actual["__fixture_id"].unique().tolist())
                    result.update(
                        verdict="pass" if found == [1, 2] else "fail",
                        expected=[1, 2],
                        observed=found,
                        reason="Half-open window boundary fixtures",
                    )
    except (KeyError, ValueError, TypeError, ZeroDivisionError) as error:
        result.update(
            verdict="unknown", reason=type(error).__name__ + ": " + str(error)
        )
    result["seconds"] = time.perf_counter() - start
    if result["seconds"] > timeout:
        result.update(verdict="unknown", reason="Execution reservation exceeded")
    return result


class Scorer:
    def __init__(self):
        self.vectorizer = DictVectorizer(sparse=False)
        self.heads = []

    def fit(self, rows, regularization=1):
        x = self.vectorizer.fit_transform([r["features"] for r in rows])
        y = np.asarray([r["labels"] for r in rows], dtype=int)
        tasks = [r["source_id"] for r in rows]
        weights = np.asarray([1 / tasks.count(t) for t in tasks])
        for k in range(7):
            if len(np.unique(y[:, k])) == 1:
                self.heads.append(float(np.average(y[:, k], weights=weights)))
            else:
                self.heads.append(
                    LogisticRegression(C=regularization, max_iter=1000).fit(
                        x, y[:, k], sample_weight=weights
                    )
                )
        return self

    def predict(self, values):
        x = self.vectorizer.transform(values)
        return np.stack(
            [
                np.full(len(x), head)
                if isinstance(head, float)
                else head.predict_proba(x)[:, 1]
                for head in self.heads
            ],
            1,
        )


def select(
    tests,
    predictions,
    slots=4,
    seconds=8,
    penalty=0.5,
    cost_penalty=0.05,
    coverage=True,
):
    supported = {test["clause"]["type"] for test in tests}
    weights = (
        np.asarray([1 / len(supported) if kind in supported else 0 for kind in TYPES])
        if supported
        else np.zeros(6)
    )
    chosen, maximum, spent = [], np.zeros(6), 0.0
    while len(chosen) < slots:
        best, gain = None, 0.0
        for i, test in enumerate(tests):
            if i in chosen or spent + test["cost"] > seconds:
                continue
            detection = predictions[i, :6]
            marginal = np.maximum(0, detection - maximum) if coverage else detection
            improvement = float(
                weights @ marginal
                - penalty * predictions[i, 6]
                - cost_penalty * test["cost"] / seconds
            )
            density = improvement / test["cost"]
            if density > gain:
                best, gain = i, density
        if best is None:
            break
        chosen.append(best)
        maximum = np.maximum(maximum, predictions[best, :6])
        spent += tests[best]["cost"]
    return [tests[i] for i in chosen]
