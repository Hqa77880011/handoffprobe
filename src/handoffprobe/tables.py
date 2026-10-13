import copy

import numpy as np
import pandas as pd
from scipy import stats


def filter_rows(frame, conditions):
    selected = pd.Series(True, index=frame.index)
    for condition in conditions:
        values = frame[condition["field"]]
        operator, target = condition["operator"], condition.get("value")
        relations = {
            "eq": lambda: values == target,
            "ne": lambda: values != target,
            "lt": lambda: values < target,
            "le": lambda: values <= target,
            "gt": lambda: values > target,
            "ge": lambda: values >= target,
            "in": lambda: values.isin(target),
            "notna": values.notna,
        }
        if operator not in relations:
            raise ValueError("Unsupported filter operator: " + operator)
        selected &= relations[operator]()
    return frame.loc[selected].copy()


def transform(tables, plan):
    frame = tables[plan["source"]].copy(deep=True)
    if "__source_id" not in frame:
        frame["__source_id"] = np.arange(len(frame))
    for operation in plan.get("operations", []):
        kind = operation["op"]
        if kind == "filter":
            frame = filter_rows(frame, operation["conditions"])
        elif kind == "convert":
            frame[operation["output"]] = pd.to_numeric(
                frame[operation["field"]], errors="raise"
            ) * float(operation["factor"]) + float(operation.get("offset", 0))
        elif kind == "rename":
            frame = frame.rename(columns=operation["mapping"])
        elif kind == "derive":
            a = frame[operation["fields"][0]]
            values = [frame[f] for f in operation["fields"][1:]]
            for b in values:
                if operation["operator"] == "add":
                    a = a + b
                elif operation["operator"] == "subtract":
                    a = a - b
                elif operation["operator"] == "multiply":
                    a = a * b
                elif operation["operator"] == "divide":
                    a = a / b.replace(0, np.nan)
                else:
                    raise ValueError("Unsupported derived operator")
            frame[operation["output"]] = a
        elif kind == "join":
            frame = frame.merge(
                tables[operation["table"]],
                on=operation["keys"],
                how=operation.get("how", "left"),
                validate=operation.get("cardinality"),
            )
        elif kind == "time_window":
            dates = pd.to_datetime(frame[operation["field"]], errors="raise")
            dates = (
                dates.dt.tz_localize(
                    operation["timezone"], ambiguous="raise", nonexistent="raise"
                )
                if dates.dt.tz is None
                else dates.dt.tz_convert(operation["timezone"])
            )
            start, end = (
                pd.Timestamp(operation["start"]),
                pd.Timestamp(operation["end"]),
            )
            if start.tzinfo is None:
                start = start.tz_localize(operation["timezone"])
                end = end.tz_localize(operation["timezone"])
            frame = frame.loc[(dates >= start) & (dates < end)].copy()
        elif kind == "drop_missing":
            frame = frame.dropna(subset=operation["fields"])
        elif kind == "fill_missing":
            frame = frame.fillna(operation["values"])
        elif kind == "deduplicate":
            frame = frame.drop_duplicates(
                subset=operation["keys"], keep=operation.get("keep", "first")
            )
        elif kind == "group":
            frame = (
                frame.groupby(operation["keys"], dropna=operation.get("dropna", False))
                .agg(
                    **{
                        name: (spec["field"], spec["operator"])
                        for name, spec in operation["aggregations"].items()
                    }
                )
                .reset_index()
            )
        elif kind == "sort":
            frame = frame.sort_values(
                operation["keys"], ascending=operation.get("ascending", True)
            )
        else:
            raise ValueError("Unsupported transformation primitive: " + kind)
    return frame


def consume(frame, operation):
    result = {}
    for item in operation["outputs"]:
        view = filter_rows(frame, item.get("conditions", []))
        kind = item["operator"]
        values = (
            pd.to_numeric(view[item["field"]], errors="raise")
            if "field" in item and kind not in ("value", "unique")
            else None
        )
        if kind in ("sum", "mean", "median", "min", "max"):
            value = getattr(values, kind)(skipna=item.get("skipna", True))
        elif kind == "std":
            value = values.std(ddof=item.get("ddof", 0))
        elif kind == "count":
            value = len(view) if "field" not in item else view[item["field"]].count()
        elif kind == "unique":
            value = view[item["field"]].nunique(dropna=item.get("dropna", True))
        elif kind == "entity_mean":
            value = view.groupby(item["entity"])[item["field"]].sum().mean()
        elif kind == "ratio":
            numerator = float(values.sum())
            denominator = (
                view[item["entity"]].nunique() if "entity" in item else len(view)
            )
            value = numerator / denominator if denominator else np.nan
        elif kind in ("pearson", "spearman"):
            pair = view[item["fields"]].dropna()
            function = stats.pearsonr if kind == "pearson" else stats.spearmanr
            measured = function(pair.iloc[:, 0], pair.iloc[:, 1])
            value = (
                measured.pvalue
                if item.get("statistic") == "pvalue"
                else measured.statistic
            )
        elif kind == "normality":
            value = (
                "yes"
                if stats.shapiro(values.dropna()).pvalue >= item.get("alpha", 0.05)
                else "no"
            )
        elif kind == "quantile":
            value = values.quantile(item["q"])
        elif kind == "value":
            value = view[item["field"]].iloc[item.get("index", 0)]
        else:
            raise ValueError("Unsupported consumer operator: " + kind)
        if "round" in item and isinstance(value, (float, np.floating)):
            value = round(float(value), item["round"])
        result[item["name"]] = value.item() if isinstance(value, np.generic) else value
    return result


def equal(left, right, tolerance=1e-8):
    if left.keys() != right.keys():
        return False
    for key in left:
        a, b = left[key], right[key]
        if isinstance(a, (float, int)) and isinstance(b, (float, int)):
            if np.isnan(a) and np.isnan(b):
                continue
            if not np.isclose(a, b, atol=tolerance, rtol=0, equal_nan=True):
                return False
        elif str(a) != str(b):
            return False
    return True


def inject(plan, clause, kind):
    changed = copy.deepcopy(plan)
    params = clause["params"]
    if kind == "unit":
        for op in changed["operations"]:
            if op["op"] == "convert" and op["output"] in clause["fields"]:
                op["factor"] *= 1000
                return changed
    elif kind == "time":
        for op in changed["operations"]:
            if op["op"] == "time_window":
                op["end"] = str(pd.Timestamp(op["end"]) - pd.Timedelta(days=1))
                return changed
    elif kind == "field":
        fields = params.get("alternative_fields", [])
        if fields:
            changed["operations"].append(
                {
                    "op": "rename",
                    "mapping": {
                        fields[0]: clause["fields"][0],
                        clause["fields"][0]: fields[0],
                    },
                }
            )
            return changed
    elif kind == "join":
        changed["operations"].append(
            {
                "op": "join",
                "table": params["table"],
                "keys": params["keys"],
                "how": "inner",
            }
        )
        return changed
    elif kind == "denominator":
        changed["operations"].append(
            {"op": "deduplicate", "keys": params["entity_keys"]}
        )
        return changed
    elif kind == "aggregation":
        for op in changed["operations"]:
            if op["op"] == "group":
                for item in op["aggregations"].values():
                    if item["operator"] == "sum":
                        item["operator"] = "mean"
                        return changed
    raise ValueError("No applicable controlled mutation for " + kind)
