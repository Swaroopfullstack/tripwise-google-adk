"""Bounded tabular operations; model-produced code is never evaluated."""
from __future__ import annotations

import ast
import csv
from io import BytesIO, StringIO
import json
import math
import operator

import pandas as pd

from .config import Settings


def read_csv(content: bytes, settings: Settings) -> pd.DataFrame:
    if not content or len(content) > settings.max_upload_mb * 1024 * 1024:
        raise ValueError("CSV is empty or exceeds the upload size limit")
    text = content.decode("utf-8-sig")
    try:
        dialect = csv.Sniffer().sniff(text[:8000], delimiters=",;\t")
        separator = dialect.delimiter
    except csv.Error:
        separator = ","
    header = next(csv.reader(StringIO(text), delimiter=separator))
    if len(set(header)) != len(header) or any(not col.strip() or len(col) > 100 for col in header):
        raise ValueError("CSV headers must be unique, nonempty and at most 100 characters")
    if not 2 <= len(header) <= settings.max_data_columns:
        raise ValueError("CSV column count is outside the configured limits")
    frame = pd.read_csv(BytesIO(content), sep=separator, nrows=settings.max_data_rows + 1)
    if len(frame) > settings.max_data_rows or frame.empty:
        raise ValueError("CSV row count is outside the configured limits")
    frame.columns = [str(col).strip() for col in frame.columns]
    if frame.columns.duplicated().any():
        raise ValueError("Whitespace-normalized CSV headers must remain unique")
    return frame


def profile(frame: pd.DataFrame) -> dict:
    summary = json.loads(frame.describe(include="all").to_json())
    return {"rows": len(frame), "columns": list(frame.columns),
            "types": {col: str(dtype) for col, dtype in frame.dtypes.items()},
            "missing": {col: int(n) for col, n in frame.isna().sum().items()},
            "duplicates": int(frame.duplicated().sum()), "summary": summary}


def infer_target(frame: pd.DataFrame) -> dict:
    preferred = ["target", "label", "outcome", "class", "species", "disease_progression", "price", "sales", "charges"]
    normalized = {col.casefold(): col for col in frame.columns}
    matches = [normalized[name] for name in preferred if name in normalized]
    target = matches[0] if matches else str(frame.columns[-1])
    values = frame[target].dropna()
    if values.empty:
        raise ValueError("The proposed target contains no values; select a different target")
    unique = values.nunique()
    numeric = pd.api.types.is_numeric_dtype(values)
    problem = "classification" if not numeric or unique <= min(12, max(2, len(values)//5)) else "regression"
    return {"target": target, "problem_type": problem,
            "confidence": "high" if len(matches) == 1 else "heuristic",
            "reason": ("A conventional target-column name was found" if matches else
                       "Fallback to the last column; confirm or override before training"),
            "unique_target_values": int(unique), "alternatives": matches[1:]}


def analyze(frame: pd.DataFrame, operation: str, column: str | None = None) -> dict:
    if operation == "missing":
        return {"missing": {col: int(n) for col, n in frame.isna().sum().items()}}
    if operation == "correlation":
        numeric = frame.select_dtypes(include="number")
        if numeric.shape[1] < 2:
            raise ValueError("Correlation requires at least two numeric columns")
        return {"correlation": json.loads(numeric.corr().to_json())}
    if operation == "chart":
        numeric = frame.select_dtypes(include="number").columns.tolist()
        selected = column or (numeric[0] if numeric else frame.columns[0])
        if selected not in frame.columns:
            raise ValueError("The selected chart column does not exist")
        series = frame[selected].dropna()
        if pd.api.types.is_numeric_dtype(series):
            import numpy as np
            finite = series.replace([float("inf"), float("-inf")], float("nan")).dropna()
            if finite.empty:
                raise ValueError("No finite chart values")
            counts, edges = np.histogram(finite, bins=min(12, max(2, int(len(series)**.5))))
            return {"chart": {"kind": "histogram", "column": selected,
                              "x": [f"{a:.3g}–{b:.3g}" for a, b in zip(edges[:-1], edges[1:])],
                              "y": counts.tolist()}}
        counts = series.astype(str).value_counts().head(15)
        return {"chart": {"kind": "bar", "column": selected, "x": counts.index.tolist(), "y": counts.tolist()}}
    if operation != "summary":
        raise ValueError("Unsupported data operation")
    return profile(frame)


def calculate(expression: str) -> float:
    """Small arithmetic AST, with depth, magnitude and exponent limits."""
    if len(expression) > 150:
        raise ValueError("Expression too long")
    tree = ast.parse(expression.strip(), mode="eval")
    nodes = list(ast.walk(tree))
    if len(nodes) > 35:
        raise ValueError("Expression is too complex")
    operations = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                  ast.Div: operator.truediv, ast.Mod: operator.mod, ast.Pow: operator.pow}

    def evaluate(node, depth=0):
        if depth > 12:
            raise ValueError("Expression nested too deeply")
        if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
            value = node.value
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = evaluate(node.operand, depth+1) * (-1 if isinstance(node.op, ast.USub) else 1)
        elif isinstance(node, ast.BinOp) and type(node.op) in operations:
            left, right = evaluate(node.left, depth+1), evaluate(node.right, depth+1)
            if isinstance(node.op, ast.Pow) and abs(right) > 8:
                raise ValueError("Exponent exceeds the arithmetic limit")
            value = operations[type(node.op)](left, right)
        else:
            raise ValueError("Only numeric arithmetic is permitted")
        if not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value) > 1e12:
            raise ValueError("Arithmetic result exceeds the limit")
        return value

    return float(evaluate(tree.body))
