"""Parent-side training admission and subprocess deadline."""
import json
import os
import subprocess
import sys

from .config import ROOT, Settings
from .data import infer_target


def train_models(frame, settings: Settings, target=None, problem=None) -> dict:
    inferred = infer_target(frame)
    selected = target or inferred["target"]
    if selected not in frame.columns:
        raise ValueError("Selected target does not exist")
    if target and not problem:
        inferred_type = infer_target(frame[[target]].rename(columns={target: "target"}))["problem_type"]
    else:
        inferred_type = inferred["problem_type"]
    problem = problem or inferred_type
    if target:
        inferred = {"target": selected, "problem_type": problem,
                    "reason": "Explicit user target; problem type inferred unless overridden", "confidence": "user_selected"}
    payload = json.dumps({"frame": frame.to_json(orient="split"), "target": selected, "problem": problem})
    if len(payload.encode()) > 16*1024*1024:
        raise ValueError("Training payload exceeds the worker budget")
    child_env = {key: value for key, value in os.environ.items()
                 if key.upper() in {"PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "USERPROFILE", "VIRTUAL_ENV"}}
    child_env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUTF8="1")
    try:
        completed = subprocess.run([sys.executable, "-B", "-m", "portfolio.ml_worker"], input=payload,
                                   capture_output=True, text=True, encoding="utf-8", cwd=ROOT,
                                   env=child_env, timeout=settings.ml_timeout_seconds, check=False)
    except subprocess.TimeoutExpired:
        raise TimeoutError("Training exceeded its deadline; worker stopped. Reduce rows/features.") from None
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError:
        raise RuntimeError("Training worker failed before producing a result; inspect local diagnostics") from None
    if completed.returncode:
        raise ValueError(result.get("error", "Training failed"))
    result["target_inference"] = inferred
    return result
