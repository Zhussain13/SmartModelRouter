"""Runs the full demo pipeline (label -> train -> evaluate) in a temporary copy."""
import os
import shutil
import subprocess
import sys

from conftest import ROOT


def test_demo_pipeline_end_to_end(tmp_path):
    for name in os.listdir(ROOT):
        if name.endswith(".py"):
            shutil.copy(os.path.join(ROOT, name), tmp_path)
    shutil.copytree(os.path.join(ROOT, "data"), tmp_path / "data")

    out = subprocess.run([sys.executable, "run_pipeline.py", "--demo"], cwd=tmp_path,
                         capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stdout + out.stderr
    for f in ["labels.csv", "test_split.csv", "router.joblib", "tuning.json", "results.csv", "results.png"]:
        assert (tmp_path / "outputs" / "demo" / f).exists(), f
    assert "oracle (perfect router)" in out.stdout
