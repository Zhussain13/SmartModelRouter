"""The web dashboard export: runs on demo labels and embeds valid JSON in the page."""
import json
import os
import shutil
import subprocess
import sys

from conftest import ROOT


def test_export_embeds_data(tmp_path):
    for name in os.listdir(ROOT):
        if name.endswith(".py"):
            shutil.copy(os.path.join(ROOT, name), tmp_path)
    shutil.copytree(os.path.join(ROOT, "data"), tmp_path / "data")
    shutil.copytree(os.path.join(ROOT, "docs"), tmp_path / "docs")
    run = lambda *a: subprocess.run([sys.executable, *a], cwd=tmp_path, capture_output=True, text=True, timeout=600)
    assert run("run_pipeline.py", "--demo").returncode == 0
    out = run("export_web_data.py", "--demo")
    assert out.returncode == 0, out.stdout + out.stderr

    page = (tmp_path / "docs" / "index.html").read_text(encoding="utf-8")
    start = page.index('<script id="smr-data" type="application/json">') + len('<script id="smr-data" type="application/json">')
    data = json.loads(page[start:page.index("</script>", start)])
    assert data["meta"]["mode"] == "demo"
    assert len(data["questions"]) == 327
    assert len(data["questions"][0]) == len(data["cols"])
    assert data["faq"], "some FAQ phrasings should pass the retrieval threshold"
    assert all(0 <= row[data["cols"].index("p_learned")] <= 1 for row in data["questions"])
