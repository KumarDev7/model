"""Build experiments/results/report.html from the JSON results."""

from __future__ import annotations

import json
import os

HERE = os.path.dirname(__file__)
RESULTS = os.environ.get("KT_RESULTS", os.path.join(HERE, "results"))


def _load(name):
    path = os.path.join(RESULTS, name)
    return json.load(open(path)) if os.path.exists(path) else None


def main():
    data = {
        "analysis": _load("analysis_main.json"),
        "dense": _load("dense_comparison.json"),
        "capacity": _load("capacity.json"),
        "retention": _load("retention.json"),
        "generalization": _load("generalization.json"),
    }
    for row in data["dense"] or []:
        row.pop("history", None)
    template = open(os.path.join(HERE, "report_template.html")).read()
    text = open(os.path.join(HERE, "report_text.js")).read()
    html = template.replace("__DATA__", json.dumps(data, separators=(",", ":"))).replace("__TEXT__", text)
    out = os.path.join(RESULTS, "report.html")
    open(out, "w").write(html)
    print("wrote", out)


if __name__ == "__main__":
    main()
