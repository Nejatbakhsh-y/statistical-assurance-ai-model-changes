"""Generated standalone HTML with escaped values."""
import html
import json
from pathlib import Path


def save_report(result, path):
    payload = html.escape(json.dumps(result, indent=2, sort_keys=True))
    page = ("<!doctype html><html lang='en'><meta charset='utf-8'>"
            "<title>Illustrative independent model validation protocol</title>"
            "<style>body{font:16px system-ui;max-width:980px;margin:3rem auto;line-height:1.5}"
            "pre{white-space:pre-wrap;background:#f4f4f4;padding:1rem}h1{color:#19324d}</style>"
            "<h1>Multi-level model change validation</h1>"
            "<p><strong>Research prototype only.</strong> Built-in dataset, controlled score "
            "transformation. Not an official independent audit or deployment approval.</p>"
            "<h2>Executive summary and advisory disposition</h2><p>"
            + html.escape(result["recommendation"]["status"] + ": " + result["recommendation"]["reason"])
            + "</p><h2>Configuration, metrics, uncertainty, assumptions, reproducibility</h2>"
            "<pre>"+payload+"</pre><h2>Probability reliability</h2>"
            "<img style='max-width:100%' alt='Reliability diagram' src='../figures/reliability.png'>"
            "<p>Limitations: one training fit/split; bootstrap across test cases only; "
            "human review required; no safety or subgroup assessment.</p></html>")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(page, encoding="utf-8")
