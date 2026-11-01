"""Render report/report.md to report/report.html and print it to report/report.pdf with Microsoft Edge."""
import subprocess
import sys
import re
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
CSS = """
@page { size: Letter; margin: 0.7in 0.75in;
  @bottom-right { content: counter(page) " / " counter(pages); font: 8pt Arial; color: #666; }
}
body { font-family: "Segoe UI", Arial, sans-serif; font-size: 10pt; line-height: 1.38; color: #0b0b0b; overflow-wrap: anywhere; }
h1 { font-size: 17pt; margin: 0 0 4pt; }
h2 { font-size: 13pt; margin: 14pt 0 5pt; border-bottom: 1px solid #ccc; padding-bottom: 2pt; }
h3 { font-size: 11pt; margin: 10pt 0 4pt; }
p, li { margin: 3pt 0; }
table { border-collapse: collapse; margin: 6pt 0 8pt; font-size: 8.3pt; width: 100%; break-inside: avoid; }
th { overflow-wrap: normal; }
table.metrics { font-size: 7.8pt; }
table.metrics td:not(:first-child) { white-space: nowrap; overflow-wrap: normal; }
table.metrics td:first-child { min-width: 65px; }
table.models th:nth-child(4), table.models td:nth-child(4) { min-width: 58px; }
table.setup th:first-child, table.setup td:first-child { min-width: 65px; }
tr { break-inside: avoid; }
h2, h3, p:has(+ table) { break-after: avoid; }
th, td { border: 1px solid #c8c7c2; padding: 2.5pt 4pt; text-align: left; }
th { background: #f1f0ec; }
code { font-family: Consolas, monospace; font-size: 8.5pt; background: #f4f3ef; padding: 0 2px; }
pre { background: #f4f3ef; padding: 6pt; font-size: 8pt; white-space: pre-wrap; page-break-inside: avoid; }
pre code { background: none; }
img { max-width: 100%; max-height: 2.65in; object-fit: contain; display: block; margin: 6pt auto 2pt; page-break-inside: avoid; }
p:has(> img) { break-after: avoid; }
li { break-inside: avoid; }
a { color: #15557c; text-decoration: underline; }
.caption, em.caption { font-size: 8.5pt; color: #52514e; }
.subtitle { color: #52514e; margin-bottom: 10pt; }
"""


def main():
    source = (ROOT / "report/report.md").read_text(encoding="utf-8")
    for dash in ('\u2011', '\u2013', '\u2014'):
        source = source.replace(dash, '-')
    references = source.split('## 8. References', 1)[1].strip()
    references = re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)', r'\1 (\2)', references)
    references = references.replace('*', '').replace('`', '')
    (ROOT / 'report/references.txt').write_text('References\n\n' + references + '\n', encoding='utf-8')
    body = markdown.markdown(source, extensions=["tables", "fenced_code", "md_in_html"])
    table_index = iter(range(20))
    def classify_table(match):
        index = next(table_index)
        css_class = 'models' if index == 0 else 'setup' if index == 2 else 'metrics' if index >= 3 else 'taxonomy'
        return f'<table class="{css_class}">'
    body = re.sub(r'<table>', classify_table, body)
    html_path = ROOT / "report/report.html"
    title = source.splitlines()[0].removeprefix('# ').strip()
    html_path.write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title>"
                         f"<style>{CSS}</style></head><body>{body}</body></html>", encoding="utf-8")
    pdf_path = ROOT / "report/report.pdf"
    profile_path = ROOT / "tmp/report-edge-profile"
    profile_path.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(EDGE), "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--user-data-dir={profile_path}",
                    f"--print-to-pdf={pdf_path}", html_path.as_uri()], check=True, timeout=120)
    print(f"Wrote {pdf_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
