"""M19 typography guard: use ASCII hyphen in repository prose and web UI.

Checks tracked human-readable docs/source, not binary models, images or
historical notebooks. Never rewrite real CLI options such as --help, JS
operators, CSS custom properties or required schema keys.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTENSIONS = {".md", ".txt", ".py", ".js", ".css", ".html", ".yml", ".yaml", ".toml"}


def test_no_unicode_en_or_em_dashes_in_docs_and_web_sources():
    bad = []
    for file in ROOT.rglob("*"):
        if not file.is_file() or file.suffix.lower() not in EXTENSIONS:
            continue
        if any(part in {".git", ".venv", "node_modules", "__pycache__", "dist", "artifacts", "data"} for part in file.relative_to(ROOT).parts):
            continue
        content = file.read_text("utf-8")
        for codepoint in (0x2013, 0x2014):
            if chr(codepoint) in content:
                bad.append((str(file.relative_to(ROOT)), "U+%04X" % codepoint))
    assert bad == [], (
        "Replace typography en/em dash with ASCII hyphen in prose, while "
        "preserving syntactic double hyphens used in CLI flags and CSS: "
        + repr(bad)
    )
