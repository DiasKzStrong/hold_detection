"""Guard against reintroducing untranslated Cyrillic project text."""

import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def test_maintained_project_text_has_no_untranslated_cyrillic():
    cyrillic = re.compile(r"[\u0400-\u052f]")
    extensions = {".py", ".md", ".ipynb", ".toml", ".json", ".yaml", ".yml", ".txt"}
    paths = [ROOT / name for name in ("AGENTS.md", "README.md", "pyproject.toml")]
    for directory in ("src", "tests", "scripts", "notebooks", "docs"):
        paths.extend(path for path in (ROOT / directory).rglob("*")
                     if path.is_file() and path.suffix in extensions)
    untranslated = []
    for path in sorted(paths):
        # Decode notebook JSON so escaped Unicode cannot bypass the language check.
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".ipynb":
            text = json.dumps(json.loads(text), ensure_ascii=False)
        if cyrillic.search(text):
            untranslated.append(str(path.relative_to(ROOT)))
    assert not untranslated, f"Translate maintained project text into English: {untranslated}"
