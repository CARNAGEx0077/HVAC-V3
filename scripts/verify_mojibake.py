# -*- coding: utf-8 -*-
"""Verify zero mojibake in all frontend files."""
from pathlib import Path

files = [
    Path("frontend/index.html"),
    Path("frontend/js/app.js"),
    Path("frontend/js/pages.js"),
    Path("frontend/js/components.js"),
    Path("frontend/js/state.js"),
    Path("frontend/css/style.css"),
]

bad_sequences = [
    "\u00c3",  # Ã
    "\u00c2",  # Â
    "\u00e2\u20ac",  # â€
    "\u00e2\u2020",  # â†
    "\ufeff",  # BOM
]

total_errors = 0
for f in files:
    content = f.read_text(encoding="utf-8", errors="replace")
    for bad in bad_sequences:
        cnt = content.count(bad)
        if cnt > 0:
            print(f"ERROR: {f} contains {cnt} occurrences of {repr(bad)}")
            total_errors += cnt

if total_errors == 0:
    print("ALL FILES CLEAN: ZERO MOJIBAKE FOUND!")
else:
    print(f"FAILED: {total_errors} mojibake sequences found.")
