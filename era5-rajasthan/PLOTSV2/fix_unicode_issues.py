"""
Quick fix script to replace Unicode characters with ASCII in all plotting scripts
"""

import os
import glob

# Find all .py files in this directory
scripts = glob.glob("*.py")
scripts = [s for s in scripts if s not in ["fix_unicode_issues.py", "run_all_plots.py"]]

replacements = [
    ("✓", "[OK]"),
    ("⚠", "[WARN]"),
    ("ℹ", "[INFO]"),
    ("ρ", "rho"),
    ("°", "deg"),
    ("–", "-"),
    ("—", "-"),
]

for script in scripts:
    filepath = script
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    original = content
    for old, new in replacements:
        content = content.replace(old, new)

    if content != original:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Fixed: {script}")
    else:
        print(f"OK: {script}")

print("\nDone! All Unicode characters replaced with ASCII equivalents.")
