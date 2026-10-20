"""Pre-flight check for Colab training scripts BEFORE handing them to the user.

Usage: python tools/preflight_train_script.py train_engine13_swin_focal.py [more.py ...]
Exit code 0 = pass, 1 = fail. Catches the errors that actually crashed us:
  - SyntaxError / unterminated f-strings (engine13 line 246)
  - missing gradient checkpointing for big backbones (14.4 GB OOM)
  - checkpoint only saved at the end (Colab disconnect loses everything)
  - Drive mount without try/except (headless hang)
  - hard-coded credentials
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

BIG_BACKBONES = ("swin_base", "swin_large", "convnext_large", "convnext_base", "vit_large", "coatnet")


def check(path: Path) -> list[str]:
    errs: list[str] = []
    src = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError as e:
        return [f"SyntaxError line {e.lineno}: {e.msg}"]

    low = src.lower()
    if any(b in low for b in BIG_BACKBONES) and "set_grad_checkpointing" not in src:
        errs.append("large backbone without set_grad_checkpointing(True) -> OOM risk on T4")
    if "drive.mount" in src and not re.search(r"try:\s*\n[^\n]*drive\.mount", src):
        errs.append("drive.mount not wrapped in try/except -> headless hang risk")
    if re.search(r"(KAGGLE_KEY|api_key|token)\s*=\s*['\"][A-Za-z0-9_\-]{16,}['\"]", src):
        errs.append("possible hard-coded credential")
    saves = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "save"]
    in_loop = any(isinstance(p, (ast.For, ast.While)) for p in ast.walk(tree) for s in saves if s in ast.walk(p))
    if saves and not in_loop:
        errs.append("torch.save never inside a loop -> no per-epoch checkpoint (disconnect risk)")
    if not saves:
        errs.append("no torch.save call found")
    return errs


def main() -> int:
    bad = 0
    for arg in sys.argv[1:]:
        p = Path(arg)
        errs = check(p)
        status = "PASS" if not errs else "FAIL"
        print(f"[{status}] {p}")
        for e in errs:
            print(f"   - {e}")
        bad += bool(errs)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
