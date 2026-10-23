#!/usr/bin/env python3
import ast
import sys
import argparse

def validate_script(script_path: str):
    with open(script_path, "r", encoding="utf-8") as f:
        code = f.read()

    tree = ast.parse(code)
    assert_nodes = [n for n in ast.walk(tree) if isinstance(n, ast.Assert)]
    has_nan_guard = "nan" in code and "median" in code
    has_clamping = "clip" in code and "0.0001" in code

    print(f"=== Guardrail Audit for: {script_path} ===")
    print(f"[+] Syntax: Valid AST")
    print(f"[+] Fatal assert statements: {len(assert_nodes)} (Target: 0)")
    print(f"[+] Fail-soft NaN imputation guard: {'PRESENT' if has_nan_guard else 'MISSING'}")
    print(f"[+] Output probability clamping [0.0001, 0.9999]: {'PRESENT' if has_clamping else 'MISSING'}")

    if len(assert_nodes) > 0:
        print("[!] Warning: script contains fatal assert statements.")
        return False
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", default="submissions/candidate_v60_pure_oai_versia_power_master.py")
    args = parser.parse_args()
    validate_script(args.script)
