#!/usr/bin/env python3
import os
import torch
from pathlib import Path
from typing import List

def build_fp16_model_soup(checkpoint_paths: List[Path], output_path: Path):
    """Merges multiple fold checkpoints into a single FP16 weight-space average."""
    if not checkpoint_paths:
        print("[!] No checkpoints provided.")
        return
    print(f"[*] Creating FP16 model soup from {len(checkpoint_paths)} checkpoints...")
    base_ckpt = torch.load(checkpoint_paths[0], map_location="cpu", weights_only=False)
    base_sd = base_ckpt.get("model", base_ckpt)
    clean_base = {k[7:] if k.startswith("module.") else k: v.float() for k, v in base_sd.items() if isinstance(v, torch.Tensor)}
    
    accum_sd = {k: v.clone() for k, v in clean_base.items()}
    count = 1
    
    for p in checkpoint_paths[1:]:
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        sd = ckpt.get("model", ckpt)
        clean = {k[7:] if k.startswith("module.") else k: v.float() for k, v in sd.items() if isinstance(v, torch.Tensor)}
        if clean.keys() == clean_base.keys():
            for k in accum_sd:
                accum_sd[k] += clean[k]
            count += 1
            
    soup_fp16 = {k: (accum_sd[k] / count).half() for k in accum_sd}
    torch.save({"model": soup_fp16, "folds_merged": count}, output_path)
    print(f"[+] Saved FP16 soup ({output_path.stat().st_size / 1e6:.1f} MB) to {output_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        pts = [Path(p) for p in sys.argv[1:-1]]
        out = Path(sys.argv[-1])
        build_fp16_model_soup(pts, out)
    else:
        print("Usage: python soup_weights.py fold1.pt fold2.pt ... output_soup_fp16.pt")
