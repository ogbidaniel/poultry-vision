"""
Merge the labeled hen-crop pose dataset into the base pose dataset.

The Roboflow crop export uses a different keypoint ORDER than the base dataset,
so its labels are remapped to the base schema before merging. Splits are
preserved: crop train -> merged train, crop val -> merged val. The base test
split is left unchanged.

Base schema order:
    tail_base, tail_tip, left_hock, right_hock, left_foot, right_foot,
    neck_back, middle_back, comb, beak

Crop export order:
    beak, crown(=comb), back_neck(=neck_back), middle_back, tail_base, tail_tip,
    left_hock, left_foot, right_hock, right_foot
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml

# base_kpts[i] = crop_kpts[PC_FROM_BASE[i]]  (match by anatomical name)
PC_FROM_BASE = [4, 5, 6, 8, 7, 9, 2, 3, 1, 0]
N_KPTS = 10

BASE_YAML = {
    "kpt_shape": [10, 3],
    "flip_idx": [0, 1, 3, 2, 5, 4, 6, 7, 8, 9],
    "names": {0: "hen"},
    "kpt_names": {0: ["tail_base", "tail_tip", "left_hock", "right_hock",
                      "left_foot", "right_foot", "neck_back", "middle_back",
                      "comb", "beak"]},
}


def remap_line(line: str) -> str | None:
    p = line.split()
    if len(p) < 5 + N_KPTS * 3:
        return None
    head = p[:5]                                   # cls cx cy w h (unchanged)
    kp = [p[5 + 3*j: 5 + 3*j + 3] for j in range(N_KPTS)]
    new_kp = [kp[PC_FROM_BASE[i]] for i in range(N_KPTS)]
    return " ".join(head + [v for trip in new_kp for v in trip])


def copy_split(src: Path, dst: Path, split: str, remap: bool) -> int:
    si, sl = src / "images" / split, src / "labels" / split
    di, dl = dst / "images" / split, dst / "labels" / split
    di.mkdir(parents=True, exist_ok=True); dl.mkdir(parents=True, exist_ok=True)
    n = 0
    if not si.exists():
        return 0
    for img in sorted(si.glob("*.*")):
        lbl = sl / (img.stem + ".txt")
        if not lbl.exists():
            continue
        shutil.copy2(img, di / img.name)
        if remap:
            out = [remap_line(ln) for ln in lbl.read_text().splitlines() if ln.strip()]
            (dl / lbl.name).write_text("\n".join(o for o in out if o))
        else:
            shutil.copy2(lbl, dl / lbl.name)
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="dataset/pose/pose-dataset-merged-split")
    ap.add_argument("--crops", default="dataset/pose/pose-crops")
    ap.add_argument("--out", default="dataset/pose/pose-dataset-merged-v2-split")
    args = ap.parse_args()

    base, crops, out = Path(args.base), Path(args.crops), Path(args.out)
    if out.exists():
        shutil.rmtree(out)

    counts = {}
    for split in ("train", "val", "test"):
        counts[f"base_{split}"] = copy_split(base, out, split, remap=False)
    counts["crops_train"] = copy_split(crops, out, "train", remap=True)
    counts["crops_val"] = copy_split(crops, out, "val", remap=True)

    cfg = dict(BASE_YAML)
    cfg["path"] = str(out.resolve())
    cfg["train"] = "images/train"
    cfg["val"] = "images/val"
    cfg["test"] = "images/test"
    (out / "data.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    for split in ("train", "val", "test"):
        n = len(list((out / "images" / split).glob("*.*")))
        print(f"{split}: {n} images")
    print("counts:", counts)
    print("merged ->", out)


if __name__ == "__main__":
    main()
