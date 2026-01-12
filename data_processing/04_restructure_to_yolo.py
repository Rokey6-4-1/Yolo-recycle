import argparse
import hashlib
import os
import shutil
from pathlib import Path
import sys

YOLO_NAMES = [
    "can_steel",
    "can_aluminium",
    "paper",
    "PET_transparent",
    "PET_color",
    "plastic_PE",
    "plastic_PP",
    "plastic_PS",
    "styrofoam",
    "plastic_bag",
    "glass_brown",
    "glass_green",
    "glass_transparent",
    "battery",
    "light",
]

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def short_hash(s: str, n: int = 8) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:n]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_place(src: Path, dst: Path, mode: str) -> None:
    ensure_dir(dst.parent)
    if mode == "move":
        shutil.move(str(src), str(dst))
    elif mode == "copy":
        shutil.copy2(str(src), str(dst))
    elif mode == "hardlink":
        try:
            os.link(str(src), str(dst))
        except OSError:
            shutil.copy2(str(src), str(dst))
    else:
        raise ValueError(mode)


def unique_dest_name(dest_dir: Path, original_name: str, salt: str) -> str:
    dst = dest_dir / original_name
    if not dst.exists():
        return original_name
    stem = Path(original_name).stem
    suf = Path(original_name).suffix
    h = short_hash(salt, 8)
    return f"{stem}_{h}{suf}"


def write_data_yaml(out_root: Path) -> None:
    y = out_root / "data.yaml"
    lines = [
        f"path: {out_root.as_posix()}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(YOLO_NAMES)}",
        "names:",
    ]
    for n in YOLO_NAMES:
        lines.append(f"  - {n}")
    y.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_label_index(labels_root: Path) -> dict:
    idx = {}
    for p in labels_root.rglob("*.txt"):
        if p.is_file():
            idx.setdefault(p.stem, p)
    return idx


def count_images(p: Path) -> int:
    if not p.exists():
        return 0
    return sum(1 for f in p.rglob("*") if f.is_file() and f.suffix.lower() in IMG_EXTS)


def count_txt(p: Path) -> int:
    if not p.exists():
        return 0
    return sum(1 for f in p.rglob("*.txt") if f.is_file())


def find_best_folder(root: Path, must_contain: str, kind: str):
    """
    kind: "images" or "labels"
    must_contain: "Training" or "Validation" (경로에 이 단어가 들어간 후보만)
    """
    candidates = []
    for d in root.rglob("*"):
        if not d.is_dir():
            continue
        s = str(d).lower()
        if must_contain.lower() not in s:
            continue

        # 이름 힌트 (원천/라벨)
        name = d.name
        if kind == "images":
            if ("원천" in name) or ("image" in name.lower()) or ("images" in name.lower()):
                score = count_images(d)
                if score > 0:
                    candidates.append((score, d))
        else:
            if ("라벨" in name) or ("label" in name.lower()) or ("labels" in name.lower()):
                score = count_txt(d)
                if score > 0:
                    candidates.append((score, d))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def process_split(img_root: Path, lbl_root: Path, out_img_dir: Path, out_lbl_dir: Path, mode: str, split: str):
    ensure_dir(out_img_dir)
    ensure_dir(out_lbl_dir)

    lbl_idx = build_label_index(lbl_root)
    img_files = [p for p in img_root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    print(f"[{split}] images found: {len(img_files)}  (img_root={img_root})")
    print(f"[{split}] labels found: {len(lbl_idx)} stems (lbl_root={lbl_root})")

    missing_label = 0
    moved = 0

    for img in img_files:
        stem = img.stem
        lbl = lbl_idx.get(stem)

        img_name = unique_dest_name(out_img_dir, img.name, salt=str(img))
        lbl_name = Path(img_name).with_suffix(".txt").name

        dst_img = out_img_dir / img_name
        dst_lbl = out_lbl_dir / lbl_name

        safe_place(img, dst_img, mode)

        if lbl is None or not lbl.exists():
            dst_lbl.write_text("", encoding="utf-8")
            missing_label += 1
        else:
            safe_place(lbl, dst_lbl, mode)

        moved += 1

    print(f"[{split}] done: {moved}, missing_label_txt_created: {missing_label}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_root", help="최상위 데이터 폴더(Training/Validation이 들어있는 곳)")
    ap.add_argument("--out", default=r"C:\ROKEY\recycle_yolo")
    ap.add_argument("--mode", choices=["move", "copy", "hardlink"], default="move")
    args = ap.parse_args()

    root = Path(args.dataset_root)
    if not root.exists():
        print(f"[ERROR] dataset_root가 없습니다: {root}")
        sys.exit(1)

    # 자동 탐색
    train_img = find_best_folder(root, must_contain="Training", kind="images")
    train_lbl = find_best_folder(root, must_contain="Training", kind="labels")
    val_img = find_best_folder(root, must_contain="Validation", kind="images")
    val_lbl = find_best_folder(root, must_contain="Validation", kind="labels")

    print("[AUTO] detected folders:")
    print("  train_img:", train_img)
    print("  train_lbl:", train_lbl)
    print("  val_img  :", val_img)
    print("  val_lbl  :", val_lbl)

    if not all([train_img, train_lbl, val_img, val_lbl]):
        print("\n[ERROR] 자동 탐색 실패. 아래 중 None인 게 있으면 폴더명이 다르거나 구조가 다릅니다.")
        sys.exit(1)

    out = Path(args.out)
    out_img_train = out / "images" / "train"
    out_img_val = out / "images" / "val"
    out_lbl_train = out / "labels" / "train"
    out_lbl_val = out / "labels" / "val"

    process_split(train_img, train_lbl, out_img_train, out_lbl_train, args.mode, "train")
    process_split(val_img, val_lbl, out_img_val, out_lbl_val, args.mode, "val")

    write_data_yaml(out)
    print(f"\n[OK] YOLO dataset created at: {out}")
    print("[OK] data.yaml created.")
    print(f"[NOTE] mode={args.mode} (move면 원본에서 파일이 이동됩니다.)")


if __name__ == "__main__":
    main()
