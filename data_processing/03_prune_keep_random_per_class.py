"""
AIHub '232.재활용품 분류 및 선별 데이터' 폴더를
클래스(=각 TS_/VS_ 하위 폴더)마다 랜덤으로 train N장 / val M장만 남기고
이미지 + 라벨을 같이 삭제(prune)합니다.

트리 구조 전제:
- Training/01.원천데이터/TS_...  <-> Training/02.라벨링데이터/TL_...
- Validation/01.원천데이터/VS_... <-> Validation/02.라벨링데이터/VL_...

주의: 삭제는 되돌릴 수 없습니다. DRY_RUN로 먼저 확인 후 진행하세요.
"""

from __future__ import annotations
import random
from pathlib import Path
from typing import Iterable, Set, Tuple

# =========================
# 설정 
# =========================
DATA_ROOT = r"C:\ROKEY\232.재활용품 분류 및 선별 데이터"  # 폴더 최상단
TRAIN_KEEP = 4000
VAL_KEEP = 250
SEED = 42

DRY_RUN = True        # True면 삭제 안 하고 "몇 개 지울지"만 출력
WRITE_MANIFEST = True # 남긴 파일 목록(manifest) 저장

# 이미지 확장자 (필요하면 추가)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# 라벨 확장자 (txt로 변환했어도 혹시 json 남아있을 수 있어 같이 정리)
LBL_EXTS = {".txt", ".json"}

# =========================
# 내부 구현
# =========================

def list_files_with_ext(folder: Path, exts: Set[str]) -> list[Path]:
    if not folder.exists():
        return []
    out = []
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            out.append(p)
    return out

def infer_label_dir(images_dir: Path) -> Path:
    """
    TS_xxx -> TL_xxx, VS_xxx -> VL_xxx
    """
    name = images_dir.name
    if name.startswith("TS_"):
        lbl_name = "TL_" + name[3:]
    elif name.startswith("VS_"):
        lbl_name = "VL_" + name[3:]
    else:
        raise ValueError(f"Unexpected images dir name (not TS_/VS_): {images_dir}")

    # images_dir: .../<Split>/01.원천데이터/<TS_...>
    # labels_dir: .../<Split>/02.라벨링데이터/<TL_...>
    split_dir = images_dir.parents[1]   # .../<Split>  (Training or Validation)
    labels_root = split_dir / "02.라벨링데이터"
    return labels_root / lbl_name

def choose_keep_stems(img_files: list[Path], keep_n: int, rng: random.Random) -> Set[str]:
    stems = [p.stem for p in img_files]
    if len(stems) <= keep_n:
        return set(stems)
    return set(rng.sample(stems, keep_n))

def safe_unlink(p: Path) -> None:
    # Windows에서 읽기전용이면 삭제 실패할 수 있어 권한 플래그 해제 시도
    try:
        p.unlink()
    except PermissionError:
        try:
            p.chmod(0o666)
            p.unlink()
        except Exception:
            raise

def prune_one_class(images_dir: Path, keep_n: int, rng: random.Random, manifest_root: Path) -> Tuple[int,int,int,int]:
    """
    returns: (kept_imgs, deleted_imgs, kept_lbls, deleted_lbls)
    """
    label_dir = infer_label_dir(images_dir)

    img_files = list_files_with_ext(images_dir, IMG_EXTS)
    keep_stems = choose_keep_stems(img_files, keep_n, rng)

    # --- prune images ---
    kept_imgs = 0
    deleted_imgs = 0
    for p in img_files:
        if p.stem in keep_stems:
            kept_imgs += 1
        else:
            deleted_imgs += 1
            if not DRY_RUN:
                safe_unlink(p)

    # --- prune labels (txt/json) ---
    lbl_files = list_files_with_ext(label_dir, LBL_EXTS)
    kept_lbls = 0
    deleted_lbls = 0
    for p in lbl_files:
        if p.stem in keep_stems:
            kept_lbls += 1
        else:
            deleted_lbls += 1
            if not DRY_RUN:
                safe_unlink(p)

    # --- manifest 저장 ---
    if WRITE_MANIFEST:
        manifest_root.mkdir(parents=True, exist_ok=True)
        mf = manifest_root / f"{images_dir.name}__keep_{len(keep_stems)}.txt"
        mf_content = "\n".join(sorted(keep_stems))
        if DRY_RUN:
            # DRY_RUN이어도 목록은 저장해두면 좋음
            mf.write_text(mf_content, encoding="utf-8")
        else:
            mf.write_text(mf_content, encoding="utf-8")

    # --- kept 이미지인데 라벨이 없는 경우 경고(삭제는 안 함) ---
    lbl_stems = {p.stem for p in lbl_files}
    missing_lbl = keep_stems - lbl_stems
    if missing_lbl:
        # 너무 길어질 수 있으니 개수만
        print(f"  [WARN] {images_dir.name}: kept {len(keep_stems)} imgs but {len(missing_lbl)} labels are missing in {label_dir}")

    return kept_imgs, deleted_imgs, kept_lbls, deleted_lbls

def prune_split(split_name: str, keep_n: int, rng: random.Random) -> None:
    split_dir = Path(DATA_ROOT) / "01-1.정식개방데이터" / split_name
    images_root = split_dir / "01.원천데이터"

    if not images_root.exists():
        raise FileNotFoundError(f"Images root not found: {images_root}")

    manifest_root = Path(DATA_ROOT) / "__prune_manifest__" / split_name

    class_dirs = [p for p in images_root.iterdir() if p.is_dir()]
    class_dirs.sort(key=lambda p: p.name)

    print(f"\n=== {split_name} | keep per class = {keep_n} | classes = {len(class_dirs)} ===")
    total = {"kept_i":0,"del_i":0,"kept_l":0,"del_l":0}

    for idx, d in enumerate(class_dirs, 1):
        if not (d.name.startswith("TS_") or d.name.startswith("VS_")):
            print(f"  [SKIP] not a class dir (not TS_/VS_): {d}")
            continue

        print(f"[{idx}/{len(class_dirs)}] {d.name}")
        ki, di, kl, dl = prune_one_class(d, keep_n, rng, manifest_root)
        print(f"  images: keep {ki}, delete {di} | labels: keep {kl}, delete {dl}")
        total["kept_i"] += ki
        total["del_i"]  += di
        total["kept_l"] += kl
        total["del_l"]  += dl

    print(f"\n--- {split_name} summary ---")
    print(f"images keep {total['kept_i']:,}, delete {total['del_i']:,}")
    print(f"labels keep {total['kept_l']:,}, delete {total['del_l']:,}")

def main():
    rng = random.Random(SEED)

    print("DATA_ROOT =", DATA_ROOT)
    print("DRY_RUN   =", DRY_RUN)
    print("SEED      =", SEED)
    print("TRAIN_KEEP=", TRAIN_KEEP, "| VAL_KEEP=", VAL_KEEP)

    # Training: TS_... / TL_...
    prune_split("Training", TRAIN_KEEP, rng)

    # Validation: VS_... / VL_...
    prune_split("Validation", VAL_KEEP, rng)

    if DRY_RUN:
        print("\n[DRY_RUN] 실제 삭제는 하지 않았습니다. DRY_RUN=False로 바꾼 뒤 다시 실행하세요.")
    else:
        print("\n[DONE] 삭제 완료.")

if __name__ == "__main__":
    main()
