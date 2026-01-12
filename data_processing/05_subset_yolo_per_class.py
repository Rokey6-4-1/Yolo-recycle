import random
import shutil
from pathlib import Path
from collections import defaultdict

# ===== 기본 설정 (원하면 여기만 바꿔도 됨) =====
N_CLASSES = 15
TRAIN_PER_CLASS = 1000
VAL_PER_CLASS = 200
SEED = 42
MODE = "copy"   # "copy" 추천(원본 유지). "move"는 원본에서 빼옴(주의)
# ============================================

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_classes_from_yolo_txt(txt_path: Path):
    classes = set()
    if not txt_path.exists():
        return classes
    try:
        with txt_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                try:
                    cls = int(float(parts[0]))
                    classes.add(cls)
                except:
                    continue
    except:
        pass
    return classes


def index_split(img_dir: Path, lbl_dir: Path):
    items = []
    per_class = defaultdict(list)

    for img_path in img_dir.iterdir():
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in IMG_EXTS:
            continue

        stem = img_path.stem
        lbl_path = lbl_dir / f"{stem}.txt"
        classes = read_classes_from_yolo_txt(lbl_path)

        items.append({"stem": stem, "img": img_path, "lbl": lbl_path, "classes": classes})
        for c in classes:
            per_class[c].append(stem)

    return items, per_class


def greedy_select(items, n_classes: int, quota_per_class: int, seed: int):
    rnd = random.Random(seed)
    rnd.shuffle(items)

    need = {c: quota_per_class for c in range(n_classes)}
    selected = []
    selected_set = set()

    def done():
        return all(v <= 0 for v in need.values())

    for it in items:
        if done():
            break
        if it["stem"] in selected_set:
            continue

        # 이 이미지가 아직 quota가 남은 클래스라도 포함하면 선택
        hit = any((c in need and need[c] > 0) for c in it["classes"])
        if not hit:
            continue

        selected.append(it)
        selected_set.add(it["stem"])

        # 포함된 클래스들의 quota를 동시에 깎음(멀티클래스면 여러 개 채움)
        for c in it["classes"]:
            if c in need and need[c] > 0:
                need[c] -= 1

    unmet = {c: v for c, v in need.items() if v > 0}
    return selected, unmet


def try_read_names_from_yaml(data_yaml: Path, n_classes: int):
    # 간단 파싱: names: [ ... ]
    names = [str(i) for i in range(n_classes)]
    if not data_yaml.exists():
        return names
    try:
        import re
        txt = data_yaml.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"names\s*:\s*\[(.*?)\]", txt, re.S)
        if not m:
            return names
        inside = m.group(1)
        pairs = re.findall(r"'([^']*)'|\"([^\"]*)\"", inside)
        parsed = [a if a else b for a, b in pairs]
        if len(parsed) == n_classes:
            return parsed
    except:
        pass
    return names


def write_data_yaml(out_root: Path, class_names, n_classes: int):
    lines = [
        f"path: {out_root.as_posix()}",
        "train: images/train",
        "val: images/val",
        f"nc: {n_classes}",
        "names: [" + ", ".join([f"'{n}'" for n in class_names]) + "]",
    ]
    (out_root / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def pick_dirs_with_tk():
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox
    except Exception as e:
        print("[ERROR] tkinter를 못 불러왔습니다. (파이썬 설치에 따라 다를 수 있음)")
        print("대신 아래 ROOT/OUT 변수를 코드 상단에 직접 넣고 실행하세요.")
        raise

    root = tk.Tk()
    root.withdraw()

    messagebox.showinfo("폴더 선택", "원본 데이터셋 폴더(recycle_yolo)를 선택하세요.\n예: C:\\ROKEY\\recycle_yolo")
    src = filedialog.askdirectory(title="원본 데이터셋 폴더 선택")
    if not src:
        raise SystemExit("취소됨")

    messagebox.showinfo("폴더 선택", "출력 폴더(새로 만들거나 비어있는 폴더)를 선택하세요.\n예: C:\\ROKEY\\recycle_yolo_small")
    out = filedialog.askdirectory(title="출력 폴더 선택(비어있는 폴더 추천)")
    if not out:
        raise SystemExit("취소됨")

    return Path(src), Path(out)


def ask_int(prompt, default):
    s = input(f"{prompt} (기본 {default}): ").strip()
    if not s:
        return default
    return int(s)


def main():
    print("=== YOLO 데이터셋 클래스별 서브셋 만들기(그냥 실행용) ===")
    print("폴더 선택 창이 뜹니다. (안 뜨면 tkinter 문제일 수 있음)\n")

    src_root, out_root = pick_dirs_with_tk()

    n_classes = ask_int("클래스 수(nc)", N_CLASSES)
    train_per = ask_int("train 클래스당 목표 이미지 수", TRAIN_PER_CLASS)
    val_per = ask_int("val 클래스당 목표 이미지 수", VAL_PER_CLASS)
    seed = ask_int("랜덤 seed", SEED)

    mode = input(f"mode(copy/move) (기본 {MODE}): ").strip().lower() or MODE
    if mode not in ("copy", "move"):
        print("[WARN] mode가 이상해서 copy로 진행합니다.")
        mode = "copy"

    img_train = src_root / "images" / "train"
    lbl_train = src_root / "labels" / "train"
    img_val = src_root / "images" / "val"
    lbl_val = src_root / "labels" / "val"

    for p in [img_train, lbl_train, img_val, lbl_val]:
        if not p.exists():
            raise FileNotFoundError(f"경로가 없습니다: {p}")

    # 인덱싱
    train_items, _ = index_split(img_train, lbl_train)
    val_items, _ = index_split(img_val, lbl_val)
    print(f"\n[INFO] train images found: {len(train_items)}")
    print(f"[INFO] val   images found: {len(val_items)}")

    # 선택
    train_sel, train_unmet = greedy_select(train_items, n_classes, train_per, seed)
    val_sel, val_unmet = greedy_select(val_items, n_classes, val_per, seed + 1)

    print(f"\n[INFO] selected train images: {len(train_sel)}")
    if train_unmet:
        print("[WARN] train quota 못 채운 클래스:", train_unmet)

    print(f"[INFO] selected val images  : {len(val_sel)}")
    if val_unmet:
        print("[WARN] val quota 못 채운 클래스:", val_unmet)

    # 출력 폴더 구조 생성
    (out_root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out_root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (out_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (out_root / "labels" / "val").mkdir(parents=True, exist_ok=True)

    mover = shutil.copy2 if mode == "copy" else shutil.move

    def transfer(selected, split):
        img_dst_dir = out_root / "images" / split
        lbl_dst_dir = out_root / "labels" / split
        moved = 0
        missing_lbl = 0

        for it in selected:
            mover(str(it["img"]), str(img_dst_dir / it["img"].name))

            if it["lbl"].exists():
                mover(str(it["lbl"]), str(lbl_dst_dir / it["lbl"].name))
            else:
                # 라벨 없으면 빈 txt 생성
                (lbl_dst_dir / f"{it['stem']}.txt").write_text("", encoding="utf-8")
                missing_lbl += 1

            moved += 1

        return moved, missing_lbl

    m1, ml1 = transfer(train_sel, "train")
    m2, ml2 = transfer(val_sel, "val")

    print(f"\n[DONE] train {mode}: {m1}, missing labels(empty created): {ml1}")
    print(f"[DONE] val   {mode}: {m2}, missing labels(empty created): {ml2}")

    # data.yaml 생성
    class_names = try_read_names_from_yaml(src_root / "data.yaml", n_classes)
    write_data_yaml(out_root, class_names, n_classes)
    print(f"[DONE] wrote: {out_root / 'data.yaml'}")

    print("\n=== 완료 ===")
    print(f"OUT: {out_root}")


if __name__ == "__main__":
    main()
