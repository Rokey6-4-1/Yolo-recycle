from pathlib import Path
import json

# =======================
# 1) 여기 경로만 맞게 설정
# =======================
DATASET_ROOT = Path(r"C:\ROKEY\232.재활용품 분류 및 선별 데이터\01-1.정식개방데이터")

LABEL_DIRS = [
    DATASET_ROOT / "Training" / "02.라벨링데이터",
    DATASET_ROOT / "Validation" / "02.라벨링데이터",
]

# =======================
# 2) 클래스 매핑(DETAILS 15종)
# =======================
DETAILS_15 = [
    "철캔", "알루미늄캔", "종이",
    "무색단일", "유색단일",
    "PE", "PP", "PS",
    "스티로폼", "비닐",
    "갈색", "녹색", "투명",
    "건전지", "형광등",
]
DETAILS_TO_ID = {name: i for i, name in enumerate(DETAILS_15)}

# =======================
# 3) 옵션
# =======================
DELETE_JSON = True              # json 삭제할지
WRITE_EMPTY_TXT = False         # 박스가 0개여도 빈 txt 파일을 만들지 (False면 txt 안만듦)
CLAMP_TO_IMAGE = True           # bbox가 이미지 밖으로 나가면 잘라낼지


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def yolo_from_xyxy(class_id: int, x1: float, y1: float, x2: float, y2: float, img_w: int, img_h: int) -> str | None:
    # xyxy in pixels
    if CLAMP_TO_IMAGE:
        x1 = clamp(x1, 0.0, img_w - 1.0)
        x2 = clamp(x2, 0.0, img_w - 1.0)
        y1 = clamp(y1, 0.0, img_h - 1.0)
        y2 = clamp(y2, 0.0, img_h - 1.0)

    w = x2 - x1
    h = y2 - y1
    if w <= 1e-6 or h <= 1e-6:
        return None

    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    bw = w / img_w
    bh = h / img_h

    cx = clamp(cx, 0.0, 1.0)
    cy = clamp(cy, 0.0, 1.0)
    bw = clamp(bw, 0.0, 1.0)
    bh = clamp(bh, 0.0, 1.0)
    return f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def bbox_from_polygon(points) -> tuple[float, float, float, float] | None:
    # points: [[x,y], [x,y], ...]
    if not isinstance(points, list) or len(points) < 3:
        return None

    xs = []
    ys = []
    for p in points:
        if not (isinstance(p, list) and len(p) >= 2):
            continue
        try:
            x = float(p[0]); y = float(p[1])
        except Exception:
            continue
        xs.append(x); ys.append(y)

    if not xs or not ys:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def convert_one(json_path: Path) -> tuple[bool, int, str | None]:
    """
    returns: (success, boxes_written, reason_if_failed)
    """
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as e:
        return (False, 0, f"json parse fail: {e}")

    info = data.get("IMAGE_INFO", {})
    img_w = info.get("IMAGE_WIDTH")
    img_h = info.get("IMAGE_HEIGHT")
    if not isinstance(img_w, int) or not isinstance(img_h, int) or img_w <= 0 or img_h <= 0:
        return (False, 0, "missing IMAGE_INFO.IMAGE_WIDTH/HEIGHT")

    anns = data.get("ANNOTATION_INFO", [])
    if not isinstance(anns, list):
        return (False, 0, "ANNOTATION_INFO not a list")

    lines = []
    for ann in anns:
        if not isinstance(ann, dict):
            continue

        details = ann.get("DETAILS")
        if not isinstance(details, str):
            continue
        details = details.strip()
        if details not in DETAILS_TO_ID:
            continue
        cid = DETAILS_TO_ID[details]

        st = ann.get("SHAPE_TYPE")

        if st == "BOX":
            pts = ann.get("POINTS")
            # POINTS: [[x,y,w,h]]
            if not (isinstance(pts, list) and len(pts) >= 1 and isinstance(pts[0], list) and len(pts[0]) >= 4):
                continue
            try:
                x, y, w, h = float(pts[0][0]), float(pts[0][1]), float(pts[0][2]), float(pts[0][3])
            except Exception:
                continue
            if w <= 0 or h <= 0:
                continue
            x1, y1, x2, y2 = x, y, x + w, y + h
            line = yolo_from_xyxy(cid, x1, y1, x2, y2, img_w, img_h)
            if line:
                lines.append(line)

        elif st == "POLYGON":
            pts = ann.get("POINTS")
            bb = bbox_from_polygon(pts)
            if not bb:
                continue
            x1, y1, x2, y2 = bb
            line = yolo_from_xyxy(cid, x1, y1, x2, y2, img_w, img_h)
            if line:
                lines.append(line)

        else:
            # 다른 타입은 일단 무시
            continue

    txt_path = json_path.with_suffix(".txt")

    if lines or WRITE_EMPTY_TXT:
        txt_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    if DELETE_JSON:
        try:
            json_path.unlink()
        except Exception as e:
            return (False, len(lines), f"txt ok but json delete fail: {e}")

    return (True, len(lines), None)


def main():
    print("=== JSON -> YOLO TXT (in-place) v2 : BOX + POLYGON ===")
    print("Dataset root:", DATASET_ROOT)
    print("Delete JSON:", DELETE_JSON)
    print("Write empty txt:", WRITE_EMPTY_TXT)
    print()

    total_json = 0
    converted = 0
    failed = 0
    total_boxes = 0

    for label_root in LABEL_DIRS:
        if not label_root.exists():
            print("[WARN] missing label dir:", label_root)
            continue

        json_files = sorted(label_root.rglob("*.json"), key=lambda p: str(p).lower())
        print(f"[INFO] {label_root} | json files: {len(json_files)}")
        total_json += len(json_files)

        for i, jp in enumerate(json_files, start=1):
            ok, n_boxes, reason = convert_one(jp)
            if ok:
                converted += 1
                total_boxes += n_boxes
            else:
                failed += 1
                if failed <= 10:
                    print("[FAIL]", jp, "->", reason)

            if i % 2000 == 0:
                print(f"  progress: {i}/{len(json_files)}")

        print()

    print("=== DONE ===")
    print("total json found:", total_json)
    print("converted:", converted)
    print("failed:", failed)
    print("total boxes written:", total_boxes)


if __name__ == "__main__":
    main()
