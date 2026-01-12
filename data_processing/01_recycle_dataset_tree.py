from pathlib import Path

# === 경로로 바꾸면 됨 ===
ROOT = Path(r"C:\ROKEY\232.재활용품 분류 및 선별 데이터")


def print_tree(root: Path, max_files: int = 5) -> None:
    root = root.resolve()
    if not root.exists():
        print(f"[ERROR] Not found: {root}")
        return
    if not root.is_dir():
        print(f"[ERROR] Not a directory: {root}")
        return

    print(str(root))

    def walk(dir_path: Path, prefix: str = ""):
        # 하위 항목(폴더/파일) 분리
        dirs = sorted([p for p in dir_path.iterdir() if p.is_dir()], key=lambda p: p.name.lower())
        files = sorted([p for p in dir_path.iterdir() if p.is_file()], key=lambda p: p.name.lower())

        # 출력할 "표시 항목" 만들기: 폴더는 전부 + 파일은 상위 max_files
        show_files = files[:max_files]
        file_more = len(files) - len(show_files)

        display_items = dirs + show_files
        total = len(display_items)

        for i, p in enumerate(display_items):
            is_last = (i == total - 1) and (file_more <= 0)
            branch = "└── " if is_last else "├── "

            if p.is_dir():
                print(prefix + branch + p.name + "/")
                extension = "    " if is_last else "│   "
                walk(p, prefix + extension)
            else:
                print(prefix + branch + p.name)

        # 파일 더 있는 경우 표시(폴더 목록 아래에 한 줄 추가)
        if file_more > 0:
            print(prefix + "└── " + f"… (+{file_more} more files)")

    walk(root)


if __name__ == "__main__":
    print_tree(ROOT, max_files=5)
