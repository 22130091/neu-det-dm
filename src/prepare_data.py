#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import Dict, List, Optional, Tuple

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

IMG_EXTS = {".jpg", ".jpeg", ".png"}
OLD_NEU_CLASSES = {"crazing", "inclusion", "patches", "pitted_surface", "rolled_in_scale", "scratches"}


def get_project_root() -> Path:
    current_file = Path(__file__).resolve()
    if current_file.parent.name == "src":
        return current_file.parent.parent
    return Path.cwd()


def clean_breed(folder_name: str) -> str:
    """Chuẩn hóa tên giống chó khớp 100% với notebook Colab."""
    m = re.match(r"^n\d+-(.+)$", folder_name)
    return (m.group(1) if m else folder_name).replace("_", " ").lower()


def class_dirs(d: Path) -> List[Path]:
    return sorted(p for p in d.iterdir() if p.is_dir())


def find_layout(base: Path) -> Tuple[str, Path, Optional[Path]]:
    cands = [base] + sorted((p for p in base.rglob("*") if p.is_dir()), key=lambda p: len(p.parts))
    for c in cands:
        sub = {p.name.lower(): p for p in c.iterdir() if p.is_dir()}
        if "train" in sub and "test" in sub and class_dirs(sub["train"]) and class_dirs(sub["test"]):
            return "split", sub["train"], sub["test"]
    for c in cands:
        sub = {p.name.lower(): p for p in c.iterdir() if p.is_dir()}
        if "images" in sub and class_dirs(sub["images"]):
            return "single", sub["images"], None
    raise FileNotFoundError(f"Không tìm thấy bố cục train/test hoặc Images/ trong {base}")


def scan_images(root: Path, split_name: str) -> pd.DataFrame:
    rows = []
    for cdir in class_dirs(root):
        for p in sorted(cdir.iterdir()):
            if p.suffix.lower() in IMG_EXTS:
                rows.append({
                    "path": str(p),
                    "folder": cdir.name,
                    "label": clean_breed(cdir.name),
                    "orig_split": split_name,
                })
    return pd.DataFrame(rows)


def check_images(df: pd.DataFrame) -> pd.DataFrame:
    corrupted = []
    for path in df["path"]:
        try:
            with Image.open(path) as img:
                img.verify()
        except Exception:
            corrupted.append(path)
    return df[~df["path"].isin(set(corrupted))].reset_index(drop=True)


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    hash_groups: Dict[str, List[int]] = defaultdict(list)
    for idx, path in zip(df.index, df["path"]):
        with open(path, "rb") as f:
            hash_groups[hashlib.sha256(f.read()).hexdigest()].append(idx)

    duplicates = [ids for ids in hash_groups.values() if len(ids) > 1]
    drop_idx = []
    for ids in duplicates:
        in_test = [i for i in ids if df.at[i, "orig_split"] == "test"]
        keep = in_test[0] if in_test else ids[0]
        drop_idx += [i for i in ids if i != keep]
    return df.drop(index=drop_idx).reset_index(drop=True)


def clear_split_dir(split_dir: Path) -> None:
    """Xóa toàn bộ thư mục con cũ bên trong train/val/test trước khi chép dữ liệu mới."""
    if split_dir.exists():
        for item in split_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            elif item.name != ".gitkeep":
                item.unlink()
    split_dir.mkdir(parents=True, exist_ok=True)


def export_split_to_folders(df: pd.DataFrame, target_dir: Path, split_name: str) -> int:
    """Chép ảnh vào thư mục `data/<split>/<tên_giống_chó>/<tên_file>`."""
    clear_split_dir(target_dir)
    copied = 0
    for row in df.itertuples(index=False): 
        breed_folder = row.label.replace(" ", "_")
        dest_class_dir = target_dir / breed_folder
        dest_class_dir.mkdir(parents=True, exist_ok=True)

        src_path = Path(row.path)
        dest_path = dest_class_dir / src_path.name 
        if dest_path.exists():
            dest_path = dest_class_dir / f"{src_path.parent.name}_{src_path.name}"

        shutil.copy2(src_path, dest_path)
        copied += 1

    print(f"  -> Đã xuất {copied:,} ảnh (120 lớp) vào: {target_dir}")
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description="Tải và tạo dữ liệu train/val/test cho Stanford Dogs.")
    parser.add_argument("--dataset", type=str, default="miljan/stanford-dogs-dataset-traintest")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--skip_checks", action="store_true", help="Bỏ qua bước kiểm tra ảnh lỗi/trùng SHA-256")
    args = parser.parse_args()

    project_root = get_project_root()
    data_dir = project_root / "data"
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    # Kiểm tra xem có dữ liệu cũ NEU-DET không
    for d in (train_dir, val_dir, test_dir):
        if d.is_dir():
            existing = {p.name for p in d.iterdir() if p.is_dir()}
            if existing & OLD_NEU_CLASSES:
                print(f"[Thông báo] Phát hiện thư mục đề tài cũ trong `{d}`, sẽ tự động thay thế bằng Stanford Dogs.")

    try:
        import kagglehub
    except ImportError as exc:
        raise ImportError("Vui lòng cài đặt kagglehub trước: pip install kagglehub") from exc

    print(f"1. Đang tải bộ dữ liệu `{args.dataset}` từ Kaggle...")
    dl_path = Path(kagglehub.dataset_download(args.dataset))
    print(f"   Đã tải xong tại: {dl_path}")

    layout, dir_a, dir_b = find_layout(dl_path)
    if layout == "split" and dir_b is not None:
        df = pd.concat([scan_images(dir_a, "train"), scan_images(dir_b, "test")], ignore_index=True)
    else:
        df = scan_images(dir_a, "all")

    classes = sorted(df["label"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    df["y"] = df["label"].map(class_to_idx)

    if not args.skip_checks:
        print("2. Đang kiểm tra ảnh hỏng và lọc trùng SHA-256...")
        df = check_images(df)
        df = remove_duplicates(df)
        print(f"   Tổng số ảnh hợp lệ: {len(df):,} ảnh thuộc {len(classes)} giống chó.")

    print("3. Đang chia tập Train / Validation / Test (seed=42, val_ratio=0.15)...")
    if layout == "split":
        full_train_df = df[df["orig_split"] == "train"]
        test_df = df[df["orig_split"] == "test"]
        train_df, val_df = train_test_split(
            full_train_df,
            test_size=args.val_ratio,
            stratify=full_train_df["y"],
            random_state=args.seed,
        )
    else:
        train_df, tmp_df = train_test_split(df, test_size=0.30, stratify=df["y"], random_state=args.seed)
        val_df, test_df = train_test_split(tmp_df, test_size=0.50, stratify=tmp_df["y"], random_state=args.seed)

    print("4. Đang sao chép ảnh vào thư mục `data/train`, `data/val`, `data/test`...")
    export_split_to_folders(train_df, train_dir, "train")
    export_split_to_folders(val_df, val_dir, "val")
    export_split_to_folders(test_df, test_dir, "test")

    # Lưu class_to_idx.json ngay trong data/ để dùng chung cho evaluate.py và Web Demo
    with open(data_dir / "class_to_idx.json", "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, ensure_ascii=False, indent=2)

    print("\nHOÀN TẤT! Thư mục `data/train`, `data/val`, `data/test` đã có đầy đủ 120 giống chó.")


if __name__ == "__main__":
    main()
