#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kịch bản đánh giá (Evaluation) cho bài toán Phân loại 120 giống chó (Stanford Dogs).
Đặt tại: src/evaluate.py

Khớp toàn bộ thiết lập trong báo cáo (Mục IV):
  - Tập Test: 8,580 ảnh thuộc 120 lớp (ánh xạ nhãn 0-119 theo thứ tự chữ cái).
  - Tiền xử lý Validation/Test: Resize(256) -> CenterCrop(224) -> Normalize(ImageNet), không dùng ngẫu nhiên.
  - Batch size: 32 (269 batch), seed: 42.
  - Đánh giá 3 mô hình: ResNet-50, EfficientNet-B0, Swin-Tiny.
  - Xuất đầy đủ chỉ số Bảng III: Acc (%), Prec (%), Rec (%), F1 (%), FLOPs (G), Latency (ms/ảnh), Params (M),
    ma trận nhầm lẫn 120x120, top cặp giống chó dễ nhầm lẫn và thống kê min/max/trung bình ảnh/lớp (Mục IV.A.1).
"""

import argparse
import ast
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

 
#  ĐỌC CONFIG.YAML & CỐ ĐỊNH SEED 
def get_project_root() -> Path:
    """Tự động xác định thư mục gốc dự án dù chạy từ root hay từ thư mục `src/`."""
    current_file = Path(__file__).resolve()
    if current_file.parent.name == "src":
        return current_file.parent.parent
    return Path.cwd()


def _parse_simple_yaml(text: str) -> Dict[str, Any]:
    """Bộ phân tích YAML dự phòng khi môi trường chưa cài thư viện `pyyaml`."""
    root: Dict[str, Any] = {}
    current_dict: Optional[Dict[str, Any]] = None

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue

        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()

        if ":" in stripped:
            key, val = stripped.split(":", 1)
            key = key.strip()
            val = val.strip()

            if indent == 0:
                if val == "":
                    root[key] = {}
                    current_dict = root[key]
                else:
                    current_dict = None
                    try:
                        root[key] = ast.literal_eval(val)
                    except Exception:
                        root[key] = val.strip("'\"")
            elif current_dict is not None:
                try:
                    current_dict[key] = ast.literal_eval(val)
                except Exception:
                    current_dict[key] = val.strip("'\"")
    return root


def load_config(config_path: Path) -> Dict[str, Any]:
    """Đọc tệp config.yaml nếu tồn tại, ngược lại dùng giá trị mặc định của báo cáo."""
    if not config_path.is_file():
        return {}
    raw_text = config_path.read_text(encoding="utf-8")
    try:
        import yaml
        return yaml.safe_load(raw_text) or {}
    except ImportError:
        return _parse_simple_yaml(raw_text)


def set_seed(seed: int = 42) -> None:
    """Cố định seed theo Mục IV.A.2 (seed = 42)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

 
#   TIỀN XỬ LÝ TẬP TEST & THỐNG KÊ PHÂN BỐ 
def get_eval_transform(
    resize_size: int = 256,
    img_size: int = 224,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225],
) -> transforms.Compose:
    """Chuỗi biến đổi chuẩn cho tập Validation/Test (Mục IV.A.2):
    Resize(256) -> CenterCrop(224) -> Normalize(ImageNet), không dùng ngẫu nhiên.
    """
    return transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def format_breed_name(folder_name: str) -> str:
    """Chuyển tên thư mục chuẩn Stanford Dogs (vd: 'n02085620-Chihuahua') thành tên giống chó dễ đọc."""
    if "-" in folder_name and folder_name.startswith("n0"):
        return folder_name.split("-", 1)[1].replace("_", " ")
    return folder_name.replace("_", " ")


def compute_dataset_stats(data_dir: Path, split_name: str) -> Dict[str, Any]:
    """Tính số lượng ảnh, số lớp, min, max, trung bình ảnh/lớp để điền Mục IV.A.1."""
    dataset = datasets.ImageFolder(root=str(data_dir))
    targets = np.array(dataset.targets)
    _, counts = np.unique(targets, return_counts=True)

    return {
        "split": split_name,
        "total_images": int(len(dataset)),
        "num_classes": int(len(dataset.classes)),
        "min_images_per_class": int(counts.min()) if len(counts) > 0 else 0,
        "max_images_per_class": int(counts.max()) if len(counts) > 0 else 0,
        "mean_images_per_class": round(float(counts.mean()), 2) if len(counts) > 0 else 0.0,
    }

 
#  KHỞI TẠO KIẾN TRÚC & NẠP TRỌNG SỐ 
def clean_state_dict(raw_ckpt: Any) -> Dict[str, torch.Tensor]:
    """Chuẩn hóa state_dict (hỗ trợ cả checkpoint lưu full dict hoặc DDP/DataParallel)."""
    if isinstance(raw_ckpt, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            if key in raw_ckpt and isinstance(raw_ckpt[key], dict):
                raw_ckpt = raw_ckpt[key]
                break

    cleaned = {}
    for k, v in raw_ckpt.items():
        new_k = k
        for prefix in ("module.", "model."):
            if new_k.startswith(prefix):
                new_k = new_k[len(prefix):]
        cleaned[new_k] = v
    return cleaned


def build_and_load_model(
    model_name: str,
    weights_path: Path,
    num_classes: int = 120,
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    """Khởi tạo kiến trúc mô hình với `num_classes=120` và nạp trọng số từ .pt / .pth."""
    name = model_name.lower().strip()
    state_dict = clean_state_dict(torch.load(str(weights_path), map_location="cpu"))

    if name in ("efficientnet_b0", "efficientnet-b0", "effnet_b0"):
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        model.load_state_dict(state_dict, strict=True)

    elif name in ("resnet50", "resnet-50"):
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        model.load_state_dict(state_dict, strict=True)

    elif name in ("swin_tiny", "swin-tiny", "swin_t"):
        is_timm = any(k.startswith("layers.") or k.startswith("patch_embed.") for k in state_dict.keys())
        if is_timm:
            try:
                import timm
                model = timm.create_model(
                    "swin_tiny_patch4_window7_224",
                    pretrained=False,
                    num_classes=num_classes,
                )
                model.load_state_dict(state_dict, strict=True)
            except ImportError as exc:
                raise ImportError(
                    "Trọng số Swin-Tiny thuộc định dạng thư viện `timm`. Cài đặt bằng: pip install timm"
                ) from exc
        else:
            model = models.swin_t(weights=None)
            in_features = model.head.in_features
            model.head = nn.Linear(in_features, num_classes)
            model.load_state_dict(state_dict, strict=True)
    else:
        raise ValueError(
            f"Mô hình '{model_name}' không được hỗ trợ. "
            "Chọn một trong: efficientnet_b0, resnet50, swin_tiny."
        )

    model.to(device)
    model.eval()
    return model


def discover_weights(weights_dir: Path) -> List[Tuple[str, str, Path]]:
    """Tự động dò tìm các file trọng số trong thư mục `weights/`."""
    if not weights_dir.is_dir():
        return []

    weight_files = sorted(
        list(weights_dir.glob("*.pth")) + list(weights_dir.glob("*.pt"))
    )
    jobs: List[Tuple[str, str, Path]] = []
    seen_keys = set()

    for wf in weight_files:
        stem = wf.stem.lower()
        if "resnet" in stem:
            key, label = "resnet50", "ResNet-50"
        elif "efficientnet" in stem or "effnet" in stem:
            key, label = "efficientnet_b0", "EfficientNet-B0"
        elif "swin" in stem:
            key, label = "swin_tiny", "Swin-Tiny"
        else:
            continue

        if key not in seen_keys:
            jobs.append((key, label, wf))
            seen_keys.add(key)

    return jobs

 
#   ĐO HIỆU NĂNG TÍNH TOÁN: PARAMS (M), FLOPS (G), LATENCY (MS/ẢNH) 
def compute_params_and_flops(
    model: nn.Module,
    device: torch.device,
    img_size: int = 224,
) -> Tuple[float, float]:
    """Tính số lượng tham số (Triệu - M) và GigaFLOPs (G)."""
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    dummy_input = torch.randn(1, 3, img_size, img_size, device=device)

    try:
        from thop import profile
        macs, _ = profile(model, inputs=(dummy_input,), verbose=False)
        gflops = (macs * 2.0) / 1e9
        return round(params_m, 2), round(gflops, 2)
    except Exception:
        pass

    try:
        from torch.utils.flop_counter import FlopCounterMode
        with torch.no_grad(), FlopCounterMode(display=False) as flop_counter:
            _ = model(dummy_input)
        gflops = flop_counter.get_total_flops() / 1e9
        return round(params_m, 2), round(gflops, 2)
    except Exception:
        return round(params_m, 2), -1.0


def measure_single_image_latency(
    model: nn.Module,
    device: torch.device,
    img_size: int = 224,
    warmup_runs: int = 30,
    measure_runs: int = 200,
) -> float:
    """Đo độ trễ suy luận trung bình trên từng ảnh đơn lẻ (ms/ảnh)."""
    dummy_input = torch.randn(1, 3, img_size, img_size, device=device)

    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()
        for _ in range(measure_runs):
            _ = model(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

    return round((elapsed / measure_runs) * 1000.0, 2)

 
#  ĐÁNH GIÁ & XUẤT FILE KẾT QUẢ  
def save_confusion_matrix_plot(
    cm: np.ndarray,
    output_png: Path,
    model_label: str,
) -> None:
    """Vẽ và lưu biểu đồ Confusion Matrix 120x120."""
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 10), dpi=200)
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(f"Confusion Matrix (120 Dog Breeds) - {model_label}", fontsize=13, pad=12)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlabel(f"Predicted Breed ID (0 - {cm.shape[0] - 1})", fontsize=11)
        ax.set_ylabel(f"True Breed ID (0 - {cm.shape[0] - 1})", fontsize=11)
        plt.tight_layout()
        plt.savefig(str(output_png))
        plt.close(fig)
    except Exception as exc:
        print(f"[Cảnh báo] Không thể vẽ biểu đồ Confusion Matrix: {exc}")


def extract_top_confused_pairs(
    cm: np.ndarray,
    breed_names: List[str],
    top_k: int = 15,
) -> pd.DataFrame:
    """Trích xuất Top-K cặp giống chó hay bị nhận diện nhầm nhất để phân tích trong báo cáo."""
    cm_off_diag = cm.copy()
    np.fill_diagonal(cm_off_diag, 0)

    flat_indices = np.argsort(cm_off_diag.ravel())[::-1][:top_k]
    rows = []
    for flat_idx in flat_indices:
        true_id, pred_id = divmod(int(flat_idx), cm.shape[1])
        err_count = int(cm_off_diag[true_id, pred_id])
        if err_count == 0:
            break
        total_true = int(cm[true_id].sum())
        rows.append({
            "Giống chó thực tế (True)": breed_names[true_id],
            "Bị dự đoán nhầm thành (Predicted)": breed_names[pred_id],
            "Số ảnh nhầm": err_count,
            "Tổng ảnh lớp thực tế": total_true,
            "Tỷ lệ nhầm (%)": round((err_count / total_true * 100.0) if total_true > 0 else 0.0, 2),
        })
    return pd.DataFrame(rows)


def evaluate_single_model(
    model_key: str,
    display_name: str,
    weights_path: Path,
    test_dataset: datasets.ImageFolder,
    test_loader: DataLoader,
    img_size: int,
    device: torch.device,
    output_dir: Path,
) -> Dict[str, Any]:
    """Chạy đánh giá đầy đủ cho 1 mô hình và lưu toàn bộ file đầu ra."""
    print(f"\n{'=' * 68}")
    print(f"ĐANG ĐÁNH GIÁ MÔ HÌNH: {display_name}")
    print(f"Tệp trọng số        : {weights_path}")
    print(f"{'=' * 68}")

    raw_classes = test_dataset.classes
    breed_names = [format_breed_name(c) for c in raw_classes]
    num_classes = len(raw_classes)

    model = build_and_load_model(model_key, weights_path, num_classes=num_classes, device=device)

    params_m, gflops = compute_params_and_flops(model, device, img_size=img_size)
    single_latency_ms = measure_single_image_latency(model, device, img_size=img_size)

    all_preds: List[int] = []
    all_labels: List[int] = []
    all_confs: List[float] = []
    top5_correct = 0
    batch_infer_time = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            logits = model(images)

            if device.type == "cuda":
                torch.cuda.synchronize()
            batch_infer_time += time.perf_counter() - t0

            probs = torch.softmax(logits, dim=1)
            confs, preds = torch.max(probs, dim=1)

            # Tính  Top-5 Accuracy
            k_top = min(5, num_classes)
            _, top5_preds = torch.topk(probs, k=k_top, dim=1)
            top5_correct += int((top5_preds.cpu() == labels.unsqueeze(1)).any(dim=1).sum().item())

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
            all_confs.extend(confs.cpu().numpy().tolist())

    total_samples = len(all_labels)
    acc = accuracy_score(all_labels, all_preds) * 100.0
    top5_acc = (top5_correct / total_samples) * 100.0
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    prec, rec, f1 = prec * 100.0, rec * 100.0, f1 * 100.0
    batch_latency_ms = (batch_infer_time / total_samples) * 1000.0

    #   Xuất dự đoán chi tiết từng ảnh  
    pred_rows = []
    for (img_path, _), true_id, pred_id, conf in zip(
        test_dataset.samples, all_labels, all_preds, all_confs
    ):
        pred_rows.append({
            "image_path": img_path,
            "true_label_id": true_id,
            "true_breed": breed_names[true_id],
            "pred_label_id": pred_id,
            "pred_breed": breed_names[pred_id],
            "confidence": round(conf, 4),
            "is_correct": int(true_id == pred_id),
        })
    pd.DataFrame(pred_rows).to_csv(
        output_dir / f"evaluate_predictions_{model_key}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    #   Xuất báo cáo chi tiết 120 giống chó
    cls_report = classification_report(
        all_labels,
        all_preds,
        target_names=breed_names,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(cls_report).transpose().to_csv(
        output_dir / f"classification_report_{model_key}.csv",
        encoding="utf-8-sig",
    )

    #  Xuất Confusion Matrix 120x120 (.npy, .csv, .png)  
    cm = confusion_matrix(all_labels, all_preds)
    np.save(output_dir / f"confusion_matrix_{model_key}.npy", cm)
    pd.DataFrame(cm, index=breed_names, columns=breed_names).to_csv(
        output_dir / f"confusion_matrix_{model_key}.csv",
        encoding="utf-8-sig",
    )
    save_confusion_matrix_plot(
        cm,
        output_dir / f"confusion_matrix_{model_key}.png",
        display_name,
    )

    df_confused = extract_top_confused_pairs(cm, breed_names, top_k=15)
    df_confused.to_csv(
        output_dir / f"top_confused_breeds_{model_key}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    #  Lưu danh sách ánh xạ nhãn 0-119  
    label_map = {str(idx): {"folder": raw_classes[idx], "breed": breed_names[idx]} for idx in range(num_classes)}
    with open(output_dir / "class_mapping_120.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=4, ensure_ascii=False)

    summary = {
        "Mô hình": display_name,
        "Model Key": model_key,
        "Số ảnh Test": total_samples,
        "Acc (%)": round(acc, 2),
        "Top-5 Acc (%)": round(top5_acc, 2),
        "Prec (%)": round(prec, 2),
        "Rec (%)": round(rec, 2),
        "F1 (%)": round(f1, 2),
        "FLOPs (G)": gflops if gflops >= 0 else "N/A",
        "Latency (ms/ảnh)": single_latency_ms,
        "Batch Latency (ms/ảnh)": round(batch_latency_ms, 2),
        "Params (M)": params_m,
    }

    with open(output_dir / f"evaluate_metrics_{model_key}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary

 
#   HÀM CHÍNH (CLI) 
def parse_args(project_root: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Đánh giá so sánh ResNet-50, EfficientNet-B0 và Swin-Tiny trên Stanford Dogs (120 giống chó)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(project_root / "config.yaml"),
        help="Đường dẫn tệp config.yaml (mặc định: <root>/config.yaml).",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default=None,
        help="Đường dẫn thư mục tập Test (mặc định đọc từ config.yaml hoặc data/test).",
    )
    parser.add_argument(
        "--train_dir",
        type=str,
        default=None,
        help="Đường dẫn tập Train để thống kê phân bố (mặc định đọc từ config.yaml hoặc data/train).",
    )
    parser.add_argument(
        "--val_dir",
        type=str,
        default=None,
        help="Đường dẫn tập Validation để thống kê phân bố (mặc định đọc từ config.yaml hoặc data/val).",
    )
    parser.add_argument(
        "--weights_dir",
        type=str,
        default=str(project_root / "weights"),
        help="Thư mục chứa các file trọng số .pth (mặc định: <root>/weights).",
    )
    parser.add_argument(
        "--effnet_weights",
        type=str,
        default=None,
        help="Đường dẫn chỉ định riêng trọng số EfficientNet-B0 (.pt / .pth).",
    )
    parser.add_argument(
        "--resnet_weights",
        type=str,
        default=None,
        help="Đường dẫn chỉ định riêng trọng số ResNet-50 (.pt / .pth).",
    )
    parser.add_argument(
        "--swin_weights",
        type=str,
        default=None,
        help="Đường dẫn chỉ định riêng trọng số Swin-Tiny (.pt / .pth).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Kích thước batch đánh giá (mặc định: 32 theo Bảng II).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Số luồng tải dữ liệu DataLoader (mặc định: 4).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Thư mục lưu toàn bộ file kết quả đánh giá (mặc định: <root>/evaluate_results).",
    )
    return parser.parse_args()


def main() -> None:
    project_root = get_project_root()
    args = parse_args(project_root)

    cfg = load_config(Path(args.config))
    data_cfg = cfg.get("data", {}) if isinstance(cfg.get("data"), dict) else {}
    weights_cfg = cfg.get("weights", {}) if isinstance(cfg.get("weights"), dict) else {}

    seed = int(cfg.get("seed", 42))
    img_size = int(cfg.get("img_size", 224))
    resize_size = int(cfg.get("resize_size", 256))
    batch_size = int(args.batch_size or cfg.get("batch_size", 32))
    mean = cfg.get("imagenet_mean", [0.485, 0.456, 0.406])
    std = cfg.get("imagenet_std", [0.229, 0.224, 0.225])

    train_dir = Path(args.train_dir or (project_root / data_cfg.get("train_dir", "data/train")))
    val_dir = Path(args.val_dir or (project_root / data_cfg.get("val_dir", "data/val")))
    test_dir = Path(args.test_dir or (project_root / data_cfg.get("test_dir", "data/test")))
    output_dir = Path(args.output_dir or (project_root / cfg.get("output_dir", "evaluate_results")))

    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Thiết bị thực thi : {device}")
    print(f"Tiền xử lý Test   : Resize({resize_size}) -> CenterCrop({img_size}), batch_size={batch_size}, seed={seed}")

    
    dataset_stats = {}
    for split_name, p in [("train", train_dir), ("validation", val_dir), ("test", test_dir)]:
        if p.is_dir():
            dataset_stats[split_name] = compute_dataset_stats(p, split_name)

    with open(output_dir / "dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump(dataset_stats, f, indent=4, ensure_ascii=False)
    print("\nThống kê tập dữ liệu:", json.dumps(dataset_stats, indent=2, ensure_ascii=False))

  
    if not test_dir.is_dir():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục tập Test tại: {test_dir}. "
            "Vui lòng đặt dữ liệu vào `data/test` hoặc truyền `--test_dir`."
        )

    test_transform = get_eval_transform(
        resize_size=resize_size,
        img_size=img_size,
        mean=mean,
        std=std,
    )
    test_dataset = datasets.ImageFolder(root=str(test_dir), transform=test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

   
    model_jobs: List[Tuple[str, str, Path]] = []
    if args.resnet_weights:
        model_jobs.append(("resnet50", "ResNet-50", Path(args.resnet_weights)))
    if args.effnet_weights:
        model_jobs.append(("efficientnet_b0", "EfficientNet-B0", Path(args.effnet_weights)))
    if args.swin_weights:
        model_jobs.append(("swin_tiny", "Swin-Tiny", Path(args.swin_weights)))

    
    if not model_jobs and weights_cfg:
        for key, label in [
            ("resnet50", "ResNet-50"),
            ("efficientnet_b0", "EfficientNet-B0"),
            ("swin_tiny", "Swin-Tiny"),
        ]:
            if key in weights_cfg:
                cand = project_root / weights_cfg[key]
                if cand.is_file():
                    model_jobs.append((key, label, cand))

    
    if not model_jobs:
        model_jobs = discover_weights(Path(args.weights_dir))

    if not model_jobs:
        raise ValueError(
            f"Không tìm thấy tệp trọng số nào trong `{args.weights_dir}`. "
            "Hãy đặt các file .pth (resnet50.pth, efficientnet_b0.pth, swin_tiny.pth) "
            "vào thư mục `weights/` hoặc truyền tham số --effnet_weights / --resnet_weights / --swin_weights."
        )

    #   Chạy đánh giá và tổng hợp Bảng  
    all_summaries = []
    for model_key, display_name, weights_path in model_jobs:
        if not weights_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy file trọng số: {weights_path}")
        summary = evaluate_single_model(
            model_key=model_key,
            display_name=display_name,
            weights_path=weights_path,
            test_dataset=test_dataset,
            test_loader=test_loader,
            img_size=img_size,
            device=device,
            output_dir=output_dir,
        )
        all_summaries.append(summary)

    df_summary = pd.DataFrame(all_summaries)
    df_summary.to_csv(output_dir / "evaluate_summary_table3.csv", index=False, encoding="utf-8-sig")
    with open(output_dir / "evaluate_summary_table3.json", "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=4, ensure_ascii=False)

    print(f"\n{'=' * 68}")
    print("BẢNG TỔNG HỢP KẾT QUẢ THỰC NGHIỆM (BẢNG III - STANFORD DOGS)")
    print(f"{'=' * 68}")
    print(df_summary.to_string(index=False))
    print(f"\nToàn bộ file đánh giá đã được xuất tại thư mục: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
