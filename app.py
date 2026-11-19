import json
import sys
import time
from pathlib import Path

import streamlit as st
import torch
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.evaluate_stanford_dogs import (
    build_and_load_model,
    discover_weights,
    get_eval_transform,
    load_config,
)

# cấu hình trang
st.set_page_config(
    page_title="Phân Loại 120 Giống Chó | Stanford Dogs",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# css
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1180px;
    }

    /*  tiêu đề */
    .hero-banner {
        background: #0f172a;
        padding: 1.6rem 2.2rem;
        border-radius: 12px;
        color: #f8fafc;
        margin-bottom: 1.8rem;
        border: 1px solid #1e293b;
    }
    .hero-banner h1 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 700;
        letter-spacing: -0.4px;
    }
    .badge-row {
        margin-top: 0.9rem;
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
    }
    .tech-badge {
        background: #1e293b;
        border: 1px solid #334155;
        padding: 0.2rem 0.7rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 500;
        color: #e2e8f0;
    }

    /* thẻ kết quả cho từng mô hình */
    .model-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.35rem;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
    }
    .model-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #f1f5f9;
        padding-bottom: 0.7rem;
        margin-bottom: 1rem;
    }
    .model-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0;
    }
    .latency-pill {
        background: #f1f5f9;
        color: #334155;
        font-size: 0.8rem;
        font-weight: 600;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
    }
    .top1-box {
        background: #f8fafc;
        border-left: 4px solid #0f172a;
        border-top: 1px solid #e2e8f0;
        border-right: 1px solid #e2e8f0;
        border-bottom: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.85rem 1rem;
        margin-bottom: 1rem;
    }
    .top1-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        font-weight: 700;
        color: #64748b;
        letter-spacing: 0.5px;
    }
    .top1-breed {
        font-size: 1.2rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0.15rem 0;
    }
    .top1-conf {
        font-size: 0.9rem;
        font-weight: 600;
        color: #2563eb;
    }

    /* thanh bar top 5 */
    .sub-pred-row {
        margin-bottom: 0.65rem;
    }
    .sub-pred-text {
        display: flex;
        justify-content: space-between;
        font-size: 0.86rem;
        color: #475569;
        margin-bottom: 0.2rem;
    }
    .bar-bg {
        width: 100%;
        height: 6px;
        background: #f1f5f9;
        border-radius: 4px;
        overflow: hidden;
    }
    .bar-fill {
        height: 100%;
        background: #475569;
        border-radius: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_MODELS = ["resnet50", "efficientnet_b0", "swin_tiny"]


@st.cache_resource
def load_breed_labels() -> list[str]:
    mapping_path = ROOT_DIR / "evaluate_results" / "class_mapping_120.json"
    if not mapping_path.is_file():
        st.error(f"Không tìm thấy file nhãn tại: {mapping_path}")
        st.stop()
    data = json.loads(mapping_path.read_text(encoding="utf-8"))
    return [data[str(i)]["breed"] for i in range(len(data))]


@st.cache_resource
def load_three_models():
    """nạp trọng số của 3 mô hình"""
    all_jobs = discover_weights(ROOT_DIR / "weights")
    job_map = {key: (key, label, path) for key, label, path in all_jobs if key in TARGET_MODELS}

    ordered_models = []
    for key in TARGET_MODELS:
        if key in job_map:
            m_key, display_name, weights_path = job_map[key]
            model = build_and_load_model(
                model_name=m_key,
                weights_path=weights_path,
                num_classes=120,
                device=DEVICE,
            )
            ordered_models.append({
                "key": m_key,
                "name": display_name,
                "model": model,
            })

    if not ordered_models:
        st.error("Không tìm thấy trọng số của 3 mô hình trong thư mục weights/.")
        st.stop()
    return ordered_models


def predict_single_model(model: torch.nn.Module, input_tensor: torch.Tensor, breed_names: list[str]):
    with torch.no_grad():
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        logits = model(input_tensor)

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        probs = torch.softmax(logits, dim=1)[0]
        top5_probs, top5_indices = torch.topk(probs, k=5)

    top5 = [
        {"breed": breed_names[int(idx)], "confidence": float(prob) * 100.0}
        for prob, idx in zip(top5_probs, top5_indices)
    ]
    return top5, latency_ms


def render_model_card(out: dict):
    """vẽ thẻ kết quả sạch sẽ cho 1 mô hình."""
    top1 = out["top5"][0]
    others = out["top5"][1:]

    sub_rows_html = ""
    for rank, item in enumerate(others, start=2):
        conf = item["confidence"]
        bar_width = min(max(conf, 2.0), 100.0)
        sub_rows_html += (
            f'<div class="sub-pred-row">'
            f'<div class="sub-pred-text">'
            f'<span><b>#{rank}</b> {item["breed"]}</span>'
            f'<span><b>{conf:.2f}%</b></span>'
            f'</div>'
            f'<div class="bar-bg">'
            f'<div class="bar-fill" style="width: {bar_width:.1f}%;"></div>'
            f'</div>'
            f'</div>'
        )

    card_html = (
        f'<div class="model-card">'
        f'<div class="model-header">'
        f'<h3 class="model-title">{out["name"]}</h3>'
        f'<span class="latency-pill">{out["latency_ms"]:.1f} ms</span>'
        f'</div>'
        f'<div class="top1-box">'
        f'<div class="top1-label">Dự đoán cao nhất (Top-1)</div>'
        f'<div class="top1-breed">{top1["breed"]}</div>'
        f'<div class="top1-conf">Độ tin cậy: {top1["confidence"]:.2f}%</div>'
        f'</div>'
        f'{sub_rows_html}'
        f'</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)

def main():
    # Banner đầu trang 
    st.markdown(
        """
        <div class="hero-banner">
            <h1>Phân Loại & So Sánh 120 Giống Chó (Stanford Dogs)</h1>
            <div class="badge-row">
                <span class="tech-badge">ResNet-50</span>
                <span class="tech-badge">EfficientNet-B0</span>
                <span class="tech-badge">Swin-Tiny</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    breed_names = load_breed_labels()
    cfg = load_config(ROOT_DIR / "config.yaml")
    transform = get_eval_transform(
        resize_size=int(cfg.get("resize_size", 256)),
        img_size=int(cfg.get("img_size", 224)),
        mean=cfg.get("imagenet_mean", [0.485, 0.456, 0.406]),
        std=cfg.get("imagenet_std", [0.229, 0.224, 0.225]),
    )

    with st.spinner("Đang nạp trọng số 3 mô hình..."):
        models_list = load_three_models()

    # Upload & Xem trước ảnh
    left_col, right_col = st.columns([1.15, 1], gap="large")

    with left_col:
        st.subheader("Tải ảnh đầu vào")
        uploaded_file = st.file_uploader(
            "Chọn ảnh chú chó (JPG, PNG, JPEG)",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )
        run_btn = st.button(
            "Phân loại với 3 mô hình",
            type="primary",
            use_container_width=True,
            disabled=(uploaded_file is None),
        )

    with right_col:
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(
                image,
                caption=f"Ảnh kiểm thử ({image.size[0]}x{image.size[1]} px)",
                use_container_width=True,
            )
        else:
            st.info("Chọn một tệp ảnh ở khung bên trái để bắt đầu phân loại.")

    # hiển thị kết quả của 3 mô hình
    if run_btn and uploaded_file is not None:
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)

        all_outputs = []
        for item in models_list:
            top5, latency_ms = predict_single_model(item["model"], input_tensor, breed_names)
            all_outputs.append({
                "name": item["name"],
                "top5": top5,
                "latency_ms": latency_ms,
            })

        st.markdown("---")
        st.subheader("Kết quả dự đoán")

        cols = st.columns(len(all_outputs), gap="medium")
        for col, out in zip(cols, all_outputs):
            with col:
                render_model_card(out)


if __name__ == "__main__":
    main()