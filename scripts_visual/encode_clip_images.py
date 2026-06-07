"""
encode_clip_images.py
=========================
Encode each product image with the CLIP ViT-L/14 vision tower into a 768-dimensional
L2-normalized embedding (Layer 2 of the three-layer visual encoding described in the paper).

Output is sized 200 x 768 and saved as both parquet and CSV; intermediate per-product
checkpoints (.npy) support resumable runs.
"""

from pathlib import Path
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from transformers import CLIPModel, CLIPProcessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = PROJECT_ROOT / "data" / "images_1024"
VISION_MODEL_DIR = PROJECT_ROOT / "models" / "openaiclip-vit-large-patch14"

OUTPUT_DIR = PROJECT_ROOT / r"data\interim"
CHECKPOINT_DIR = OUTPUT_DIR / "clip_image_checkpoints"

INVENTORY_CSV = OUTPUT_DIR / "image_inventory_200.csv"
PROGRESS_CSV = OUTPUT_DIR / "clip_encoding_progress.csv"
RAW_PARQUET = OUTPUT_DIR / "visual_clip_raw.parquet"
RAW_CSV = OUTPUT_DIR / "visual_clip_raw.csv"
FOR_PCA_CSV = PROJECT_ROOT / r"data\processed\visual_clip_for_pca.csv"
METADATA_JSON = OUTPUT_DIR / "clip_encoding_metadata.json"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

BATCH_SIZE = 8
EXPECTED_DIM = 768


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def list_images(image_dir: Path) -> list[Path]:
    image_files = sorted(
        [
            p for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        ],
        key=lambda p: p.stem
    )

    if not image_files:
        raise FileNotFoundError(f"没有找到图片文件：{image_dir}")

    return image_files


def product_id_from_image(image_path: Path) -> str:
    return image_path.stem.strip()


def checkpoint_path(product_id: str) -> Path:
    return CHECKPOINT_DIR / f"{product_id}.npy"


def is_valid_checkpoint(path: Path) -> bool:
    if not path.exists():
        return False

    try:
        arr = np.load(path)
        return arr.ndim == 1 and arr.shape[0] == EXPECTED_DIM and np.isfinite(arr).all()
    except Exception:
        return False


def load_image(image_path: Path) -> Image.Image:
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img)

    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        bg.alpha_composite(img.convert("RGBA"))
        img = bg.convert("RGB")
    else:
        img = img.convert("RGB")

    return img


def save_inventory(image_files: list[Path]) -> pd.DataFrame:
    rows = []

    for p in image_files:
        with Image.open(p) as img:
            width, height = img.size

        rows.append({
            "product_id": product_id_from_image(p),
            "image_file": p.name,
            "image_path": str(p),
            "width": width,
            "height": height,
            "suffix": p.suffix.lower(),
        })

    inventory = pd.DataFrame(rows)

    if inventory["product_id"].duplicated().any():
        dup = inventory.loc[inventory["product_id"].duplicated(), "product_id"].tolist()
        raise ValueError(f"图片 product_id 重复：{dup[:20]}")

    inventory.to_csv(INVENTORY_CSV, index=False, encoding="utf-8-sig")
    return inventory


def save_progress(inventory: pd.DataFrame) -> None:
    rows = []

    for _, row in inventory.iterrows():
        pid = row["product_id"]
        ckpt = checkpoint_path(pid)

        rows.append({
            "product_id": pid,
            "image_file": row["image_file"],
            "checkpoint_path": str(ckpt),
            "done": is_valid_checkpoint(ckpt),
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

    pd.DataFrame(rows).to_csv(PROGRESS_CSV, index=False, encoding="utf-8-sig")


def load_model() -> tuple[CLIPModel, CLIPProcessor, torch.device]:
    device = get_device()

    model = CLIPModel.from_pretrained(
        VISION_MODEL_DIR,
        local_files_only=True,
        use_safetensors=True,
    )
    processor = CLIPProcessor.from_pretrained(
        VISION_MODEL_DIR,
        local_files_only=True,
    )

    model.eval()
    model.to(device)

    return model, processor, device


@torch.no_grad()
def encode_batch(
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
    batch_paths: list[Path],
) -> np.ndarray:
    images = [load_image(p) for p in batch_paths]

    inputs = processor(
        images=images,
        return_tensors="pt",
    )

    pixel_values = inputs["pixel_values"].to(device)

    vision_outputs = model.vision_model(pixel_values=pixel_values)
    pooled_output = vision_outputs.pooler_output

    image_features = model.visual_projection(pooled_output)

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    return image_features.detach().cpu().numpy()




def encode_with_resume(
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
    inventory: pd.DataFrame,
) -> None:
    pending_paths = []

    for _, row in inventory.iterrows():
        pid = row["product_id"]
        image_path = Path(row["image_path"])

        if is_valid_checkpoint(checkpoint_path(pid)):
            print(f"[跳过] {pid} 已有checkpoint")
            continue

        pending_paths.append(image_path)

    print(f"待编码图片数：{len(pending_paths)}")

    total = len(pending_paths)
    done = 0

    for start in range(0, total, BATCH_SIZE):
        batch_paths = pending_paths[start:start + BATCH_SIZE]

        try:
            vectors = encode_batch(model, processor, device, batch_paths)

            for image_path, vector in zip(batch_paths, vectors):
                pid = product_id_from_image(image_path)
                np.save(checkpoint_path(pid), vector.astype(np.float32))

            done += len(batch_paths)
            save_progress(inventory)

            print(f"[完成] {done}/{total} | 本批 {len(batch_paths)} 张")

        except Exception as e:
            save_progress(inventory)
            print(f"[失败] batch start={start} | {e}")
            raise


def build_outputs(inventory: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, row in inventory.iterrows():
        pid = row["product_id"]
        ckpt = checkpoint_path(pid)

        if not is_valid_checkpoint(ckpt):
            raise ValueError(f"缺少有效checkpoint：{pid} | {ckpt}")

        vec = np.load(ckpt).astype(np.float32)

        item = {
            "product_id": pid,
            "image_file": row["image_file"],
            "image_path": row["image_path"],
        }

        for i, value in enumerate(vec):
            item[f"clip_dim_{i:03d}"] = float(value)

        rows.append(item)

    df = pd.DataFrame(rows)
    dim_cols = [c for c in df.columns if c.startswith("clip_dim_")]

    if len(dim_cols) != EXPECTED_DIM:
        raise ValueError(f"CLIP维度异常：{len(dim_cols)}，预期 {EXPECTED_DIM}")

    df.to_parquet(RAW_PARQUET, index=False)
    df.to_csv(RAW_CSV, index=False, encoding="utf-8-sig")

    FOR_PCA_CSV.parent.mkdir(parents=True, exist_ok=True)
    df[["product_id"] + dim_cols].to_csv(FOR_PCA_CSV, index=False, encoding="utf-8-sig")

    return df


def save_metadata(inventory: pd.DataFrame, output_df: pd.DataFrame, device: torch.device) -> None:
    dim_cols = [c for c in output_df.columns if c.startswith("clip_dim_")]

    metadata = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image_dir": str(IMAGE_DIR),
        "vision_model_dir": str(VISION_MODEL_DIR),
        "vision_model_name": "openai/clip-vit-large-patch14",
        "image_count": int(len(inventory)),
        "embedding_dim": int(len(dim_cols)),
        "embedding_norm": "L2 normalized",
        "device": str(device),
        "batch_size": BATCH_SIZE,
        "outputs": {
            "inventory_csv": str(INVENTORY_CSV),
            "progress_csv": str(PROGRESS_CSV),
            "raw_parquet": str(RAW_PARQUET),
            "raw_csv": str(RAW_CSV),
            "for_pca_csv": str(FOR_PCA_CSV),
            "checkpoint_dir": str(CHECKPOINT_DIR),
        },
    }

    with open(METADATA_JSON, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def main() -> None:
    start_time = time.time()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    image_files = list_images(IMAGE_DIR)
    inventory = save_inventory(image_files)

    print(f"图片数量：{len(inventory)}")
    print(f"图片目录：{IMAGE_DIR}")
    print(f"模型目录：{VISION_MODEL_DIR}")

    model, processor, device = load_model()
    print(f"使用设备：{device}")

    encode_with_resume(model, processor, device, inventory)

    output_df = build_outputs(inventory)
    save_progress(inventory)
    save_metadata(inventory, output_df, device)

    elapsed = time.time() - start_time

    print("\nCLIP图片编码完成")
    print(f"输出 parquet：{RAW_PARQUET}")
    print(f"输出 csv：{RAW_CSV}")
    print(f"PCA输入文件：{FOR_PCA_CSV}")
    print(f"checkpoint目录：{CHECKPOINT_DIR}")
    print(f"耗时秒数：{elapsed:.1f}")


if __name__ == "__main__":
    main()
