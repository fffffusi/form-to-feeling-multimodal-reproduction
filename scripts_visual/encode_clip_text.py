"""
encode_clip_text.py
=========================
Compute CLIP-text concept scores per product (Layer 3 of the three-layer visual encoding).

For each affective concept (ME1..ME5), three positive and three negative Chinese prompts
are encoded with the Chinese Taiyi CLIP text encoder. Each product's image embedding is
then compared with the positive and negative centroids, and the concept score is defined
as the difference of mean cosine similarities:

    concept_score = mean_cos(image, positive_prompts) - mean_cos(image, negative_prompts)

The five affective concepts follow the paper's Appendix A.3 naming:
  ME1 Perceived Premiumness   (premium vs. cheap)
  ME2 Visual Solidity         (solid/substantial vs. flimsy)
  ME3 Minimalist Simplicity   (minimal/clean vs. cluttered)
  ME4 Aesthetic Refinement    (refined/beautiful vs. rough/ugly)
  ME5 Tech-Modernity          (tech-modern vs. old-fashioned)
"""

from pathlib import Path
import json
import re
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from transformers import BertForSequenceClassification, BertTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REVIEWS_XLSX = PROJECT_ROOT / "data" / "input" / "product_reviews_clean.xlsx"
IMAGE_CLIP_PARQUET = PROJECT_ROOT / r"data\interim\visual_clip_raw.parquet"

TAIYI_TEXT_MODEL_DIR = PROJECT_ROOT / "models" / "IDEA-CCNLTaiyi-CLIP-Roberta-large-326M-Chinese"

OUTPUT_DIR = PROJECT_ROOT / r"data\interim"
OUTPUT_PARQUET = OUTPUT_DIR / "visual_concept_scores.parquet"
OUTPUT_CSV = OUTPUT_DIR / "visual_concept_scores.csv"
PROMPT_CSV = OUTPUT_DIR / "visual_concept_prompts.csv"
METADATA_JSON = OUTPUT_DIR / "visual_concept_metadata.json"

NO_COL = "no"
PRODUCT_ID_COL = "product_id"
EXPECTED_DIM = 768


# Affective concept prompts.
# Each entry's key (e.g. "ME1_高级_廉价") becomes a column prefix in the output CSV.
# Three Chinese positive prompts and three Chinese negative prompts are encoded by the
# Taiyi CLIP text encoder for each concept, and the final concept score is
# mean_cos(image, positive) - mean_cos(image, negative).
CONCEPT_PROMPTS = {
    # ME1 Perceived Premiumness (premium vs. cheap)
    "ME1_高级_廉价": {
        "positive": [
            # "A photo of a premium, refined motorized standing desk"
            "一张高级精致的电动升降桌产品图",
            # "A photo of a high-end, high-texture computer desk"
            "一张看起来高档有质感的电脑桌产品图",
            # "A photo of an office desk with a premium feel and a sense of quality"
            "一张具有高级感和品质感的办公桌产品图",
        ],
        "negative": [
            # "A photo of a cheap, low-end motorized standing desk"
            "一张看起来廉价低端的电动升降桌产品图",
            # "A photo of a cheap computer desk that lacks any sense of quality"
            "一张缺乏质感的便宜电脑桌产品图",
            # "A photo of an office desk that gives a cheap impression"
            "一张有廉价感的办公桌产品图",
        ],
    },
    # ME2 Visual Solidity (solid/substantial vs. flimsy)
    "ME2_厚重_轻薄": {
        "positive": [
            # "A photo of a heavy, sturdy motorized standing desk with substantial materials"
            "一张厚重稳固用料扎实的电动升降桌产品图",
            # "A photo of a computer desk with a solid structure and a substantial visual weight"
            "一张结构结实分量感强的电脑桌产品图",
            # "A photo of an office desk whose legs and top look thick and firm"
            "一张桌腿和桌面看起来厚实牢固的办公桌产品图",
        ],
        "negative": [
            # "A photo of a thin, flimsy motorized standing desk with a fragile structure"
            "一张轻薄单薄结构不结实的电动升降桌产品图",
            # "A photo of a computer desk that looks lightweight and flimsy"
            "一张看起来轻飘单薄的电脑桌产品图",
            # "A photo of an office desk whose legs and top appear weak"
            "一张桌腿和桌面显得薄弱的办公桌产品图",
        ],
    },
    # ME3 Minimalist Simplicity (minimal/clean vs. cluttered)
    "ME3_简约大气_复杂凌乱": {
        "positive": [
            # "A photo of a minimalist, clean and tidy motorized standing desk"
            "一张简约大气干净利落的电动升降桌产品图",
            # "A photo of a computer desk with a clean and uncluttered design"
            "一张设计简洁清爽的电脑桌产品图",
            # "A photo of an office desk with a clean look and no unnecessary decoration"
            "一张外观大方没有多余装饰的办公桌产品图",
        ],
        "negative": [
            # "A photo of a busy, cluttered, gaudy motorized standing desk"
            "一张复杂凌乱花哨的电动升降桌产品图",
            # "A photo of a computer desk with excessive decoration and a visually messy look"
            "一张装饰繁琐视觉杂乱的电脑桌产品图",
            # "A photo of an office desk that looks overloaded and not clean"
            "一张外观累赘不清爽的办公桌产品图",
        ],
    },
    # ME4 Aesthetic Refinement (refined/beautiful vs. rough/ugly)
    "ME4_精致美观_粗糙难看": {
        "positive": [
            # "A photo of a finely crafted, good-looking motorized standing desk"
            "一张做工精致外观好看的电动升降桌产品图",
            # "A photo of a computer desk with refined details and high visual appeal"
            "一张细节精细颜值高的电脑桌产品图",
            # "A photo of an office desk with an attractive style and well-coordinated colors"
            "一张款式美观颜色协调的办公桌产品图",
        ],
        "negative": [
            # "A photo of a poorly crafted, ugly motorized standing desk"
            "一张做工粗糙外观难看的电动升降桌产品图",
            # "A photo of a computer desk with rough details and a cheap feel"
            "一张细节粗糙有廉价感的电脑桌产品图",
            # "A photo of an office desk with mismatched colors and an unattractive style"
            "一张颜色不协调款式难看的办公桌产品图",
        ],
    },
    # ME5 Tech-Modernity (tech-modern vs. old-fashioned)
    "ME5_时尚科技_传统老气": {
        "positive": [
            # "A photo of a stylish, modern, tech-feeling motorized standing desk"
            "一张时尚现代科技感强的电动升降桌产品图",
            # "A photo of a cool computer desk with a futuristic feel"
            "一张酷炫有未来感的电脑桌产品图",
            # "A photo of an office desk with an esports / tech style"
            "一张具有电竞风和科技感的办公桌产品图",
        ],
        "negative": [
            # "A photo of a traditional, old-fashioned, outdated motorized standing desk"
            "一张传统老气过时的电动升降桌产品图",
            # "A photo of a computer desk with a dated styling and no tech feel"
            "一张造型老套缺乏科技感的电脑桌产品图",
            # "A photo of an office desk with a conservative, unfashionable style"
            "一张风格保守不时尚的办公桌产品图",
        ],
    },
}


def normalize_product_id(value: object) -> str | None:
    if pd.isna(value):
        return None

    text = str(value).strip()
    text = re.sub(r"\((\d+)\)", r"\1", text)
    text = re.sub(r"-+", "-", text)

    if re.fullmatch(r"\d+-\d+", text):
        return text

    return None


def load_product_order_from_reviews() -> list[str]:
    df = pd.read_excel(REVIEWS_XLSX, dtype=str)

    if NO_COL not in df.columns:
        raise ValueError(f"评论文件缺少列：{NO_COL}")

    product_ids = df[NO_COL].apply(normalize_product_id)

    if product_ids.isna().any():
        bad = df.loc[product_ids.isna(), NO_COL].head(10).tolist()
        raise ValueError(f"no列中存在无法识别的产品编号，示例：{bad}")

    ordered_ids = []
    seen = set()

    for pid in product_ids:
        if pid not in seen:
            ordered_ids.append(pid)
            seen.add(pid)

    return ordered_ids


def load_image_clip_features(product_order: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    clip_df = pd.read_parquet(IMAGE_CLIP_PARQUET)

    if PRODUCT_ID_COL not in clip_df.columns:
        raise ValueError(f"CLIP编码文件缺少列：{PRODUCT_ID_COL}")

    clip_df = clip_df.copy()
    clip_df[PRODUCT_ID_COL] = clip_df[PRODUCT_ID_COL].apply(normalize_product_id)

    if clip_df[PRODUCT_ID_COL].duplicated().any():
        dup = clip_df.loc[clip_df[PRODUCT_ID_COL].duplicated(), PRODUCT_ID_COL].tolist()
        raise ValueError(f"CLIP编码文件 product_id 重复：{dup[:20]}")

    missing = [pid for pid in product_order if pid not in set(clip_df[PRODUCT_ID_COL])]
    if missing:
        raise ValueError(f"以下产品在图片CLIP编码中不存在：{missing}")

    clip_df = (
        clip_df.set_index(PRODUCT_ID_COL)
        .loc[product_order]
        .reset_index()
    )

    dim_cols = [c for c in clip_df.columns if c.startswith("clip_dim_")]

    if len(dim_cols) != EXPECTED_DIM:
        raise ValueError(f"CLIP维度异常：{len(dim_cols)}，预期 {EXPECTED_DIM}")

    image_features = clip_df[dim_cols].to_numpy(dtype=np.float32)

    norms = np.linalg.norm(image_features, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("存在零向量图片特征")

    image_features = image_features / norms

    return clip_df[[PRODUCT_ID_COL]], image_features


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_taiyi_text_encoder():
    device = get_device()

    tokenizer = BertTokenizer.from_pretrained(
        TAIYI_TEXT_MODEL_DIR,
        local_files_only=True,
    )

    model = BertForSequenceClassification.from_pretrained(
        TAIYI_TEXT_MODEL_DIR,
        local_files_only=True,
    )

    model.eval()
    model.to(device)

    return tokenizer, model, device


@torch.no_grad()
def encode_texts(
    texts: list[str],
    tokenizer: BertTokenizer,
    model: BertForSequenceClassification,
    device: torch.device,
) -> np.ndarray:
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64,
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs)

    text_features = outputs.logits

    if text_features.shape[1] != EXPECTED_DIM:
        raise ValueError(
            f"Taiyi文本向量维度异常：{text_features.shape[1]}，预期 {EXPECTED_DIM}"
        )

    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features.detach().cpu().numpy().astype(np.float32)


def build_prompt_table() -> pd.DataFrame:
    rows = []

    for concept_id, prompt_group in CONCEPT_PROMPTS.items():
        for polarity, prompts in prompt_group.items():
            for i, prompt in enumerate(prompts, start=1):
                rows.append({
                    "concept_id": concept_id,
                    "prompt_polarity": polarity,
                    "prompt_index": i,
                    "prompt_text": prompt,
                })

    return pd.DataFrame(rows)


def compute_concept_scores(
    image_features: np.ndarray,
    tokenizer: BertTokenizer,
    model: BertForSequenceClassification,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prompt_df = build_prompt_table()

    score_data = {}

    for concept_id, prompt_group in CONCEPT_PROMPTS.items():
        pos_prompts = prompt_group["positive"]
        neg_prompts = prompt_group["negative"]

        pos_vecs = encode_texts(pos_prompts, tokenizer, model, device)
        neg_vecs = encode_texts(neg_prompts, tokenizer, model, device)

        pos_centroid = pos_vecs.mean(axis=0)
        neg_centroid = neg_vecs.mean(axis=0)

        pos_centroid = pos_centroid / np.linalg.norm(pos_centroid)
        neg_centroid = neg_centroid / np.linalg.norm(neg_centroid)

        sim_pos = image_features @ pos_centroid
        sim_neg = image_features @ neg_centroid

        score_data[f"concept_{concept_id}"] = sim_pos - sim_neg
        score_data[f"sim_pos_{concept_id}"] = sim_pos
        score_data[f"sim_neg_{concept_id}"] = sim_neg

    return pd.DataFrame(score_data), prompt_df


def save_metadata(product_order: list[str], device: torch.device) -> None:
    metadata = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "order_source": str(REVIEWS_XLSX),
        "order_rule": "按评论文件 no 列第一次出现顺序输出",
        "image_clip_source": str(IMAGE_CLIP_PARQUET),
        "taiyi_text_model_dir": str(TAIYI_TEXT_MODEL_DIR),
        "product_count": len(product_order),
        "embedding_dim": EXPECTED_DIM,
        "score_formula": "concept_score = mean_cos(image, positive_prompts) - mean_cos(image, negative_prompts)",
        "device": str(device),
        "outputs": {
            "visual_concept_scores_parquet": str(OUTPUT_PARQUET),
            "visual_concept_scores_csv": str(OUTPUT_CSV),
            "visual_concept_prompts_csv": str(PROMPT_CSV),
            "metadata_json": str(METADATA_JSON),
        },
    }

    with open(METADATA_JSON, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    product_order = load_product_order_from_reviews()
    product_df, image_features = load_image_clip_features(product_order)

    tokenizer, model, device = load_taiyi_text_encoder()

    score_df, prompt_df = compute_concept_scores(
        image_features=image_features,
        tokenizer=tokenizer,
        model=model,
        device=device,
    )

    output_df = pd.concat([product_df, score_df], axis=1)

    # Re-check ordering to guard against any mid-pipeline misalignment
    if output_df[PRODUCT_ID_COL].tolist() != product_order:
        raise ValueError("输出 product_id 顺序与评论 no 顺序不一致，已停止保存。")

    output_df.to_parquet(OUTPUT_PARQUET, index=False)
    output_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    prompt_df.to_csv(PROMPT_CSV, index=False, encoding="utf-8-sig")
    save_metadata(product_order, device)

    print("视觉感性概念分数计算完成")
    print(f"产品数：{len(output_df)}")
    print(f"顺序来源：{REVIEWS_XLSX} 的 no 列第一次出现顺序")
    print(f"输出 parquet：{OUTPUT_PARQUET}")
    print(f"输出 csv：{OUTPUT_CSV}")
    print(f"prompt记录：{PROMPT_CSV}")
    print(f"元数据：{METADATA_JSON}")

    print("\n前10个产品顺序：")
    print(output_df[PRODUCT_ID_COL].head(10).tolist())


if __name__ == "__main__":
    main()
