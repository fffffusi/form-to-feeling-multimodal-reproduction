"""
llm_extract_function.py
===========================
Full-corpus functional extraction with Qwen-plus.

Dimensions extracted (matches Appendix A.4 of the paper):
  MF1 Lifting Performance
  MF2 Noise Control
  MF3 Load-Bearing Stability
  MF4 Material and Build Quality
  MF5 Odor and Environmental Safety
  MF6 Control and Interface Integration
  MF7 Workspace Size Adequacy

Input:  product_reviews_clean.xlsx (41,951 reviews x 200 products)
Prompt: lexicons/function_extraction.txt  (used as system message; enables prompt cache)
Model:  qwen-plus
Concurrency: 5

Outputs:
- data/extraction/checkpoint_function.csv          (one row per review, raw_json)
- data/extraction/llm_function_extractions.parquet (long format, one row per match)
- data/extraction/llm_function_extractions.csv     (long format CSV backup)
- data/extraction/llm_function_metadata.json
"""

import os
import re
import sys
import time
import json
import traceback
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass


def p(*args, **kwargs):
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


import numpy as np
import pandas as pd
from tqdm import tqdm
import httpx
from openai import OpenAI


# ============================================================
# Configuration
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_XLSX = PROJECT_ROOT / "data" / "input" / "product_reviews_clean.xlsx"
PROMPT_FILE = PROJECT_ROOT / "lexicons" / "function_extraction.txt"

OUTPUT_DIR = PROJECT_ROOT / "data" / "extraction"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_CSV = OUTPUT_DIR / "checkpoint_function.csv"
FINAL_PARQUET = OUTPUT_DIR / "llm_function_extractions.parquet"
FINAL_CSV = OUTPUT_DIR / "llm_function_extractions.csv"
METADATA_JSON = OUTPUT_DIR / "llm_function_metadata.json"

MODEL_NAME = "qwen-plus"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

CONCURRENCY = 5
MAX_RETRIES = 3
CHECKPOINT_EVERY = 300
MAX_OUTPUT_TOKENS = 600


HTTP_TIMEOUT = httpx.Timeout(timeout=90.0, connect=10.0, read=80.0, write=10.0, pool=5.0)

# Whitelisted functional dimension labels (Chinese tokens are the keys used by the prompt).
# Mapping to the paper notation:
#   升降功能      -> MF1 Lifting Performance
#   噪音控制      -> MF2 Noise Control
#   稳定性承重    -> MF3 Load-Bearing Stability
#   材质做工      -> MF4 Material and Build Quality
#   异味环保      -> MF5 Odor and Environmental Safety
#   操控智能      -> MF6 Control and Interface Integration
#   空间尺寸      -> MF7 Workspace Size Adequacy
FUNCTION_DIMENSIONS = {
    "升降功能",
    "噪音控制",
    "稳定性承重",
    "材质做工",
    "异味环保",
    "操控智能",
    "空间尺寸",
}


# ============================================================
# Helpers
# ============================================================
def load_system_prompt():
    text = PROMPT_FILE.read_text(encoding="utf-8")
    marker = "## 待处理评论"
    if marker not in text:
        raise ValueError(f"prompt 中找不到分隔符 '{marker}'")
    return text.split(marker, 1)[0].strip()


def get_cached_tokens(usage):
    if not usage:
        return 0
    details = usage.get("prompt_tokens_details", {})
    if isinstance(details, dict):
        v = details.get("cached_tokens", 0)
        return int(v) if v is not None else 0
    for key in ["cached_tokens", "prompt_cache_hit_tokens", "input_cache_hit_tokens"]:
        v = usage.get(key)
        if v is not None:
            return int(v)
    return 0


def parse_matches_json(raw):
    if not raw:
        return None, "empty"
    raw_strip = raw.strip()
    raw_strip = re.sub(r"^```(?:json)?\s*", "", raw_strip)
    raw_strip = re.sub(r"\s*```$", "", raw_strip)
    try:
        data = json.loads(raw_strip)
    except Exception:
        m = re.search(r"\{.*\}", raw_strip, re.DOTALL)
        if not m:
            return None, "no_json_block"
        try:
            data = json.loads(m.group(0))
        except Exception as e:
            return None, f"json_parse_fail: {e}"
    if not isinstance(data, dict) or "matches" not in data:
        return None, "no_matches_key"
    if not isinstance(data["matches"], list):
        return None, "matches_not_list"
    return data["matches"], None


def validate_matches(matches, comment_text):
    """Filter out hallucinated spans, non-whitelisted dimensions, and abnormal polarity (functional allows 0)."""
    if not matches:
        return [], 0
    valid = []
    n_inv = 0
    for m in matches:
        if not isinstance(m, dict):
            n_inv += 1
            continue
        span = m.get("span", "")
        dim = m.get("dimension", "")
        pol = m.get("polarity", None)
        if not isinstance(span, str) or not span:
            n_inv += 1
            continue
        if span not in comment_text:
            n_inv += 1
            continue
        if dim not in FUNCTION_DIMENSIONS:
            n_inv += 1
            continue
        if pol not in (1, -1, 0):
            n_inv += 1
            continue
        valid.append({
            "dimension": dim,
            "span": span,
            "polarity": int(pol),
            "degree_word": str(m.get("degree_word", "") or ""),
            "negated": bool(m.get("negated", False)),
        })
    return valid, n_inv


def call_api(client, system_prompt, user_text):
    user_msg = f'## 待处理评论\n\n评论: "{user_text}"\n\n输出:'
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=MAX_OUTPUT_TOKENS,
        )
        latency = time.time() - t0
        raw = resp.choices[0].message.content
        usage = {}
        try:
            usage = resp.usage.model_dump()
        except Exception:
            try:
                usage = dict(resp.usage)
            except Exception:
                pass
        return raw, usage, latency, None
    except Exception as e:
        return None, {}, time.time() - t0, f"{type(e).__name__}: {e}"


def process_one(client, system_prompt, row_id, text):
    if not isinstance(text, str) or not text.strip():
        return {
            "_row_id": row_id, "n_matches": 0, "n_invalid_spans": 0,
            "raw_matches_json": "[]",
            "in_tokens": 0, "cached_tokens": 0, "out_tokens": 0,
            "error": "empty_text",
        }

    last_err = "unknown"
    last_usage = {}
    for attempt in range(MAX_RETRIES):
        raw, usage, latency, err = call_api(client, system_prompt, text)
        if err:
            last_err = err
            if attempt < MAX_RETRIES - 1:
                time.sleep(min(2 ** attempt, 8))
            continue
        last_usage = usage
        matches, parse_err = parse_matches_json(raw)
        if parse_err:
            last_err = parse_err
            if attempt < MAX_RETRIES - 1:
                time.sleep(min(2 ** attempt, 3))
            continue
        valid, n_inv = validate_matches(matches, text)
        return {
            "_row_id": row_id,
            "n_matches": len(valid),
            "n_invalid_spans": n_inv,
            "raw_matches_json": json.dumps(valid, ensure_ascii=False),
            "in_tokens": int(usage.get("prompt_tokens") or 0),
            "cached_tokens": get_cached_tokens(usage),
            "out_tokens": int(usage.get("completion_tokens") or 0),
            "error": "",
        }

    return {
        "_row_id": row_id, "n_matches": 0, "n_invalid_spans": 0,
        "raw_matches_json": "",
        "in_tokens": int(last_usage.get("prompt_tokens") or 0),
        "cached_tokens": get_cached_tokens(last_usage),
        "out_tokens": int(last_usage.get("completion_tokens") or 0),
        "error": last_err,
    }


def flush_checkpoint(buffer, path):
    if not buffer:
        return
    new_df = pd.DataFrame(buffer)
    if path.exists():
        new_df.to_csv(path, mode="a", header=False, index=False, encoding="utf-8-sig")
    else:
        new_df.to_csv(path, mode="w", header=True, index=False, encoding="utf-8-sig")
    buffer.clear()


# ============================================================
# Main pipeline
# ============================================================
def main():
    p("=" * 64)
    p(f"  全量功能抽取 — {MODEL_NAME} ")
    p("=" * 64)

    p("\n[准备-1/3] 输入 DashScope API Key")
    api_key = input("API Key: ").strip()
    if not api_key:
        p("ERROR: API key 不能为空")
        sys.exit(1)
    p(f"  OK 长度 {len(api_key)}, 前4位 {api_key[:4]}...")

    p(f"\n[准备-2/3] 加载 {PROMPT_FILE.name}")
    if not PROMPT_FILE.exists():
        p(f"ERROR: 找不到 {PROMPT_FILE}")
        sys.exit(1)
    sys_prompt = load_system_prompt()
    p(f"  system prompt: {len(sys_prompt)} 字符 (~{len(sys_prompt)/1.5:.0f} tokens)")

    p("\n[准备-3/3] 测试 API 连接")
    client = OpenAI(api_key=api_key, base_url=BASE_URL, timeout=HTTP_TIMEOUT, max_retries=0)
    p(f"  [发送] {time.strftime('%H:%M:%S')}")
    t0 = time.time()
    try:
        raw, usage, lat, err = call_api(client, sys_prompt, "升降很丝滑,质量也很好")
        p(f"  [响应] {time.strftime('%H:%M:%S')}  耗时 {lat:.2f}s")
        if err:
            p(f"  FAIL: {err}")
            sys.exit(1)
        ms, perr = parse_matches_json(raw)
        if perr:
            p(f"  解析告警: {perr}, raw={raw!r}")
        else:
            p(f"  OK  测试输出 matches: {len(ms)} 个")
            p(f"  in_tokens={usage.get('prompt_tokens')} out_tokens={usage.get('completion_tokens')}")
    except Exception as e:
        p(f"  FAIL: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)

    p(f"\n[1/4] 加载 {INPUT_XLSX.name}")
    df = pd.read_excel(INPUT_XLSX, dtype=str)
    df = df.reset_index(drop=True)
    df["_row_id"] = df.index
    df["_product_id"] = df["excel文件名"].str.extract(r"^(\d+-\d+)")
    p(f"  总行数: {len(df):,}")
    p(f"  唯一产品: {df['_product_id'].nunique()}")

    p(f"\n[2/4] 检查断点 {CHECKPOINT_CSV.name}")
    if CHECKPOINT_CSV.exists():
        done_df = pd.read_csv(CHECKPOINT_CSV)
        done_df = done_df.drop_duplicates(subset="_row_id", keep="last")
        ok_mask = done_df["error"].isna() | (done_df["error"].astype(str) == "")
        valid_done = done_df[ok_mask]
        done_ids = set(valid_done["_row_id"].astype(int).tolist())
        n_fail = len(done_df) - len(valid_done)
        p(f"  已完成(有效): {len(done_ids):,} 条")
        if n_fail > 0:
            p(f"  之前失败(将重试): {n_fail:,} 条")
        valid_done.to_csv(CHECKPOINT_CSV, index=False, encoding="utf-8-sig")
    else:
        done_ids = set()
        p("  无断点,全新运行")

    todo_df = df[~df["_row_id"].isin(done_ids)].copy()
    p(f"  待处理: {len(todo_df):,} 条")

    if len(todo_df) > 0:
        p(f"\n[3/4] 调用 {MODEL_NAME} (并发={CONCURRENCY}, 重试={MAX_RETRIES})")
        p(f"  提示:Max 模型较慢,预计 6-8 小时;中断后重跑会自动续")
        buffer = []
        start = time.time()

        with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
            futures = [
                executor.submit(
                    process_one,
                    client,
                    sys_prompt,
                    int(row_id),
                    comment_text
                )
                for row_id, comment_text in zip(todo_df["_row_id"], todo_df["commentData"])
            ]

            with tqdm(total=len(futures), desc="功能抽取", unit="条",
                      dynamic_ncols=True, file=sys.stdout) as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    buffer.append(result)
                    pbar.update(1)
                    if len(buffer) >= CHECKPOINT_EVERY:
                        flush_checkpoint(buffer, CHECKPOINT_CSV)
        flush_checkpoint(buffer, CHECKPOINT_CSV)
        elapsed = time.time() - start
        p(f"  耗时: {elapsed/60:.1f} 分钟 ({len(todo_df)/elapsed:.1f} 条/秒)")
    else:
        p(f"\n[3/4] 全部已完成,跳过 API 调用")

    p(f"\n[4/4] 展开 long-format → {FINAL_PARQUET.name}")
    ck = pd.read_csv(CHECKPOINT_CSV)
    ck = ck.drop_duplicates(subset="_row_id", keep="last")
    ck["_row_id"] = ck["_row_id"].astype(int)

    keep = df[["_row_id", "excel文件名", "commentId", "commentData"]].copy()
    keep["product_id"] = keep["excel文件名"].str.extract(r"^(\d+-\d+)")
    merged = keep.merge(ck, on="_row_id", how="left")
    merged = merged.sort_values("_row_id").reset_index(drop=True)

    long_rows = []
    for _, r in merged.iterrows():
        raw = r.get("raw_matches_json", "")
        if not isinstance(raw, str) or raw == "" or raw == "[]":
            continue
        try:
            ms = json.loads(raw)
        except Exception:
            continue
        if not isinstance(ms, list):
            continue
        for midx, m in enumerate(ms):
            long_rows.append({
                "row_id": int(r["_row_id"]),
                "product_id": r["product_id"],
                "commentId": r["commentId"],
                "match_id": midx,
                "dimension": m.get("dimension"),
                "span": m.get("span"),
                "polarity": int(m.get("polarity", 0)),
                "degree_word": m.get("degree_word", "") or "",
                "negated": bool(m.get("negated", False)),
            })
    long_df = pd.DataFrame(long_rows)

    if len(long_df) > 0:
        long_df.to_parquet(FINAL_PARQUET, index=False)
        long_df.to_csv(FINAL_CSV, index=False, encoding="utf-8-sig")
        p(f"  写入 {FINAL_PARQUET.name} + .csv  ({len(long_df):,} matches)")
    else:
        p(f"  WARN: 无 matches,输出空文件")
        pd.DataFrame(columns=["row_id", "product_id", "commentId", "match_id",
                              "dimension", "span", "polarity", "degree_word", "negated"]
                     ).to_parquet(FINAL_PARQUET, index=False)

    # Metadata
    n_total = len(merged)
    n_with_match = int((merged["n_matches"].fillna(0) > 0).sum()) if "n_matches" in merged.columns else 0
    n_total_in = int(ck["in_tokens"].fillna(0).sum())
    n_total_cached = int(ck["cached_tokens"].fillna(0).sum())
    n_total_out = int(ck["out_tokens"].fillna(0).sum())

    dim_dist = long_df["dimension"].value_counts().to_dict() if len(long_df) else {}
    pol_dist = long_df["polarity"].value_counts().to_dict() if len(long_df) else {}

    meta = {
        "model": MODEL_NAME,
        "base_url": BASE_URL,
        "prompt_file": str(PROMPT_FILE),
        "input_file": str(INPUT_XLSX),
        "n_comments_total": int(n_total),
        "n_comments_with_match": n_with_match,
        "comment_match_rate": round(n_with_match / max(n_total, 1), 4),
        "n_total_matches": int(len(long_df)),
        "avg_matches_per_comment": round(len(long_df) / max(n_total, 1), 4),
        "dimension_distribution": {str(k): int(v) for k, v in dim_dist.items()},
        "polarity_distribution": {str(k): int(v) for k, v in pol_dist.items()},
        "n_total_input_tokens": n_total_in,
        "n_total_cached_tokens": n_total_cached,
        "n_total_output_tokens": n_total_out,
        "cache_hit_rate": round(n_total_cached / max(n_total_in, 1), 4),
        "concurrency": CONCURRENCY,
        "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    METADATA_JSON.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    p("\n" + "=" * 64)
    p("=" * 64)
    p(f"评论总数: {n_total:,}")
    p(f"含命中评论: {n_with_match:,} ({n_with_match/max(n_total,1)*100:.1f}%)")
    p(f"总 matches: {len(long_df):,}, 平均每评论 {len(long_df)/max(n_total,1):.2f}")
    p(f"\nToken 用量:")
    p(f"  输入: {n_total_in:,} (缓存 {n_total_cached:,}, 命中 {n_total_cached/max(n_total_in,1)*100:.1f}%)")
    p(f"  输出: {n_total_out:,}")
    p(f"\n维度分布:")
    for k, v in sorted(dim_dist.items(), key=lambda x: -x[1]):
        p(f"  {k}: {v:,}")
    p(f"\n输出文件:")
    p(f"  {FINAL_PARQUET}")
    p(f"  {METADATA_JSON}")


if __name__ == "__main__":
    main()
