"""
llm_sentiment_scoring.py
=============================
Score each review with Qwen-plus on a continuous sentiment scale (-1.0 to 1.0).

Notes:
- Force line-buffered stdout (so prints are visible in IDEs and some terminals)
- OpenAI client uses httpx.Timeout(connect=5s, read=20s)
- Disable the SDK's internal silent retry (max_retries=0)
- All prints use flush=True
- The connectivity test prints timestamps and tracebacks

Features:
- DashScope OpenAI-compatible endpoint
- Multi-threaded concurrency (default 20)
- Resumable run (flush checkpoint every 500 records)
- API key entered at runtime (never written to code)
- Token-efficient prompt
- Order preserved (anchored by the original row index)
- Automatic retry up to 3 times on failure (exponential backoff)

Input: product_reviews_clean.xlsx (41,951 reviews x 200 products)
Outputs:
  data/satisfaction/comment_sentiment_scores.csv
  data/satisfaction/checkpoint_sentiment.csv
  data/satisfaction/run_metadata.json
  data/satisfaction/prompt_sentiment.txt
"""

import os
import re
import sys
import time
import json
import getpass
import traceback
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# === Force line-buffered stdout (Python 3.7+) ===
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass


def p(*args, **kwargs):
    """Unified print helper that always flushes."""
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
OUTPUT_DIR = PROJECT_ROOT / "data" / "satisfaction"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_CSV = OUTPUT_DIR / "checkpoint_sentiment.csv"
FINAL_CSV = OUTPUT_DIR / "comment_sentiment_scores.csv"
METADATA_JSON = OUTPUT_DIR / "run_metadata.json"
PROMPT_FILE = OUTPUT_DIR / "prompt_sentiment_v1.1.txt"

MODEL_NAME = "qwen-plus"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

CONCURRENCY = 5
MAX_RETRIES = 3
CHECKPOINT_EVERY = 500

# httpx timeout configuration
HTTP_TIMEOUT = httpx.Timeout(
    timeout=25.0,    # overall timeout
    connect=5.0,     # connect timeout (DNS + TCP)
    read=20.0,       # response read timeout
    write=10.0,
    pool=5.0,
)

PROMPT_TEMPLATE = (
    "评估评论情感倾向,输出-1到1的小数(-1极负面,0中性,1极正面,"
    "可用0.3/-0.7等中间值精确反映强度)。仅输出数字,不要任何其他内容。\n"
    "评论:{text}"
)
'''
    "Evaluate the sentiment of the comment and output a decimal value between -1 and 1 (-1 is extremely negative, 0 is neutral, and 1 is extremely positive). 
    You may use intermediate values such as 0.3 or -0.7 to accurately reflect the intensity. Output only the number; do not include any other content. \n"
    "Comment: {text}"
'''


# ============================================================
# Helpers
# ============================================================
def parse_sentiment(raw):
    if raw is None:
        return np.nan
    raw = str(raw).strip()
    if not raw:
        return np.nan
    m = re.search(r"-?\d+\.?\d*", raw)
    if m is None:
        return np.nan
    try:
        val = float(m.group(0))
    except ValueError:
        return np.nan
    if np.isnan(val):
        return np.nan
    return float(max(-1.0, min(1.0, val)))


def score_one(client: OpenAI, row_id: int, text):
    if not isinstance(text, str) or not text.strip():
        return row_id, np.nan, "empty_text"

    prompt = PROMPT_TEMPLATE.format(text=text)
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10,
            )
            raw_out = resp.choices[0].message.content
            val = parse_sentiment(raw_out)
            if not np.isnan(val):
                return row_id, val, None
            last_err = f"parse_fail: {raw_out!r}"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            if attempt < MAX_RETRIES - 1:
                time.sleep(min(2 ** attempt, 6))
    return row_id, np.nan, last_err


def flush_checkpoint(buffer, path):
    if not buffer:
        return
    new_df = pd.DataFrame(buffer, columns=["_row_id", "sentiment_score", "error"])
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
    p("  LLM 情感打分 — Qwen-plus  ")
    p("=" * 64)

    # ---------- 1. Read API key ----------
    p("\n[准备-1/3] 输入 DashScope API Key (输入不显示,粘贴后回车)")

    api_key = input("API Key: ").strip()

    if not api_key:
        p("ERROR: API key 不能为空")
        sys.exit(1)
    p(f"[准备-1/3] OK  收到 API key (长度 {len(api_key)} 字符, 前4位 = {api_key[:4]}...)")

    # ---------- 2. Create client ----------
    p("\n[准备-2/3] 创建 OpenAI client...")
    client = OpenAI(
        api_key=api_key,
        base_url=BASE_URL,
        timeout=HTTP_TIMEOUT,
        max_retries=0,
    )
    p("[准备-2/3] OK")

    # ---------- 3. Connectivity test ----------
    p("\n[准备-3/3] 测试 API 连接...")
    p(f"  [发送] {time.strftime('%H:%M:%S')}")
    t0 = time.time()
    try:
        p("  正在请求 DashScope，如果这里超过60秒无响应，基本是网络/API接口问题...")
        rid, test_val, test_err = score_one(client, -1, "这个桌子非常好用,很满意")
        p("  DashScope 请求已返回")

        elapsed = time.time() - t0
        p(f"  [响应] {time.strftime('%H:%M:%S')}  耗时 {elapsed:.2f}s")
        if np.isnan(test_val):
            p(f"  FAIL  测试调用失败: {test_err}")
            p("  常见原因: API key 错误 / 服务未开通 / 网络不通")
            sys.exit(1)
        p(f"  OK    测试评论得分 = {test_val}")
    except Exception as e:
        elapsed = time.time() - t0
        p(f"  FAIL  耗时 {elapsed:.2f}s, 异常: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)

    # ---------- 4. Load data ----------
    p(f"\n[1/4] 加载 {INPUT_XLSX.name}")
    if not INPUT_XLSX.exists():
        p(f"ERROR: 找不到 {INPUT_XLSX}")
        sys.exit(1)
    df = pd.read_excel(INPUT_XLSX, dtype=str)
    df = df.reset_index(drop=True)
    df["_row_id"] = df.index
    df["_product_id"] = df["excel文件名"].str.extract(r"^(\d+-\d+)")
    p(f"  总行数: {len(df):,}")
    p(f"  唯一产品: {df['_product_id'].nunique()}")
    p(f"  非空 commentData: {df['commentData'].notna().sum():,}")

    # ---------- 5. Checkpoint detection ----------
    p(f"\n[2/4] 检查断点 {CHECKPOINT_CSV.name}")
    if CHECKPOINT_CSV.exists():
        done_df = pd.read_csv(CHECKPOINT_CSV)
        done_df = done_df.drop_duplicates(subset="_row_id", keep="last")
        valid_done = done_df[done_df["sentiment_score"].notna()]
        done_ids = set(valid_done["_row_id"].astype(int).tolist())
        n_fail_in_ckpt = len(done_df) - len(valid_done)
        p(f"  已完成(有效): {len(done_ids):,} 条")
        if n_fail_in_ckpt > 0:
            p(f"  之前失败(将重试): {n_fail_in_ckpt:,} 条")
        valid_done.to_csv(CHECKPOINT_CSV, index=False, encoding="utf-8-sig")
    else:
        done_ids = set()
        p("  无断点,全新运行")

    todo_df = df[~df["_row_id"].isin(done_ids)].copy()
    p(f"  待处理: {len(todo_df):,} 条")

    # ---------- 6. Concurrent calls ----------
    if len(todo_df) > 0:
        p(f"\n[3/4] 调用 {MODEL_NAME} (并发={CONCURRENCY}, 重试={MAX_RETRIES})")
        results_buffer = []
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
            tasks = list(todo_df[["_row_id", "commentData"]].itertuples(index=False, name=None))

            futures = [
                executor.submit(score_one, client, int(row_id), comment_text)
                for row_id, comment_text in tasks
            ]

            with tqdm(
                    total=len(futures),
                    desc="打分",
                    unit="条",
                    dynamic_ncols=True,
                    file=sys.stdout
            ) as pbar:
                for future in as_completed(futures):
                    rid, score, err = future.result()
                    results_buffer.append((rid, score, err))
                    pbar.update(1)

                    if len(results_buffer) >= CHECKPOINT_EVERY:
                        flush_checkpoint(results_buffer, CHECKPOINT_CSV)

        flush_checkpoint(results_buffer, CHECKPOINT_CSV)
        elapsed = time.time() - start_time
        p(f"  耗时: {elapsed/60:.1f} 分钟 ({len(todo_df)/elapsed:.1f} 条/秒)")
    else:
        p("\n[3/4] 全部已完成,跳过打分阶段")

    # ---------- 7. Assemble final output ----------
    p(f"\n[4/4] 整理输出 → {FINAL_CSV.name}")
    final_df = pd.read_csv(CHECKPOINT_CSV)
    final_df = final_df.drop_duplicates(subset="_row_id", keep="last")

    out = df[["_row_id", "_product_id", "commentId", "commentData"]].merge(
        final_df[["_row_id", "sentiment_score"]], on="_row_id", how="left"
    )
    out = out.sort_values("_row_id").reset_index(drop=True)
    out = out.rename(columns={"_row_id": "row_id", "_product_id": "product_id"})

    out[["row_id", "product_id", "commentId", "sentiment_score"]].to_csv(
        FINAL_CSV, index=False, encoding="utf-8-sig"
    )

    # ---------- 8. Metadata ----------
    n_total = len(out)
    n_valid = int(out["sentiment_score"].notna().sum())
    n_fail = n_total - n_valid

    score_dist = out["sentiment_score"].dropna()
    metadata = {
        "model": MODEL_NAME,
        "base_url": BASE_URL,
        "concurrency": CONCURRENCY,
        "max_retries": MAX_RETRIES,
        "input_file": str(INPUT_XLSX),
        "output_file": str(FINAL_CSV),
        "n_total": n_total,
        "n_valid": n_valid,
        "n_failed": n_fail,
        "success_rate": round(n_valid / n_total, 4) if n_total else 0,
        "score_distribution": {
            "mean": float(score_dist.mean()) if len(score_dist) else None,
            "std": float(score_dist.std()) if len(score_dist) else None,
            "min": float(score_dist.min()) if len(score_dist) else None,
            "max": float(score_dist.max()) if len(score_dist) else None,
            "median": float(score_dist.median()) if len(score_dist) else None,
            "pct_positive": float((score_dist > 0.1).mean()) if len(score_dist) else None,
            "pct_neutral": float(((score_dist >= -0.1) & (score_dist <= 0.1)).mean()) if len(score_dist) else None,
            "pct_negative": float((score_dist < -0.1).mean()) if len(score_dist) else None,
        },
        "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "prompt_template": PROMPT_TEMPLATE,
    }
    METADATA_JSON.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    PROMPT_FILE.write_text(PROMPT_TEMPLATE, encoding="utf-8")

    # ---------- 9. Wrap up ----------
    p("\n" + "=" * 64)
    p("  完成")
    p("=" * 64)
    p(f"成功: {n_valid:,} / {n_total:,}  ({n_valid/n_total*100:.2f}%)")
    if n_fail > 0:
        p(f"失败: {n_fail:,}  (失败行 sentiment_score=NaN, 重跑本脚本自动重试)")
    if len(score_dist):
        p(f"分数分布: mean={score_dist.mean():+.3f}  std={score_dist.std():.3f}")
        p(f"          正面 {(score_dist > 0.1).mean()*100:.1f}% | "
          f"中性 {((score_dist >= -0.1) & (score_dist <= 0.1)).mean()*100:.1f}% | "
          f"负面 {(score_dist < -0.1).mean()*100:.1f}%")
    p(f"\n输出: {FINAL_CSV}")
    p(f"元数据: {METADATA_JSON}")


if __name__ == "__main__":
    main()
