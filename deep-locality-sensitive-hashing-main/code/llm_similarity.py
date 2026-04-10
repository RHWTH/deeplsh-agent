"""
LLM-enhanced stack trace similarity.

Usage:
    # baseline only (no DeepLSH model)
    python llm_similarity.py --index-a 0 --index-b 1

    # with a trained DeepLSH model
    python llm_similarity.py --index-a 0 --index-b 1 \
        --model-path Models/model-deep-lsh-TraceSim.model \
        --measure TraceSim
"""

import argparse
import json
import math
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
from openai import OpenAI

# ── DeepSeek API ──────────────────────────────────────────────────────────────
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-bb8237ce78c44f6b8eeda16a0eea0892")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

python_packages = os.path.join(os.path.dirname(os.path.abspath(__file__)), "python-packages")
if python_packages not in sys.path:
    sys.path.insert(0, python_packages)

from similarities import jaccard, pdm, traceSim, process_frame


# ── IDF helper ────────────────────────────────────────────────────────────────

def _compute_idf(corpus):
    n = len(corpus)
    df = Counter()
    for doc in corpus:
        df.update(set(doc))
    return {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}


# ── DeepLSH inference ─────────────────────────────────────────────────────────

def _build_frame_vocab(corpus):
    """Rebuild the same frame vocabulary used during training."""
    frames = pd.Series(list(set([f for stack in corpus for f in stack])))
    df_frames = pd.DataFrame()
    dummies = pd.get_dummies(frames)
    df_frames["frame"] = dummies.T.reset_index().rename(columns={"index": "frame"})["frame"]
    df_frames["embedding"] = dummies.T.reset_index().apply(lambda x: x[1:].values, axis=1)
    return df_frames


def _index_frame(stack, df_frames):
    result = []
    for f in stack:
        matches = df_frames.index[df_frames["frame"] == f]
        result.append(int(matches[0]) + 1 if len(matches) > 0 else 0)
    return result


def _hamming_diff(v1, v2, b, length):
    """Improved hamming similarity in [0, 1] (same formula as HamDist layer)."""
    count = 0
    i = 0
    while i < length:
        count += np.max(np.abs(v1[i:i + b] - v2[i:i + b])) * b
        i += b
    return float(1 - count / (length * 2))


def load_deeplsh_model(model_path: str):
    """Load a saved intermediate DeepLSH encoder model."""
    import tensorflow as tf
    return tf.keras.models.load_model(model_path, compile=False)


def compute_deeplsh_score(model, stack_a: list, stack_b: list,
                          df_frames, max_length: int, b: int = 16) -> float:
    """
    Encode two stacks with the trained DeepLSH encoder and return
    their hamming similarity score in [0, 1].
    """
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    idx_a = _index_frame(stack_a, df_frames)
    idx_b = _index_frame(stack_b, df_frames)

    X = pad_sequences([idx_a, idx_b], padding="post", truncating="post", maxlen=max_length)
    vecs = model.predict(X, verbose=0)  # shape (2, size_hash_vector)

    # binarise (sign)
    v1 = np.array([1 if x > 0 else -1 for x in vecs[0]])
    v2 = np.array([1 if x > 0 else -1 for x in vecs[1]])

    size_embedding = v1.shape[0]
    return _hamming_diff(v1, v2, b, size_embedding)


# ── Baseline scores ───────────────────────────────────────────────────────────

def get_baseline_scores(stack_a: list, stack_b: list, idf: dict,
                        deeplsh_score: float | None = None) -> dict:
    """Return a dict of similarity scores (all in [0, 1])."""
    scores = {}
    scores["jaccard"]   = round(jaccard(stack_a, stack_b), 4)
    scores["pdm"]       = round(pdm(stack_a, stack_b), 4)
    scores["tracesim"]  = round(traceSim(stack_a, stack_b, idf, alpha=0.5, beta=1.0, gamma=0.0), 4)
    if deeplsh_score is not None:
        scores["deeplsh_hamming"] = round(deeplsh_score, 4)
    return scores


# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert in analyzing Java crash reports (stack traces) from a bug tracking system.
Your task is to assess how similar two stack traces are — similar stack traces typically \
represent the same underlying bug.

Rules:
- A score of 1.0 means the two stacks are essentially the same crash.
- A score of 0.0 means they are completely unrelated.
- Focus on: shared method calls, package hierarchy, call order, and error context.
- Traditional algorithms may miss semantic nuances (e.g. refactored method names, \
  equivalent call paths). Use your judgment to adjust.
- "deeplsh_hamming" (when present) is a learned neural hash similarity from a deep \
  locality-sensitive hashing model trained on this dataset — treat it as a strong \
  structural signal but still apply semantic reasoning.
- Always respond with valid JSON only, no extra text."""


def build_prompt(stack_a: list, stack_b: list, baseline: dict) -> str:
    def fmt(stack):
        return "\n".join(f"  {i+1}. {f}" for i, f in enumerate(stack))

    baseline_lines = "\n".join(f"  - {k}: {v}" for k, v in baseline.items())

    return f"""\
Stack Trace A ({len(stack_a)} frames):
{fmt(stack_a)}

Stack Trace B ({len(stack_b)} frames):
{fmt(stack_b)}

Similarity scores (traditional algorithms + learned model):
{baseline_lines}

Analyze the two stack traces and respond with JSON in exactly this format:
{{
  "adjusted_score": <float 0.0-1.0>,
  "confidence": "<high|medium|low>",
  "reasoning": "<concise explanation>",
  "key_differences": ["<diff1>", "<diff2>"]
}}"""


# ── LLM call ──────────────────────────────────────────────────────────────────

def call_deepseek(prompt: str) -> dict:
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    return json.loads(response.choices[0].message.content)


# ── Main entry ────────────────────────────────────────────────────────────────

def llm_adjusted_similarity(stack_a: list, stack_b: list, idf: dict,
                             deeplsh_score: float | None = None) -> dict:
    baseline = get_baseline_scores(stack_a, stack_b, idf, deeplsh_score)
    prompt   = build_prompt(stack_a, stack_b, baseline)
    result   = call_deepseek(prompt)
    result["baseline"] = baseline
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-repo",   default=None)
    parser.add_argument("--n",           type=int, default=1000)
    parser.add_argument("--index-a",     type=int, default=0)
    parser.add_argument("--index-b",     type=int, default=1)
    # DeepLSH model options (optional)
    parser.add_argument("--model-path",  default=None,
                        help="Path to a saved DeepLSH intermediate encoder model "
                             "(e.g. Models/model-deep-lsh-TraceSim.model). "
                             "When provided, the model's hamming score is included in the prompt.")
    parser.add_argument("--b",           type=int, default=16,
                        help="Block size used during DeepLSH training (default: 16)")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_repo = args.data_repo or os.path.join(project_root, "data")

    df = pd.read_csv(os.path.join(data_repo, "frequent_stack_traces.csv"), index_col=0)
    if args.n:
        df = df.head(args.n)
    df["listStackTrace"] = df["stackTraceCusto"].fillna("").apply(lambda x: str(x).split("\n"))

    a, b = args.index_a, args.index_b
    if not (0 <= a < len(df) and 0 <= b < len(df)):
        raise ValueError(f"index out of range: a={a}, b={b}, n={len(df)}")

    stack_a = df["listStackTrace"].iloc[a]
    stack_b = df["listStackTrace"].iloc[b]
    corpus  = df["listStackTrace"].tolist()
    idf     = _compute_idf(corpus)

    # ── Optional: compute DeepLSH score ──────────────────────────────────────
    deeplsh_score = None
    if args.model_path:
        model_path = args.model_path
        if not os.path.isabs(model_path):
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_path)

        print(f"[deeplsh] loading model from {model_path} ...")
        model      = load_deeplsh_model(model_path)
        df_frames  = _build_frame_vocab(corpus)
        max_length = df["listStackTrace"].apply(len).max()
        deeplsh_score = compute_deeplsh_score(model, stack_a, stack_b,
                                              df_frames, max_length, b=args.b)
        print(f"[deeplsh] hamming score = {deeplsh_score:.4f}")

    result = llm_adjusted_similarity(stack_a, stack_b, idf, deeplsh_score)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
