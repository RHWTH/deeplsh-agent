import argparse
import math
import os
import sys
from collections import Counter

import pandas as pd


def _compute_idf(corpus):
    n_docs = len(corpus)
    df = Counter()
    for doc in corpus:
        df.update(set(doc))
    return {term: math.log((n_docs + 1) / (df_t + 1)) + 1.0 for term, df_t in df.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-repo", default=None)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--index-a", type=int, default=0)
    parser.add_argument("--index-b", type=int, default=1)
    parser.add_argument(
        "--measure",
        choices=["jaccard", "cosine", "levenshtein", "pdm", "tracesim"],
        default="tracesim",
    )
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.0)
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    python_packages = os.path.join(project_root, "code", "python-packages")
    if python_packages not in sys.path:
        sys.path.insert(0, python_packages)

    from similarities import (
        cosine_similarity,
        jaccard,
        levenshtein_df,
        pdm,
        traceSim,
    )

    if args.data_repo is None:
        data_repo = os.path.join(project_root, "data")
    else:
        data_repo = args.data_repo

    stacks_path = os.path.join(data_repo, "frequent_stack_traces.csv")
    df = pd.read_csv(stacks_path, index_col=0)
    if args.n is not None:
        df = df.head(args.n)

    if "stackTraceCusto" not in df.columns:
        raise ValueError("Expected column 'stackTraceCusto' in frequent_stack_traces.csv")

    df["listStackTrace"] = df["stackTraceCusto"].fillna("").apply(lambda x: str(x).split("\n"))

    a = int(args.index_a)
    b = int(args.index_b)
    if a < 0 or b < 0 or a >= len(df) or b >= len(df):
        raise ValueError(f"index out of range: a={a}, b={b}, n={len(df)}")

    stack_a = df["listStackTrace"].iloc[a]
    stack_b = df["listStackTrace"].iloc[b]

    if args.measure == "jaccard":
        score = jaccard(stack_a, stack_b)
    elif args.measure == "cosine":
        frames = sorted(set(stack_a) | set(stack_b))
        v1 = [stack_a.count(f) for f in frames]
        v2 = [stack_b.count(f) for f in frames]
        score = cosine_similarity(v1, v2)
    elif args.measure == "levenshtein":
        s = pd.Series([stack_b])
        score = float(levenshtein_df(stack_a, s, index=0, distinct=False)[0])
    elif args.measure == "pdm":
        score = float(pdm(stack_a, stack_b))
    elif args.measure == "tracesim":
        corpus = df["listStackTrace"].tolist()
        idf = _compute_idf(corpus)
        score = float(traceSim(stack_a, stack_b, idf, args.alpha, args.beta, args.gamma))
    else:
        raise ValueError(f"Unknown measure: {args.measure}")

    print(
        f"measure={args.measure} n={len(df)} index_a={a} index_b={b} score={score:.6f}"
    )


if __name__ == "__main__":
    main()
