import argparse
import os
import sys

import pandas as pd


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _ensure_python_packages_on_path() -> None:
    root = _project_root()
    python_packages = os.path.join(root, "code", "python-packages")
    if python_packages not in sys.path:
        sys.path.insert(0, python_packages)


def cmd_list(args) -> int:
    root = _project_root()
    data_repo = args.data_repo or os.path.join(root, "data")
    df_measures = pd.read_csv(os.path.join(data_repo, "similarity-measures-pairs.csv"), index_col=0)

    print("available_measures:")
    for col in df_measures.columns:
        print(col)

    print("\navailable_commands:")
    print("lite   - query a single pair similarity value from similarity-measures-pairs.csv")
    print("deeplsh - train DeepLSH for a selected measure and build LSH hash tables")
    return 0


def cmd_lite(args) -> int:
    _ensure_python_packages_on_path()
    from similarities import get_index_sim

    root = _project_root()
    data_repo = args.data_repo or os.path.join(root, "data")

    df_stacks = pd.read_csv(os.path.join(data_repo, "frequent_stack_traces.csv"), index_col=0)
    n_stacks = int(args.n_stacks or df_stacks.shape[0])

    if n_stacks != 1000:
        raise ValueError(
            "lite mode uses similarity-measures-pairs.csv which corresponds to 1000 stacks in this repo. "
            "Please use --n-stacks 1000."
        )

    df_measures = pd.read_csv(os.path.join(data_repo, "similarity-measures-pairs.csv"), index_col=0)
    if args.measure not in df_measures.columns:
        raise ValueError(
            f"Unknown --measure '{args.measure}'. Available: {', '.join(df_measures.columns.tolist())}"
        )

    a = int(args.index_a)
    b = int(args.index_b)
    if a == b:
        raise ValueError("index_a and index_b must be different")
    if a > b:
        a, b = b, a

    row_idx = get_index_sim(n_stacks, a, b)
    score = float(df_measures[args.measure].loc[row_idx])
    print(f"mode=lite measure={args.measure} n_stacks={n_stacks} a={a} b={b} row={row_idx} score={score:.6f}")
    return 0


def cmd_deeplsh(args) -> int:
    from train_deeplsh import main as train_main

    argv = [
        "--measure",
        args.measure,
        "--n",
        str(args.n),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--m",
        str(args.m),
        "--b",
        str(args.b),
        "--seed",
        str(args.seed),
        "--lsh-param-index",
        str(args.lsh_param_index),
    ]
    if args.kw:
        argv.extend(["--kw", *[str(x) for x in args.kw]])
    if args.data_repo:
        argv.extend(["--data-repo", args.data_repo])

    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *argv]
        train_main()
    finally:
        sys.argv = old_argv
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="DeepLSH local runner (non-notebook)")
    parser.add_argument("--data-repo", default=None, help="Path to data directory (default: <repo>/data)")

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="List available measures and commands")
    p_list.set_defaults(func=cmd_list)

    p_lite = sub.add_parser("lite", help="Query one similarity value from the precomputed pairs file")
    p_lite.add_argument("--measure", required=True)
    p_lite.add_argument("--index-a", type=int, required=True)
    p_lite.add_argument("--index-b", type=int, required=True)
    p_lite.add_argument("--n-stacks", type=int, default=1000)
    p_lite.set_defaults(func=cmd_lite)

    p_deep = sub.add_parser("deeplsh", help="Train DeepLSH for a selected measure and build LSH hash tables")
    p_deep.add_argument("--measure", required=True)
    p_deep.add_argument("--n", type=int, default=1000)
    p_deep.add_argument("--epochs", type=int, default=20)
    p_deep.add_argument("--batch-size", type=int, default=512)
    p_deep.add_argument("--m", type=int, default=64)
    p_deep.add_argument("--b", type=int, default=16)
    p_deep.add_argument("--kw", type=int, nargs="+", default=[3, 4])
    p_deep.add_argument("--seed", type=int, default=42)
    p_deep.add_argument("--lsh-param-index", type=int, default=2)
    p_deep.set_defaults(func=cmd_deeplsh)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
