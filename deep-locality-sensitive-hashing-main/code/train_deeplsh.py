import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import (
    Conv1D,
    Dense,
    Embedding,
    GlobalMaxPooling1D,
    Input,
    concatenate,
)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-repo", default=None)
    parser.add_argument(
        "--measure",
        required=True,
        help="Target similarity measure column in similarity-measures-pairs.csv (e.g., TraceSim, Jaccard, Cosine, TfIdf, Levensh, PDM, Brodie, DURFEX, Lerch, Moroo)",
    )
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--m", type=int, default=64)
    parser.add_argument("--b", type=int, default=16)
    parser.add_argument("--kw", type=int, nargs="+", default=[3, 4])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-model", action="store_true", default=True)
    parser.add_argument("--save-hash-tables", action="store_true", default=True)
    parser.add_argument(
        "--lsh-param-index",
        type=int,
        default=2,
        help="Index in lsh_hyperparams(m). Default 2 corresponds to (4,16) when m=64.",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    python_packages = os.path.join(project_root, "code", "python-packages")
    if python_packages not in sys.path:
        sys.path.insert(0, python_packages)

    from deep_hashing_models import intermediate_model_trained, siamese_model, train_siamese_model
    from lsh_search import convert_to_hamming, create_hash_tables, lsh_hyperparams

    if args.data_repo is None:
        data_repo = os.path.join(project_root, "data")
    else:
        data_repo = args.data_repo

    # 读取数据
    df_distinct_stacks = pd.read_csv(os.path.join(data_repo, "frequent_stack_traces.csv"), index_col=0)
    df_measures = pd.read_csv(os.path.join(data_repo, "similarity-measures-pairs.csv"), index_col=0)

    if args.measure not in df_measures.columns:
        raise ValueError(
            f"Unknown --measure '{args.measure}'. Available: {', '.join(df_measures.columns.tolist())}"
        )

    if args.n is not None:
        df_distinct_stacks = df_distinct_stacks.head(args.n)

    n_stacks = df_distinct_stacks.shape[0]
    if n_stacks < 2:
        raise ValueError("Need at least 2 stacks")

    if "stackTraceCusto" not in df_distinct_stacks.columns:
        raise ValueError("Expected column 'stackTraceCusto' in frequent_stack_traces.csv")

    # 将stackTraceCusto转换为list，形象理解就是把word内容变成excel，方便后续标号
    df_distinct_stacks["listStackTrace"] = df_distinct_stacks["stackTraceCusto"].apply(lambda x: str(x).split("\n"))
    corpus = df_distinct_stacks["listStackTrace"].tolist()

    # 给list编号，并为每个编号生成向量
    frames = pd.Series(list(set([elt for l in corpus for elt in l])))
    df_frames = pd.DataFrame()
    df_frames["frame"] = pd.get_dummies(frames).T.reset_index().rename(columns={"index": "frame"})["frame"]
    df_frames["embedding"] = pd.get_dummies(frames).T.reset_index().apply(lambda x: x[1:].values, axis=1)

    from deep_hashing_models import index_frame, padding

    df_distinct_stacks["rankFrames"] = df_distinct_stacks["listStackTrace"].apply(lambda x: index_frame(x, df_frames))

    # 生成pair
    df_pairs = pd.DataFrame()
    df_pairs[args.measure] = df_measures[args.measure]

    from deep_hashing_models import assign_stacks
    from similarities import rowIndex

    df_pairs["stack1"] = df_pairs.apply(lambda x: assign_stacks(rowIndex(x), df_distinct_stacks)[0], axis=1)
    df_pairs["stack2"] = df_pairs.apply(lambda x: assign_stacks(rowIndex(x), df_distinct_stacks)[1], axis=1)

    embeddings = 1 * np.random.randn(df_frames.shape[0] + 1, df_frames["embedding"][0].shape[0])
    embeddings[0] = 0
    embeddings[1:] = np.vstack(df_frames["embedding"].tolist())

    X = df_pairs[["stack1", "stack2"]]
    Y = df_pairs[args.measure].values

    from sklearn.model_selection import train_test_split

    # 划分训练集和验证集
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.2, random_state=args.seed)

    # padding(填充，对齐)
    max_length = df_distinct_stacks["listStackTrace"].apply(lambda x: len(x)).max()
    X_train = padding(X_train, max_length)
    X_validation = padding(X_validation, max_length)

    size_hash_vector = args.m * args.b

    # 构建模型
    input_tensor = Input(shape=(max_length,))
    input_layer = Embedding(
        len(embeddings),
        df_frames.shape[0],
        weights=[embeddings],
        input_shape=(max_length,),
        trainable=False,
    )(input_tensor)

    submodels = []
    for kw in args.kw:
        conv_layer = Conv1D(1024, kw, activation="tanh")(input_layer)
        maxpool_layer = GlobalMaxPooling1D()(conv_layer)
        submodels.append(maxpool_layer)

    conc_layer = concatenate(submodels, axis=1)
    dense_layer = Dense(size_hash_vector, activation="tanh")(conc_layer)
    shared_model = Sequential()
    shared_model.add(Model(inputs=input_tensor, outputs=dense_layer))

    model = siamese_model(shared_model, (max_length,), args.b, args.m, is_sparse=False, print_summary=False)
    train_siamese_model(
        model,
        X_train,
        X_validation,
        Y_train,
        Y_validation,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )

    # 训练好的模型，单边编码器
    intermediate_model = intermediate_model_trained(shared_model, output_layer=0, CNN=True, input_tensor=input_tensor)

    # 对distinct_stacks生成哈希向量
    ranks = pad_sequences(
        df_distinct_stacks["rankFrames"],
        padding="post",
        truncating="post",
        maxlen=max_length,
    )
    # 从这以后，LSH开始起作用
    hash_vectors = intermediate_model.predict(ranks, verbose=0)

    # 转换为hamming/分桶的形式，建立哈希表，注意L、K、b含义
    hash_vectors_hamming = convert_to_hamming(hash_vectors)
    params = lsh_hyperparams(args.m)
    if args.lsh_param_index < 0 or args.lsh_param_index >= len(params):
        raise ValueError(f"--lsh-param-index out of range: {args.lsh_param_index} (len={len(params)})")

    L, K = params[args.lsh_param_index]
    hash_tables = create_hash_tables(L, K, args.b, hash_vectors_hamming)

    models_dir = os.path.join(project_root, "code", "Models")
    hash_tables_dir = os.path.join(project_root, "code", "Hash-Tables")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(hash_tables_dir, exist_ok=True)

    safe_measure = "".join(c for c in args.measure if c.isalnum() or c in ("-", "_"))

    if args.save_model:
        intermediate_model.save(os.path.join(models_dir, f"model-deep-lsh-{safe_measure}.model"))

    if args.save_hash_tables:
        with open(os.path.join(hash_tables_dir, f"hash_tables_deeplsh_{safe_measure}.pkl"), "wb") as f:
            pickle.dump(hash_tables, f)

    print(
        " ".join(
            [
                "done",
                f"measure={args.measure}",
                f"n_stacks={n_stacks}",
                f"max_length={max_length}",
                f"m={args.m}",
                f"b={args.b}",
                f"size_hash_vector={size_hash_vector}",
                f"lsh=(L={L},K={K})",
                f"models_dir={models_dir}",
                f"hash_tables_dir={hash_tables_dir}",
            ]
        )
    )


if __name__ == "__main__":
    main()
