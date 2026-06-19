import itertools
import pandas as pd

# 2 levels
embedding_dtypes: list = ["float16", "float32"]
faiss_indexes: list = ["l2", "ip"]
# devices: list = ["cpu", "cuda"]

# 7 levels
batch_sizes: list = [8, 16, 32, 64, 128, 256, 512]
chunk_sizes: list = [128, 256, 512, 1024, 2048, 4096, 8192]

rows: list = []
exp_id: int = 1

for batch_size, dtype, index, chunk_size in itertools.product(
    batch_sizes,
    embedding_dtypes,
    faiss_indexes,
    chunk_sizes,
):
    rows.append([
        exp_id,
        batch_size,
        dtype,
        index,
        chunk_size
    ])
    exp_id += 1

df = pd.DataFrame(
    rows,
    columns=[
        "experiment_id",
        "batch_size",
        "embedding_dtype",
        "faiss_index",
        "chunk_size"
    ]
)

df.to_csv("src/experiments.csv", index=False)
print(f"Total experiments: {len(df)}")