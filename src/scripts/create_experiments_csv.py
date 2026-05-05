import json
import csv
from pathlib import Path

ROOT_DIR: Path = Path("src/results/embeddings")
OUTPUT_CSV: Path = Path("src/results/embeddings/all_experiments_summary.csv")

CSV_COLUMNS: list[str] = [
    "id",
    "experiment_id",
    "language",
    "embedding_model",
    "batch_size",
    "normalize_embeddings",
    "device",
    "embedding_dtype",
    "faiss_index",
    "prefix_mode",
    "chunk_size",
    "dimension",
    "chunks_count",
    "embedding_time_total_seconds",
    "embedding_time_mean_seconds",
    "faiss_creation_time_seconds",
    "faiss_index_size_mb",
    "metadata_size_mb",
    "total_storage_mb",
    "file_name",
    "file_chars",
    "file_num_chunks",
    "file_embedding_time_seconds",
    "file_embedding_memory_array_mb",
    "file_faiss_storage_mb",
    "file_text_storage_mb_est",
    "file_faiss_id_start",
    "file_faiss_id_count",
]

rows: list = []
id_number: int = 1

for experiment_dir in sorted(ROOT_DIR.glob("experiment_*")):
    if not experiment_dir.is_dir():
        continue

    for language_dir in sorted(experiment_dir.iterdir()):
        if not language_dir.is_dir():
            continue

        metrics_file = language_dir / "metrics.json"
        if not metrics_file.exists():
            continue

        try:
            with open(metrics_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            top = {
                "experiment_id": data.get("experiment_id"),
                "language": data.get("language"),
                "embedding_model": data.get("embedding_model"),
                "batch_size": data.get("batch_size"),
                "normalize_embeddings": data.get("normalize_embeddings"),
                "device": data.get("device"),
                "embedding_dtype": data.get("embedding_dtype"),
                "faiss_index": data.get("faiss_index"),
                "prefix_mode": data.get("prefix_mode"),
                "chunk_size": data.get("chunk_size"),
                "dimension": data.get("embedding_dimension"),
                "chunks_count": data.get("documents_count"),
                "embedding_time_total_seconds": data.get("embedding_time_total_seconds"),
                "embedding_time_mean_seconds": data.get("embedding_time_mean_seconds"),
                "faiss_creation_time_seconds": data.get("faiss_creation_time_seconds"),
                "faiss_index_size_mb": data.get("faiss_index_size_mb"),
                "metadata_size_mb": data.get("metadata_size_mb"),
                "total_storage_mb": data.get("total_storage_mb"),
            }

            for fdata in data["files"]:
                row = {
                    "id": id_number,
                    **top,
                    "file_name": fdata.get("filename"),
                    "file_chars": fdata.get("total_chars"),
                    "file_num_chunks": fdata.get("num_chunks"),
                    "file_embedding_time_seconds": fdata.get("embedding_time_seconds"),
                    "file_embedding_memory_array_mb": fdata.get("embedding_memory_array_mb"),
                    "file_faiss_storage_mb": fdata.get("faiss_storage_mb"),
                    "file_text_storage_mb_est": fdata.get("text_storage_mb_est"),
                    "file_faiss_id_start": fdata.get("faiss_id_start"),
                    "file_faiss_id_count": fdata.get("faiss_id_count"),
                }
                rows.append(row)
                id_number += 1

        except Exception as e:
            print(f"Erro ao processar {metrics_file}: {e}")

OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)

print(f"CSV file created: {OUTPUT_CSV}")
print(f"Total rows: {len(rows)}")