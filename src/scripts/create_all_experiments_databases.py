import os
import csv
import json
import time
import pickle
import psutil
import faiss
import numpy as np
import torch

from pathlib import Path
from statistics import mean, stdev
from sentence_transformers import SentenceTransformer
from transformers import logging

logging.set_verbosity_error()

EXPERIMENTS_CSV: str = "src/experiments.csv"
BASE_DOCS_PATH: str = "data/base_docs"
DATABASE_PATH: str = "src/database"
RESULTS_PATH: str = "src/results/embeddings"

EMBEDDING_BENCHMARK_RUNS: int = 3

def get_system_usage(
    device: str
) -> dict:

    process = psutil.Process(os.getpid())

    metrics = {
        "rss_mb": process.memory_info().rss / (1024 ** 2),
        "vms_mb": process.memory_info().vms / (1024 ** 2),
        "cpu_percent": psutil.cpu_percent(interval=0.1)
    }

    if device == "cuda" and torch.cuda.is_available():

        torch.cuda.synchronize()

        metrics["gpu_allocated_mb"] = (
            torch.cuda.memory_allocated() / (1024 ** 2)
        )

        metrics["gpu_reserved_mb"] = (
            torch.cuda.memory_reserved() / (1024 ** 2)
        )

        metrics["gpu_peak_allocated_mb"] = (
            torch.cuda.max_memory_allocated() / (1024 ** 2)
        )

    return metrics

def load_markdown_documents(
    folder_path: str
) -> tuple[list, list]:
    documents: list = []
    filenames: list = []

    markdown_files = sorted(Path(folder_path).glob("*.md"))

    for file_path in markdown_files:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        documents.append(content)
        filenames.append(file_path.name)

    return documents, filenames


def chunk_text(
    text: str, 
    chunk_size: int
) -> list:

    if chunk_size == 0:
        return [text]

    words = text.split()

    chunks: list = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks

def apply_prefix(
    documents: list, 
    prefix_mode: str
) -> list:

    if prefix_mode == "none":
        return documents

    if prefix_mode == "e5":
        return [f"passage: {doc}" for doc in documents]

    return documents


def load_experiments(
    csv_path: str
) -> list:

    experiments: list = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:

            row["batch_size"] = int(row["batch_size"])
            row["normalize_embeddings"] = (
                row["normalize_embeddings"].lower() == "true"
            )
            row["chunk_size"] = int(row["chunk_size"])

            experiments.append(row)

    return experiments

base_docs: Path = Path(BASE_DOCS_PATH)
database_root: Path = Path(DATABASE_PATH)
results_root: Path = Path(RESULTS_PATH)
database_root.mkdir(parents=True, exist_ok=True)
results_root.mkdir(parents=True, exist_ok=True)

language_folders: list = [
    folder for folder in base_docs.iterdir()
    if folder.is_dir()
]

experiments: list = load_experiments(EXPERIMENTS_CSV)

for experiment in experiments:

    experiment_id = experiment["experiment_id"]

    print(f"\n============================")
    print(f"Running Experiment {experiment_id}")
    print(f"============================")

    model_name = experiment["model_name"]
    batch_size = experiment["batch_size"]
    normalize_embeddings = experiment["normalize_embeddings"]
    device = experiment["device"]
    embedding_dtype = experiment["embedding_dtype"]
    faiss_index_type = experiment["faiss_index"]
    prefix_mode = experiment["prefix_mode"]
    chunk_size = experiment["chunk_size"]

    print(f"Loading model: {model_name}")

    model = SentenceTransformer(
        model_name,
        device=device
    )

    experiment_database_root = (
        database_root / f"experiment_{experiment_id}"
    )

    experiment_results_root = (
        results_root / f"experiment_{experiment_id}"
    )

    experiment_database_root.mkdir(parents=True, exist_ok=True)
    experiment_results_root.mkdir(parents=True, exist_ok=True)

    for language_folder in language_folders:

        language_code = language_folder.name

        print(f"\nProcessing language: {language_code}")

        db_output_folder = (
            experiment_database_root / language_code
        )

        results_output_folder = (
            experiment_results_root / language_code
        )

        db_output_folder.mkdir(parents=True, exist_ok=True)
        results_output_folder.mkdir(parents=True, exist_ok=True)

        documents, filenames = load_markdown_documents(language_folder)

        if len(documents) == 0:
            continue

        chunked_documents: list = []
        chunked_filenames: list = []

        for doc, filename in zip(documents, filenames):

            chunks = chunk_text(doc, chunk_size)

            for chunk in chunks:
                chunked_documents.append(chunk)
                chunked_filenames.append(filename)
        
        chunked_documents = apply_prefix(
            chunked_documents,
            prefix_mode
        )

        metrics = {
            "experiment_id": experiment_id,
            "language": language_code,
            "embedding_model": model_name,
            "batch_size": batch_size,
            "normalize_embeddings": normalize_embeddings,
            "device": device,
            "embedding_dtype": embedding_dtype,
            "faiss_index": faiss_index_type,
            "prefix_mode": prefix_mode,
            "chunk_size": chunk_size,
            "documents_count": len(chunked_documents),
            "embedding_runs": []
        }

        embeddings_runs: list = []

        for run in range(EMBEDDING_BENCHMARK_RUNS):

            print(
                f"Embedding benchmark run {run + 1}/"
                f"{EMBEDDING_BENCHMARK_RUNS}"
            )

            start_memory = get_system_usage(device)
            start_time = time.perf_counter()

            embeddings = model.encode(
                chunked_documents,
                batch_size=batch_size,
                normalize_embeddings=normalize_embeddings,
                convert_to_numpy=True,
                show_progress_bar=True,
            )

            if embedding_dtype == "float16":
                embeddings = embeddings.astype(np.float16)
            else:
                embeddings = embeddings.astype(np.float32)

            end_time = time.perf_counter()
            end_memory = get_system_usage(device)

            elapsed = end_time - start_time

            metrics["embedding_runs"].append({
                "run": run + 1,
                "time_seconds": elapsed,
                "start_memory": start_memory,
                "end_memory": end_memory,
            })

            embeddings_runs.append(embeddings)

        final_embeddings = embeddings_runs[-1]

        execution_times = [
            item["time_seconds"]
            for item in metrics["embedding_runs"]
        ]

        metrics["embedding_time_mean_seconds"] = mean(execution_times)

        if len(execution_times) > 1:
            metrics["embedding_time_std_seconds"] = (
                stdev(execution_times)
            )
        else:
            metrics["embedding_time_std_seconds"] = 0

        embedding_dimension = final_embeddings.shape[1]

        metrics["embedding_dimension"] = embedding_dimension

        metrics["embedding_memory_array_mb"] = (
            final_embeddings.nbytes / (1024 ** 2)
        )

        print("Creating FAISS index...")

        faiss_start_memory = get_system_usage(device)
        faiss_start_time = time.perf_counter()

        if faiss_index_type == "ip":
            index = faiss.IndexFlatIP(embedding_dimension)
        else:
            index = faiss.IndexFlatL2(embedding_dimension)

        index.add(np.array(final_embeddings, dtype=np.float32))

        faiss_end_time = time.perf_counter()
        faiss_end_memory = get_system_usage(device)

        metrics["faiss_creation_time_seconds"] = (
            faiss_end_time - faiss_start_time
        )

        metrics["faiss_memory_before"] = faiss_start_memory
        metrics["faiss_memory_after"] = faiss_end_memory

        index_path = db_output_folder / "index.faiss"
        metadata_path = db_output_folder / "documents.pkl"

        faiss.write_index(index, str(index_path))

        metadata = {
            "documents": chunked_documents,
            "filenames": chunked_filenames,
        }

        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)

        index_size_mb = index_path.stat().st_size / (1024 ** 2)
        metadata_size_mb = metadata_path.stat().st_size / (1024 ** 2)

        metrics["faiss_index_size_mb"] = index_size_mb
        metrics["metadata_size_mb"] = metadata_size_mb
        metrics["total_storage_mb"] = (
            index_size_mb + metadata_size_mb
        )

        metrics_path = results_output_folder / "metrics.json"

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)

        print(f"Finished language: {language_code}")

print("\nAll experiments completed")