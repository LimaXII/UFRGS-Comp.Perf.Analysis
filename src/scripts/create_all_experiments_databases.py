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
from statistics import mean
from sentence_transformers import SentenceTransformer
from transformers import logging

logging.set_verbosity_error()

EXPERIMENTS_CSV: str = "src/experiments.csv"
BASE_DOCS_PATH: str = "data/base_docs"
DATABASE_PATH: str = "src/database"
RESULTS_PATH: str = "src/results/embeddings"

MiB = 1024 ** 2

def get_system_usage(
    device: str
) -> dict:
    process = psutil.Process(os.getpid())
    metrics = {
        "rss_mb": process.memory_info().rss / MiB,
        "vms_mb": process.memory_info().vms / MiB,
        "cpu_percent": psutil.cpu_percent(interval=0.1)
    }
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
        metrics["gpu_allocated_mb"] = torch.cuda.memory_allocated() / MiB
        metrics["gpu_reserved_mb"] = torch.cuda.memory_reserved() / MiB
        metrics["gpu_peak_allocated_mb"] = torch.cuda.max_memory_allocated() / MiB
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
    docs: list, 
    prefix_mode: str
) -> list:
    if prefix_mode == "none":
        return docs
    if prefix_mode == "e5":
        return [f"passage: {doc}" for doc in docs]
    return docs

def encode_with_timing(
    model, 
    texts, 
    batch_size, 
    normalize_embeddings, 
    device
):
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    embs = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return embs, elapsed

base_docs: Path = Path(BASE_DOCS_PATH)
database_root: Path = Path(DATABASE_PATH)
results_root: Path = Path(RESULTS_PATH)
database_root.mkdir(parents=True, exist_ok=True)
results_root.mkdir(parents=True, exist_ok=True)

language_folders: list = [folder for folder in base_docs.iterdir() if folder.is_dir()]

def load_experiments(
    csv_path: str
) -> list:
    experiments: list = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["batch_size"] = int(row["batch_size"])
            row["normalize_embeddings"] = (row["normalize_embeddings"].lower() == "true")
            row["chunk_size"] = int(row["chunk_size"])
            experiments.append(row)
    return experiments

experiments: list = load_experiments(EXPERIMENTS_CSV)

for experiment in experiments:
    experiment_id = experiment["experiment_id"]

    print("\n============================")
    print(f"Running Experiment {experiment_id}")
    print("============================")

    model_name = experiment["model_name"]
    batch_size = experiment["batch_size"]
    normalize_embeddings = experiment["normalize_embeddings"]
    device = experiment["device"]
    embedding_dtype = experiment["embedding_dtype"]  
    faiss_index_type = experiment["faiss_index"]
    prefix_mode = experiment["prefix_mode"]
    chunk_size = experiment["chunk_size"]

    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name, device=device)

    experiment_database_root = database_root / f"experiment_{experiment_id}"
    experiment_results_root = results_root / f"experiment_{experiment_id}"
    experiment_database_root.mkdir(parents=True, exist_ok=True)
    experiment_results_root.mkdir(parents=True, exist_ok=True)

    for language_folder in language_folders:
        language_code = language_folder.name
        print(f"\nProcessing language: {language_code}")

        db_output_folder = experiment_database_root / language_code
        results_output_folder = experiment_results_root / language_code
        db_output_folder.mkdir(parents=True, exist_ok=True)
        results_output_folder.mkdir(parents=True, exist_ok=True)

        documents, filenames = load_markdown_documents(language_folder)
        if len(documents) == 0:
            print("No documents found. Skipping.")
            continue

        all_embeddings_f32: list = []
        all_documents_chunks: list = []
        all_filenames_per_chunk: list = []

        files_metrics: list = []
        total_embedding_times: list = []

        faiss_id_cursor: int = 0
        embedding_dimension: int = None

        for doc_text, filename in zip(documents, filenames):
            # 1) Chunking
            chunks = chunk_text(doc_text, chunk_size)
            if len(chunks) == 0:
                continue

            total_chars = len(doc_text)

            # 2) Prefix
            prefixed_chunks = apply_prefix(chunks, prefix_mode)

            # 3) Embedding com timing por ARQUIVO
            start_time = time.perf_counter()
            embs, encode_elapsed = encode_with_timing(
                model=model,
                texts=prefixed_chunks,
                batch_size=batch_size,
                normalize_embeddings=normalize_embeddings,
                device=device
            )

            if embedding_dtype == "float16":
                embs_cast = embs.astype(np.float16)
                emb_dtype_size = 2
            else:
                embs_cast = embs.astype(np.float32)
                emb_dtype_size = 4

            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            embedding_time_seconds = end_time - start_time  # inclui encode + cast

            # 4) Dimensão
            if embedding_dimension is None:
                embedding_dimension = embs.shape[1]

            # 5) Contribuição FAISS por arquivo
            num_vecs = embs.shape[0]
            faiss_bytes = num_vecs * embedding_dimension * 4  # float32
            faiss_storage_mb = faiss_bytes / MiB

            # 6) Memória do array de embeddings
            embedding_memory_array_mb = (embs_cast.nbytes) / MiB

            # 7) Estimativa de armazenamento de texto
            text_bytes = sum(len(ch.encode("utf-8")) for ch in chunks)
            text_storage_mb_est = text_bytes / MiB

            # 8) Guardar métricas do arquivo
            file_record = {
                "filename": filename,
                "total_chars": total_chars,
                "num_chunks": num_vecs,
                "embedding_time_seconds": embedding_time_seconds,
                "embedding_dimension": embedding_dimension,
                "embedding_dtype": embedding_dtype,
                "embedding_memory_array_mb": embedding_memory_array_mb,
                "faiss_storage_mb": faiss_storage_mb,
                "text_storage_mb_est": text_storage_mb_est,
                "faiss_id_start": faiss_id_cursor,
                "faiss_id_count": num_vecs
            }
            files_metrics.append(file_record)
            total_embedding_times.append(embedding_time_seconds)

            # 9) Acumular para index e metadata
            all_embeddings_f32.append(embs.astype(np.float32))  # index sempre em float32
            all_documents_chunks.extend(prefixed_chunks)
            all_filenames_per_chunk.extend([filename] * num_vecs)
            faiss_id_cursor += num_vecs

        # Concatenar tudo e criar o index por idioma
        if len(all_embeddings_f32) == 0:
            print(f"No chunks for language {language_code}. Skipping.")
            continue

        final_embeddings = np.vstack(all_embeddings_f32)
        documents_count = final_embeddings.shape[0]
        embedding_dimension = final_embeddings.shape[1]

        print("Creating FAISS index...")
        faiss_start_time = time.perf_counter()

        if faiss_index_type == "ip":
            index = faiss.IndexFlatIP(embedding_dimension)
        else:
            index = faiss.IndexFlatL2(embedding_dimension)

        index.add(final_embeddings)

        faiss_end_time = time.perf_counter()

        # Persistência: index e metadata
        index_path = db_output_folder / "index.faiss"
        metadata_path = db_output_folder / "documents.pkl"

        faiss.write_index(index, str(index_path))

        metadata = {
            "documents": all_documents_chunks, 
            "filenames": all_filenames_per_chunk
        }
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)

        # Tamanhos no disco
        index_size_mb = index_path.stat().st_size / MiB
        metadata_size_mb = metadata_path.stat().st_size / MiB
        total_storage_mb = index_size_mb + metadata_size_mb

        # Métricas agregadas por idioma
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
            "documents_count": documents_count,
            "embedding_time_total_seconds": sum(total_embedding_times),
            "embedding_time_mean_seconds": mean(total_embedding_times) if total_embedding_times else 0.0,
            "embedding_dimension": embedding_dimension,
            "faiss_creation_time_seconds": (faiss_end_time - faiss_start_time),
            "faiss_index_size_mb": index_size_mb,
            "metadata_size_mb": metadata_size_mb,
            "total_storage_mb": total_storage_mb,
            "files": files_metrics
        }

        metrics_path = results_output_folder / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)

        print(f"Finished language: {language_code}")

print("\nAll experiments completed")