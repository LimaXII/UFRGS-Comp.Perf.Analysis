# To run all experiments: poetry run python src/scripts/run_queries.py --experiments all --ollama_model qwen2.5:7b
# To run specific experiments: poetry run python src/scripts/run_queries.py --experiments 33,34,35 --ollama_model qwen2.5:7b

# Before running, install Ollama, via: https://ollama.com/download
# Then, run the following command: ollama pull qwen2.5:7b
# You can test the model, running: ollama run qwen2.5:7b

import os
import re
import csv
import json
import time
import pickle
import argparse
from pathlib import Path
from statistics import mean, stdev

import psutil
import numpy as np
import torch
import faiss
import ollama

from sentence_transformers import SentenceTransformer
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()

EXPERIMENTS_CSV: Path = Path("src/experiments.csv")
DATABASE_PATH: Path = Path("src/database")
BASE_QUESTIONS_PATH: Path = Path("data/base_questions")
RESULTS_QUERIES_PATH: Path = Path("src/results/queries")

DEFAULT_OLLAMA_MODEL: str = "qwen2.5:7b"
TOP_K: int = 6
CONTEXT_CHAR_LIMIT: int = 2000
OLLAMA_TEMPERATURE: float = 0.2

SYSTEM_PROMPT: str = """
    You are a retrieval-augmented assistant.

    Rules:
    1) Answer ONLY using the provided CONTEXT.
    2) If the answer is not in the CONTEXT, say you couldn't find it in the documents.
    3) Answer in the same language as the question.
    4) Be extremely concise.
    5) Prefer answers with 1-5 short sentences.
    6) Avoid explanations, introductions, or repetition.
    7) When possible, mention source filename(s) like [file.md].
"""

def get_system_usage(
    device: str
) -> dict:
    process = psutil.Process(os.getpid())
    metrics: dict = {
        "rss_mb": process.memory_info().rss / (1024 ** 2),
        "vms_mb": process.memory_info().vms / (1024 ** 2),
        "cpu_percent": psutil.cpu_percent(interval=0.1),
    }

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
        metrics["gpu_allocated_mb"] = torch.cuda.memory_allocated() / (1024 ** 2)
        metrics["gpu_reserved_mb"] = torch.cuda.memory_reserved() / (1024 ** 2)
        metrics["gpu_peak_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024 ** 2)

    return metrics

def load_experiments(
    csv_path: Path
) -> list[dict]:
    experiments: list = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["experiment_id"] = int(row["experiment_id"])
            row["batch_size"] = int(row["batch_size"])
            row["normalize_embeddings"] = (row["normalize_embeddings"].lower() == "true")
            row["chunk_size"] = int(row["chunk_size"])
            experiments.append(row)
    return experiments

def find_single_md_file(
    folder: Path
) -> Path | None:
    md_files = sorted(folder.glob("*.md"))
    if not md_files:
        return None
    return md_files[0]

def sanitize_for_csv(
    text: str
) -> str:
    if not text:
        return ""

    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def load_questions(
    md_path: Path
) -> list[dict]:
    text = md_path.read_text(encoding="utf-8")
    questions = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^(\d+)\s*[\.\)\-]\s*(.+?)\s*$", line)
        if m:
            questions.append({"query_number": int(m.group(1)), "query": m.group(2)})
    return questions

def load_faiss_and_metadata(
    experiment_id: int, 
    language_code: str
):
    db_folder = DATABASE_PATH / f"experiment_{experiment_id}" / language_code
    index_path = db_folder / "index.faiss"
    metadata_path = db_folder / "documents.pkl"

    if not index_path.exists() or not metadata_path.exists():
        return None, None, None

    index = faiss.read_index(str(index_path))
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    documents = metadata["documents"]
    filenames = metadata["filenames"]
    return index, documents, filenames

def prefix_query(
    query: str, 
    prefix_mode: str
) -> str:
    if prefix_mode == "e5":
        return f"query: {query}"
    return query

def cast_like_experiment(
    vec: np.ndarray, 
    embedding_dtype: str
) -> np.ndarray:
    vec = vec.astype(np.float32)
    if embedding_dtype == "float16":
        vec = vec.astype(np.float16).astype(np.float32)
    return vec

def format_context(
    hits: list[dict]
) -> str:
    parts = []
    for h in hits:
        parts.append(
            f"### Hit {h['rank']} | score={h['score']:.6f} | source=[{h['filename']}]"
        )
    return "\n".join(parts)

def ollama_answer(
    model: str, 
    system_prompt: str, 
    user_prompt: str
) -> tuple[str, dict]:
    resp = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": OLLAMA_TEMPERATURE},
    )

    text = resp["message"]["content"]
    meta = {
        "response_chars": len(text),
        "response_words": len(text.split()),
    }
    return text, meta

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", type=str, default="all",
                        help="Ex: 'all' ou '1,2,3'")
    parser.add_argument("--ollama_model", type=str, default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--top_k", type=int, default=TOP_K)
    args = parser.parse_args()

    RESULTS_QUERIES_PATH.mkdir(parents=True, exist_ok=True)

    experiments = load_experiments(EXPERIMENTS_CSV)
    if args.experiments != "all":
        wanted = {int(x.strip()) for x in args.experiments.split(",") if x.strip()}
        experiments = [e for e in experiments if e["experiment_id"] in wanted]

    language_folders = sorted([p for p in BASE_QUESTIONS_PATH.iterdir() if p.is_dir()])

    for exp in experiments:
        exp_id = exp["experiment_id"]
        model_name = exp["model_name"]
        batch_size = exp["batch_size"]
        normalize_embeddings = exp["normalize_embeddings"]
        device = exp["device"]
        embedding_dtype = exp["embedding_dtype"]
        prefix_mode = exp["prefix_mode"]

        print(f"\n============================")
        print(f"Experiment {exp_id}")
        print(f"  emb_model={model_name}")
        print(f"  device={device} | batch={batch_size} | norm={normalize_embeddings}")
        print(f"  embedding_dtype={embedding_dtype} | prefix_mode={prefix_mode}")
        print(f"  ollama_model={args.ollama_model} | top_k={args.top_k}")
        print(f"============================")

        encoder = SentenceTransformer(model_name, device=device)

        exp_out_root = RESULTS_QUERIES_PATH / f"experiment_{exp_id}"
        exp_out_root.mkdir(parents=True, exist_ok=True)

        for lang_folder in language_folders:
            language_code = lang_folder.name
            md_file = find_single_md_file(lang_folder)
            if md_file is None:
                print(f"[SKIP] {language_code}: no .md file found on {lang_folder}")
                continue

            questions = load_questions(md_file)
            if not questions:
                print(f"[SKIP] {language_code}: cannot parse questions on {md_file.name}")
                continue

            index, documents, filenames = load_faiss_and_metadata(exp_id, language_code)
            if index is None:
                print(f"[SKIP] {language_code}: no FAISS/metadata found in src/database/experiment_{exp_id}/{language_code}")
                continue

            lang_out = exp_out_root / language_code
            lang_out.mkdir(parents=True, exist_ok=True)

            out_jsonl = lang_out / "results.jsonl"
            out_csv = lang_out / "results.csv"

            if out_jsonl.exists():
                out_jsonl.unlink()
            if out_csv.exists():
                out_csv.unlink()

            csv_fields = [
                "experiment_id", "language",
                "query_number", "query",
                "response",
                "embedding_time_s", "retrieval_time_s", "llm_time_s", "total_time_s",
                "start_rss_mb", "end_rss_mb",
                "device",
                "embedding_model",
                "normalize_embeddings",
                "embedding_dtype",
                "prefix_mode",
                "faiss_top_k",
                "hits_json",
                "ollama_model",
                "ollama_meta_json",
            ]

            total_times = []

            with open(out_csv, "w", encoding="utf-8", newline="") as fcsv:
                writer = csv.DictWriter(fcsv, fieldnames=csv_fields)
                writer.writeheader()

                for q in questions:
                    qnum = q["query_number"]
                    query = q["query"]

                    start_mem = get_system_usage(device)
                    t0 = time.perf_counter()

                    # 1) Query embedding
                    te0 = time.perf_counter()
                    q_text = prefix_query(query, prefix_mode)
                    q_emb = encoder.encode(
                        [q_text],
                        batch_size=1,
                        normalize_embeddings=normalize_embeddings,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                    )
                    q_emb = cast_like_experiment(q_emb, embedding_dtype)
                    embedding_time = time.perf_counter() - te0

                    # 2) Retrieval
                    tr0 = time.perf_counter_ns()
                    D, I = index.search(q_emb, args.top_k)
                    retrieval_time = (time.perf_counter_ns() - tr0) / 1e9

                    hits = []
                    for rank, (score, idx) in enumerate(zip(D[0], I[0]), start=1):
                        if idx < 0:
                            continue
                        text = documents[idx]
                        if CONTEXT_CHAR_LIMIT > 0 and len(text) > CONTEXT_CHAR_LIMIT:
                            text = text[:CONTEXT_CHAR_LIMIT] + " ...[truncated]"
                        hits.append({
                            "rank": rank,
                            "score": float(score),
                            "doc_index": int(idx),
                            "filename": filenames[idx],
                            "text": text,
                        })

                    context = format_context(hits)

                    user_prompt: str = f"""CONTEXT:
                        {context}

                        QUESTION:
                        {query}

                        ANSWER:"""

                    # 3) LLM (Ollama)
                    tl0 = time.perf_counter()
                    response_text, ollama_meta = ollama_answer(
                        model=args.ollama_model,
                        system_prompt=SYSTEM_PROMPT,
                        user_prompt=user_prompt,
                    )
                    response_text = sanitize_for_csv(response_text)
                    llm_time = time.perf_counter() - tl0

                    t1 = time.perf_counter()
                    end_mem = get_system_usage(device)

                    total_time = t1 - t0
                    total_times.append(total_time)

                    record: dict = {
                        "experiment_id": exp_id,
                        "language": language_code,
                        "query_number": qnum,
                        "query": query,
                        "response": response_text,
                        "embedding_time_s": embedding_time,
                        "retrieval_time_s": retrieval_time,
                        "llm_time_s": llm_time,
                        "total_time_s": total_time,
                        "start_rss_mb": start_mem.get("rss_mb"),
                        "end_rss_mb": end_mem.get("rss_mb"),
                        "device": device,
                        "embedding_model": model_name,
                        "normalize_embeddings": normalize_embeddings,
                        "embedding_dtype": embedding_dtype,
                        "prefix_mode": prefix_mode,
                        "faiss_top_k": args.top_k,
                        "hits_json": json.dumps(hits, ensure_ascii=False),
                        "ollama_model": args.ollama_model,
                        "ollama_meta_json": json.dumps(ollama_meta, ensure_ascii=False),
                    }

                    with open(out_jsonl, "a", encoding="utf-8") as fj:
                        fj.write(json.dumps(record, ensure_ascii=False) + "\n")

                    writer.writerow(record)
                    fcsv.flush()

                    print(
                        f"[{exp_id}][{language_code}] Q{qnum:02d} ok | "
                        f"total={total_time:.2f}s | llm={llm_time:.2f}s | "
                        f"emb={embedding_time:.2f}s | retr={retrieval_time:.2f}s"
                    )

            summary: dict = {
                "experiment_id": exp_id,
                "language": language_code,
                "questions_count": len(questions),
                "ollama_model": args.ollama_model,
                "top_k": args.top_k,
                "mean_total_time_s": mean(total_times) if total_times else None,
                "std_total_time_s": stdev(total_times) if len(total_times) > 1 else 0.0,
            }
            (lang_out / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    print("\nAll done.")

if __name__ == "__main__":
    main()