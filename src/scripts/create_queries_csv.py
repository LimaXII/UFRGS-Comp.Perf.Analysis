#!/usr/bin/env python3
from pathlib import Path
import csv
import sys

QUERIES_DIR = Path("src") / "results" / "queries"
EMBEDDINGS_CSV = Path("src") / "results" / "embeddings" / "all_experiments_summary.csv"

OUTPUT_CSV = QUERIES_DIR / "all_experiments_summary.csv"


def load_chunk_sizes():
    chunk_sizes = {}

    if not EMBEDDINGS_CSV.exists():
        print(f"Arquivo não encontrado: {EMBEDDINGS_CSV}", file=sys.stderr)
        sys.exit(1)

    with EMBEDDINGS_CSV.open("r", encoding="utf-8-sig", newline="") as fin:
        reader = csv.DictReader(fin)

        for row in reader:
            exp_id = row.get("experiment_id", "").strip()
            chunk_size = row.get("chunk_size", "").strip()

            if not exp_id:
                continue

            exp_key = f"experiment_{exp_id}"

            # salva apenas o primeiro encontrado
            if exp_key not in chunk_sizes:
                chunk_sizes[exp_key] = chunk_size

    return chunk_sizes


def main():
    csv_files = sorted(QUERIES_DIR.glob("experiment_*/*/results.csv"))

    if not csv_files:
        print(
            f"Nenhum results.csv encontrado em "
            f"{QUERIES_DIR}/experiment_*/*/results.csv",
            file=sys.stderr
        )
        sys.exit(1)

    chunk_sizes = load_chunk_sizes()

    unified_headers = []
    seen = set()

    for f in csv_files:
        with f.open("r", encoding="utf-8-sig", newline="") as fin:
            reader = csv.reader(fin)
            header = next(reader, None)

            if not header:
                continue

            for h in header:
                if h not in seen:
                    unified_headers.append(h)
                    seen.add(h)

    extras = [
        "chunk_size",
        "source_experiment_dir",
        "source_language_dir",
        "source_csv_path"
    ]

    for col in extras:
        if col not in seen:
            unified_headers.append(col)
            seen.add(col)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(
            fout,
            fieldnames=unified_headers,
            extrasaction="ignore"
        )

        writer.writeheader()

        for f in csv_files:
            exp_dir = f.parent.parent.name
            lang_dir = f.parent.name

            chunk_size = chunk_sizes.get(exp_dir, "")

            with f.open("r", encoding="utf-8-sig", newline="") as fin:
                reader = csv.DictReader(fin)

                for row in reader:
                    for k in unified_headers:
                        row.setdefault(k, "")

                    row["chunk_size"] = chunk_size
                    row["source_experiment_dir"] = exp_dir
                    row["source_language_dir"] = lang_dir
                    row["source_csv_path"] = str(f)

                    writer.writerow(row)

    print(f"CSV criado com sucesso em: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()