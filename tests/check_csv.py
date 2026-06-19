import csv
from collections import deque, Counter

CSV_PATH = "src/results/queries/all_experiments_summary.csv"

NUM_EXEMPLOS_INICIO = 5
NUM_EXEMPLOS_FIM = 5

with open(CSV_PATH, "r", encoding="utf-8", newline="") as f:
    reader = csv.reader(f)

    # Cabeçalho
    columns = next(reader)

    print("=" * 80)
    print("COLUNAS")
    print("=" * 80)

    for i, col in enumerate(columns):
        print(f"{i}: {col}")

    print(f"\nTotal de colunas: {len(columns)}")

    # Localiza a coluna experiment_id
    try:
        experiment_idx = columns.index("experiment_id")
    except ValueError:
        print("\nERRO: coluna 'experiment_id' não encontrada!")
        exit()

    first_rows = []
    last_rows = deque(maxlen=NUM_EXEMPLOS_FIM)

    total_rows = 0
    max_experiment = -1
    experiment_counter = Counter()

    for row in reader:
        total_rows += 1

        if len(first_rows) < NUM_EXEMPLOS_INICIO:
            first_rows.append(row)

        last_rows.append(row)

        try:
            exp_id = int(row[experiment_idx])

            if exp_id > max_experiment:
                max_experiment = exp_id

            experiment_counter[exp_id] += 1

        except (ValueError, IndexError):
            pass

print("\n" + "=" * 80)
print("ESTATÍSTICAS")
print("=" * 80)

print(f"Total de linhas: {total_rows:,}")
print(f"Total de experimentos únicos: {len(experiment_counter)}")
print(f"Maior experiment_id encontrado: {max_experiment}")

print("\n" + "=" * 80)
print("ÚLTIMOS 10 EXPERIMENTOS")
print("=" * 80)

for exp_id in sorted(experiment_counter.keys())[-10:]:
    print(
        f"experiment_id={exp_id} -> "
        f"{experiment_counter[exp_id]:,} linhas"
    )

print("\n" + "=" * 80)
print("TODOS OS EXPERIMENTOS")
print("=" * 80)

for exp_id in sorted(experiment_counter.keys()):
    print(
        f"experiment_id={exp_id} -> "
        f"{experiment_counter[exp_id]:,} linhas"
    )

print("\n" + "=" * 80)
print("PRIMEIRAS LINHAS")
print("=" * 80)

for i, row in enumerate(first_rows, start=1):
    print(f"\nLinha {i}:")
    print(dict(zip(columns, row)))

print("\n" + "=" * 80)
print("ÚLTIMAS LINHAS")
print("=" * 80)

for i, row in enumerate(last_rows, start=1):
    print(f"\nLinha final {i}:")
    print(dict(zip(columns, row)))