# Experiment Execution Guide

This document describes the required execution order of the scripts used to prepare the datasets and run the retrieval experiments.

## Prerequisites

Ensure that all project dependencies have been installed and that the Poetry environment is properly configured before executing any script.

---

## Step 1 – Translate All Documents

Translate all source documents into the target languages.

```bash
poetry run python src/scripts/translate_all_docs.py
```

---

## Step 2 – Translate All Questions

Translate all evaluation questions into the target languages.

```bash
poetry run python src/scripts/translate_all_questions.py
```

---

## Step 3 – Generate the Full Factorial Design

Create the CSV file containing all combinations of experimental parameters.

```bash
poetry run python src/scripts/create_full_factorial_csv.py
```

Output:

* Full factorial design CSV containing all experiment configurations.

---

## Step 4 – Create all databases

Create all databases from experiments and replications (10 replications by default).

```bash
poetry run python src/scripts/run_embeddings.py
```

Output:

* .json file for every language, replication and experiment containing all collected data.

---

## Step 5 - Unify all databases CSV

Unify all .json files created to a unique csv file called `all_experiments_summary.csv`

```bash
poetry run python src/scripts/unify_embeddings_json.py
```

Output: 

* .csv file containing all collected data from all embeddings.
located in: `src/results/embeddings/all_experiments_summary.csv`

---

## Step 6 - Execute all queries and retrieval

Run all queries for all 

```bash
poetry run python src/scripts/run_queries.py
```

---

## Step 7 - Unity all queries results (Final output)

Unify all files created to a unique csv file called `??.csv`

```bash
poetry run python src/scripts/unify_queries_json.py
```

Output:

* .csv file containing all output data collected from queries
located in: `src/results/queries/all_experiments_summary.csv`