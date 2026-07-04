library(tidyverse)

df <- read_csv("src/results/queries/all_experiments_summary.csv")

output_dir <- "src/results/graphs"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

save_graph <- function(plot, filename, width = 10, height = 6) {
  ggsave(
    filename = file.path(output_dir, filename),
    plot = plot,
    width = width,
    height = height
  )
}

## A) Queries - Experiment Graphs
## Experiments: All.

# ============================================================
# 1) Worst gold_rank by language (considering only retrievals with gold_found == TRUE)
# ============================================================

g1 <- df %>%
  filter(gold_found == TRUE) %>%
  group_by(experiment_id, language, replication_id) %>%
  summarise(
    gold_rank = mean(gold_rank, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  group_by(experiment_id, language) %>%
  summarise(
    mean_gold_rank = mean(gold_rank, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  group_by(language) %>%
  summarise(
    mean_gold_rank = mean(mean_gold_rank, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(desc(mean_gold_rank)) %>%
  slice_head(n = 10) %>%
  ggplot(aes(
    x = reorder(language, mean_gold_rank),
    y = mean_gold_rank
  )) +
  geom_col(fill = "indianred", width = 0.7) +
  geom_text(
    aes(label = round(mean_gold_rank, 2)),
    vjust = -0.4,
    size = 3.5
  ) +
  scale_y_continuous(
    limits = c(0, NA),
    expand = expansion(mult = c(0, 0.1))
  ) +
  labs(
    title = "10 idiomas com pior posicionamento médio do arquivo correto.",
    subtitle = "Valores maiores indicam que o arquivo correto apareceu em posições mais baixas no ranking.",
    x = "Idioma",
    y = "Posição média do arquivo correto"
  ) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

save_graph(g1, "g6_worst_gold_rank_languages_all_experiments.png")
print("[6/9] Gráfico 6 salvo em src/results/graphs/g6_worst_gold_rank_languages_all_experiments.png")

# ============================================================
# 2) Best gold_rank by language (considering only retrievals with gold_found == TRUE)
# ============================================================

g2 <- df %>%
  filter(gold_found == TRUE) %>%
  group_by(experiment_id, language, replication_id) %>%
  summarise(
    gold_rank = mean(gold_rank, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  group_by(experiment_id, language) %>%
  summarise(
    mean_gold_rank = mean(gold_rank, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  group_by(language) %>%
  summarise(
    mean_gold_rank = mean(mean_gold_rank, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(mean_gold_rank) %>%
  slice_head(n = 10) %>%
  ggplot(aes(
    x = reorder(language, mean_gold_rank),
    y = mean_gold_rank
  )) +
  geom_col(fill = "seagreen", width = 0.7) +
  geom_text(
    aes(label = round(mean_gold_rank, 2)),
    vjust = -0.4,
    size = 3.5
  ) +
  scale_y_continuous(
    limits = c(0, NA),
    expand = expansion(mult = c(0, 0.1))
  ) +
  labs(
    title = "10 idiomas com melhor posicionamento médio do arquivo correto.",
    subtitle = "Valores menores indicam que o arquivo correto apareceu mais próximo do topo do ranking.",
    x = "Idioma",
    y = "Posição média do arquivo correto"
  ) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

save_graph(g2, "g7_best_gold_rank_languages_all_experiments.png")
print("[7/9] Gráfico 7 salvo em src/results/graphs/g7_best_gold_rank_languages_all_experiments.png")

# ============================================================
# 3) Retrieval time by language
# ============================================================

g3 <- df %>%
  group_by(experiment_id, language, replication_id) %>%
  summarise(
    retrieval_time_us = mean(retrieval_time_s, na.rm = TRUE) * 1e6,
    .groups = "drop"
  ) %>%
  group_by(experiment_id, language) %>%
  summarise(
    retrieval_time_us = mean(retrieval_time_us, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  group_by(language) %>%
  summarise(
    mean_retrieval_time_us = mean(retrieval_time_us, na.rm = TRUE),
    sd_retrieval_time_us = sd(retrieval_time_us, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  ggplot(aes(
    x = reorder(language, mean_retrieval_time_us),
    y = mean_retrieval_time_us
  )) +
  geom_point(
    color = "steelblue",
    size = 3
  ) +
  geom_errorbar(
    aes(
      ymin = mean_retrieval_time_us - sd_retrieval_time_us,
      ymax = mean_retrieval_time_us + sd_retrieval_time_us
    ),
    color = "steelblue",
    width = 0.2
  ) +
  scale_y_continuous(
    limits = c(0, NA),
    expand = expansion(mult = c(0, 0.05))
  ) +
  labs(
    title = "Tempo médio de retrieval por idioma.",
    x = "Idioma",
    y = "Tempo médio de retrieval (µs)"
  ) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

save_graph(g3, "g8_mean_retrieval_time_by_language_all_experiments.png")
print("[8/9] Gráfico 8 salvo em src/results/graphs/g8_mean_retrieval_time_by_language_all_experiments.png")

# ============================================================
# 4) Retrievals with gold_found == FALSE by language
# ============================================================

g4 <- df %>%
  group_by(experiment_id, language, query_number) %>%
  summarise(
    falhas_nas_replicacoes = sum(gold_found == FALSE, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    erro_consolidado = falhas_nas_replicacoes > 5
  ) %>%
  group_by(language) %>%
  summarise(
    total_queries = n(),
    total_erros = sum(erro_consolidado, na.rm = TRUE),
    perc_erros = (total_erros / total_queries) * 100, 
    .groups = "drop"
  ) %>%
  filter(perc_erros > 0) %>%
  ggplot(aes(
    x = reorder(language, perc_erros),
    y = perc_erros
  )) +
  geom_col(fill = "indianred", width = 0.7) +
  geom_text(
    aes(label = sprintf("%.1f%%", perc_erros)),
    vjust = -0.4,
    size = 3.5
  ) +
  scale_y_continuous(
    limits = c(0, NA),
    expand = expansion(mult = c(0, 0.1)) 
  ) +
  labs(
    title = "Porcentagem de retrievals incorretos por idioma.",
    subtitle = "Considerado erro apenas se mais da metade das 10 replicações falharem.",
    x = "Idioma",
    y = "Porcentagem de erros (%)"
  ) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

save_graph(g4, "g9_gold_false_percentage_by_language_all_experiments.png")
print("[9/9] Gráfico 9 salvo em src/results/graphs/g9_gold_false_percentage_by_language_all_experiments.png")