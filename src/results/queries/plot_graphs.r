library(tidyverse)

df <- read_csv("src/results/queries/all_experiments_summary.csv")

save_graph <- function(plot, filename, width = 10, height = 6) {
  ggsave(
    filename = paste0("src/results/queries/graphs/", filename),
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
  filter(
    gold_found == TRUE
  ) %>%
  group_by(language) %>%
  summarise(
    mean_gold_rank = mean(gold_rank, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(desc(mean_gold_rank)) %>%
  slice_head(n = 10) %>%
  ggplot(aes(
    x = reorder(language, mean_gold_rank),
    y = mean_gold_rank
  )) +
  geom_col(fill = "firebrick", alpha = 0.8) +
  labs(
    title = "10 idiomas com pior posicionamento médio do arquivo correto (experimento 10)",
    subtitle = "Valores maiores indicam que o arquivo correto apareceu em posições mais baixas no ranking.",
    x = "Idioma",
    y = "Posição média do arquivo correto"
  ) +
  coord_flip()

save_graph(g1, "g1_worst_gold_rank_languages.png")

# ============================================================
# 2) Best gold_rank by language (considering only retrievals with gold_found == TRUE)
# ============================================================

g2 <- df %>%
  filter(
    gold_found == TRUE
  ) %>%
  group_by(language) %>%
  summarise(
    mean_gold_rank = mean(gold_rank, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(mean_gold_rank) %>%
  slice_head(n = 10) %>%
  ggplot(aes(
    x = reorder(language, -mean_gold_rank),
    y = mean_gold_rank
  )) +
  geom_col(fill = "forestgreen", alpha = 0.8) +
  labs(
    title = "10 idiomas com melhor posicionamento médio do arquivo correto",
    subtitle = "Valores menores indicam que o arquivo correto apareceu mais próximo do topo do ranking.",
    x = "Idioma",
    y = "Posição média do arquivo correto"
  ) +
  coord_flip()

save_graph(g2, "g2_best_gold_rank_languages.png")

# ============================================================
# 3) Retrieval time by language
# ============================================================

g3 <- df %>%
  group_by(language) %>%
  summarise(
    mean_retrieval_time_us = mean(retrieval_time_s, na.rm = TRUE) * 1e6,
    .groups = "drop"
  ) %>%
  ggplot(aes(
    x = reorder(language, mean_retrieval_time_us),
    y = mean_retrieval_time_us
  )) +
  geom_col(fill = "steelblue", alpha = 0.8) +
  scale_y_continuous(limits = c(0, NA)) +
  labs(
    title = "Tempo médio de retrieval por idioma (experimento 10)",
    x = "Idioma",
    y = "Tempo médio de retrieval (µs)"
  ) +
  coord_flip()

save_graph(g3, "g3_mean_retrieval_time_by_language.png")

# ============================================================
# 4) Retrievals with gold_found == FALSE by language
# ============================================================

g4 <- df %>%
  group_by(language) %>%
  summarise(
    total = n(),
    total_gold_false = sum(gold_found == FALSE, na.rm = TRUE),
    error_percentage = (total_gold_false / total) * 100,
    .groups = "drop"
  ) %>%
  filter(total_gold_false > 0) %>%
  ggplot(aes(
    x = reorder(language, error_percentage),
    y = error_percentage
  )) +
  geom_col(fill = "firebrick", alpha = 0.8) +
  labs(
    title = "Porcentagem de retrievals incorretos por idioma",
    subtitle = "Idiomas ausentes no gráfico recuperaram corretamente o arquivo esperado em todas as perguntas.",
    x = "Idioma",
    y = "Porcentagem de erros (%)"
  ) +
  coord_flip()

save_graph(g4, "g4_gold_false_percentage_by_language.png")
