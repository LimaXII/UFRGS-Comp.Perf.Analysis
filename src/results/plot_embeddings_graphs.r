library(tidyverse)

df <- read_csv("src/results/embeddings/all_experiments_summary.csv")

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

# ============================================================
# 1) Character dispersion for 6 languages - Experiments don't interfere.
# Experiment Fixed: Experiment 10.
# ============================================================

langs <- c(
  "pt_br", "en_us", "ru_ru", "de_de", "nl_nl",
  "ja_jp", "zh_cn"
)

g1 <- df %>%
  filter(experiment_id == 10, language %in% langs) %>%
  ggplot(aes(
    x = reorder(language, file_chars, FUN = median),
    y = file_chars,
    fill = language
  )) +
  geom_boxplot(alpha = 0.7, outlier.alpha = 0.3) +
  labs(
    title = "Distribuição de caracteres por idioma",
    x = "Idioma",
    y = "Número de caracteres"
  ) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "none"
  )

save_graph(g1, "g1_character_dispersion.png")
print("[1/9] Gráfico 1 (embeddings) salvo em src/results/embeddings/graphs/g1_character_dispersion.png")

# ============================================================
# 2) Max file_chars per language  - Experiments don't interfere.
# Experiment Fixed: Experiment 10.
# ============================================================

g2 <- df %>%
  filter(experiment_id == 10) %>%
  group_by(language, replication_id) %>%
  summarise(
    max_file_chars = max(file_chars, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  group_by(language) %>%
  summarise(
    mean_max_file_chars = mean(max_file_chars, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  ggplot(aes(
    x = reorder(language, mean_max_file_chars),
    y = mean_max_file_chars
  )) +
  geom_col(fill = "steelblue") +
  labs(
    title = "Maior número de caracteres por idioma (diretrizes.md)",
    x = "Idioma",
    y = "Número de caracteres"
  ) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

save_graph(g2, "g2_maxchars_by_language.png")
print("[2/9] Gráfico 2 (embeddings) salvo em src/results/embeddings/graphs/g2_maxchars_by_language.png")

# ============================================================
# 3) Storage size per language - All Experiments
# ============================================================

g3 <- df %>%
  distinct(experiment_id, language, replication_id, total_storage_mb) %>%
  group_by(experiment_id, language, replication_id) %>%
  summarise(
    total_storage_mb = max(total_storage_mb, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  group_by(experiment_id, language) %>%
  summarise(
    exp_mean_storage = mean(total_storage_mb, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  group_by(language) %>%
  summarise(
    final_mean_storage = mean(exp_mean_storage, na.rm = TRUE),
    final_sd_storage = sd(exp_mean_storage, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  ggplot(aes(
    x = reorder(language, final_mean_storage),
    y = final_mean_storage
  )) +
  geom_errorbar(aes(
    ymin = final_mean_storage - final_sd_storage,
    ymax = final_mean_storage + final_sd_storage
  ), width = 0.2, color = "steelblue", alpha = 0.7) +
  geom_point(color = "steelblue", size = 3) +
  labs(
    title = "Tamanho do banco de dados por idioma",
    subtitle = "Média e Desvio Padrão entre todos os experimentos",
    x = "Idioma",
    y = "Tamanho total (MB)"
  ) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

save_graph(g3, "g3_storage_size_all.png")
print("[3/9] Gráfico 3 (embeddings) salvo em src/results/embeddings/graphs/g3_storage_size_all.png")
# ============================================================
# 4) Embedding time per language - All Experiments
# ============================================================

g4 <- df %>%
  distinct(experiment_id, language, replication_id, embedding_time_total_seconds) %>%
  group_by(experiment_id, language, replication_id) %>%
  summarise(
    max_time_seconds = max(embedding_time_total_seconds, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  group_by(experiment_id, language) %>%
  summarise(
    exp_mean_time = mean(max_time_seconds, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  group_by(language) %>%
  summarise(
    final_mean_time = mean(exp_mean_time, na.rm = TRUE),
    final_sd_time = sd(exp_mean_time, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  ggplot(aes(
    x = reorder(language, final_mean_time),
    y = final_mean_time
  )) +
  geom_point(size = 2, color = "steelblue") +
  geom_errorbar(
    aes(
      ymin = final_mean_time - final_sd_time,
      ymax = final_mean_time + final_sd_time
    ),
    width = 0.2,
    color = "steelblue"
  ) +
  scale_y_continuous(
    limits = c(0, NA),
    expand = expansion(mult = c(0, 0.05))
  ) +
  labs(
    title = "Tempo total de embedding por idioma (todos os experimentos)",
    x = "Idioma",
    y = "Tempo total (segundos)"
  ) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "none"
  )

save_graph(g4, "g4_embedding_time_by_language_all.png")
print("[4/9] Gráfico 4 (embeddings) salvo em src/results/embeddings/graphs/g4_embedding_time_by_language_all.png")

## B) Embeddings - Experiment Graphs
## Experiments: All.

# ============================================================
# 5) Average file size vs storage - Experiments don't interfere.
# ============================================================

g5 <- df %>%
  group_by(experiment_id, chunk_size, language, replication_id) %>%
  summarise(
    mean_file_chars = mean(file_chars, na.rm = TRUE),
    total_storage_mb = max(total_storage_mb, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  group_by(experiment_id, chunk_size, language) %>%
  summarise(
    mean_file_chars = mean(mean_file_chars, na.rm = TRUE),
    total_storage_mb = mean(total_storage_mb, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  ggplot(aes(
    x = mean_file_chars,
    y = total_storage_mb,
    color = as.factor(chunk_size)
  )) +
  geom_point(size = 4, alpha = 0.8) +
  scale_x_continuous(
    limits = c(0, NA),
    expand = expansion(mult = c(0, 0.05))
  ) +
  scale_y_continuous(
    limits = c(0, NA),
    expand = expansion(mult = c(0, 0.05))
  ) +
  labs(
    title = "Relação entre tamanho médio dos arquivos e armazenamento total",
    x = "Média de caracteres por idioma",
    y = "Armazenamento total (MB)",
    color = "Chunk size"
  )

save_graph(g5, "g5_avg_chars_vs_storage.png")
print("[5/9] Gráfico 5 (embeddings) salvo em src/results/embeddings/graphs/g5_avg_chars_vs_storage.png")