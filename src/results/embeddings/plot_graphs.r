library(tidyverse)

df <- read_csv("src/results/embeddings/all_experiments_summary.csv")

save_graph <- function(plot, filename, width = 10, height = 6) {
  ggsave(
    filename = paste0("src/results/embeddings/graphs/", filename),
    plot = plot,
    width = width,
    height = height
  )
}

## A) Embeddings - General Graphs - Experiments don't interfere.
## Experiment Fixed: Experiment 10.

# ============================================================
# 1) Character dispersion for 6 languages
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
    title = "Distribuição de caracteres por idioma (experimento 10)",
    x = "Idioma",
    y = "Número de caracteres"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

save_graph(g1, "g1_character_dispersion.png")

# ============================================================
# 2) Max file_chars per language
# ============================================================

g2 <- df %>%
  filter(experiment_id == 10) %>%
  group_by(language) %>%
  slice_max(order_by = file_chars, n = 1, with_ties = FALSE) %>%
  ungroup() %>%
  ggplot(aes(x = reorder(language, file_chars), y = file_chars)) +
  geom_col(fill = "steelblue", alpha = 0.8) +
  labs(
    title = "Maior número de caracteres por idioma (diretrizes.md, experimento 10)",
    x = "Idioma",
    y = "Número de caracteres"
  ) +
  coord_flip()

save_graph(g2, "g2_maxchars_by_language.png")

# ============================================================
# 3) Storage size per language
# ============================================================

g3 <- df %>%
  filter(experiment_id == 10) %>%
  distinct(language, total_storage_mb) %>%
  ggplot(aes(x = reorder(language, total_storage_mb), y = total_storage_mb)) +
  geom_col(fill = "steelblue", alpha = 0.8) +
  labs(
    title = "Tamanho do banco de dados por idioma (experimento 10)",
    x = "Idioma",
    y = "Tamanho total (MB)"
  ) +
  coord_flip()

save_graph(g3, "g3_storage_size.png")

# ============================================================
# 4) Embedding time per language
# ============================================================

g4 <- df %>%
  filter(experiment_id == 10) %>%
  distinct(language, embedding_time_total_seconds) %>%
  ggplot(aes(
    x = reorder(language, embedding_time_total_seconds),
    y = embedding_time_total_seconds
  )) +
  geom_col(fill = "steelblue", alpha = 0.8) +
  labs(
    title = "Tempo total de embedding por idioma (experimento 10)",
    x = "Idioma",
    y = "Tempo total (segundos)"
  ) +
  coord_flip()

save_graph(g4, "g4_embedding_time_by_language.png")

## B) Embeddings - Experiment Graphs
## Experiments: All.

# ============================================================
# 5) Average file size vs storage
# ============================================================

g5 <- df %>%
  group_by(experiment_id, chunk_size, language) %>%
  summarise(
    mean_file_chars = mean(file_chars, na.rm = TRUE),
    total_storage_mb = max(total_storage_mb),
    .groups = "drop"
  ) %>%
  ggplot(aes(
    x = mean_file_chars,
    y = total_storage_mb,
    color = as.factor(chunk_size)
  )) +
  geom_point(size = 4, alpha = 0.8) +
  labs(
    title = "Relação entre tamanho médio dos arquivos e armazenamento total",
    x = "Média de caracteres por idioma",
    y = "Armazenamento total (MB)",
    color = "Chunk size"
  )

save_graph(g5, "g5_avg_chars_vs_storage.png")

# ============================================================
# 6) CUDA vs CPU
# ============================================================

g6 <- df %>%
  filter(device %in% c("cpu", "cuda")) %>%
  group_by(device, batch_size) %>%
  summarise(
    mean_time = mean(embedding_time_total_seconds, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  ggplot(aes(x = batch_size, y = mean_time, color = device)) +
  geom_point(size = 4) +
  labs(
    title = "Tempo de embedding vs batch size (CPU vs CUDA)",
    subtitle = "CPU: Intel Core i5-12400F | GPU: NVIDIA GeForce RTX 4060",
    x = "Batch size",
    y = "Tempo médio (segundos)",
    color = "Device"
  ) +
  theme(
    plot.subtitle = element_text(size = 10, face = "italic")
  )

save_graph(g6, "g6_cuda_vs_cpu_embedding_time.png")