library(tidyverse)

df <- read_csv("src/results/embeddings/all_experiments_summary.csv")

# ============================================================
# 1) chunks_count dado o chunk_size (pontos)
# ============================================================

g1 <- df %>%
  ggplot(aes(x = chunk_size, y = chunks_count)) +
  geom_point(size = 2, alpha = 0.5) +
  labs(
    title = "Número de chunks em função do tamanho do chunk",
    x = "Tamanho do chunk",
    y = "Número de chunks"
  )
ggsave(
  "src/results/embeddings/graphs/g1_chunks_vs_chunk_size.png", 
  g1, 
  width = 8, 
  height = 5
)

# ============================================================
# 2) embedding_time_mean_seconds por idioma (pontos)
# ============================================================

# Escolher experimento de referência
exp_ref <- df$experiment_id[2]

g2 <- df %>%
  filter(experiment_id == exp_ref) %>%
  group_by(language) %>%
  ggplot(aes(x = language, y = embedding_time_mean_seconds)) +
  geom_point(size = 2) +
  labs(
    title = paste("Tempo médio de embedding por idioma (Exp:", exp_ref, ")"),
    x = "Idioma",
    y = "Tempo médio (s)"
  )

ggsave("src/results/embeddings/graphs/g2_embedding_time_by_language.png", g2, width = 8, height = 5)

# ============================================================
# 3) embedding_time_mean_seconds por device (CPU vs CUDA)
# ============================================================

g3 <- df %>%
  group_by(device) %>%
  summarise(
    embedding_time_mean_seconds = mean(embedding_time_mean_seconds, na.rm = TRUE)
  ) %>%
  ggplot(aes(x = device, y = embedding_time_mean_seconds)) +
  geom_point(size = 4) +
  labs(
    title = "Tempo médio de embedding por dispositivo",
    x = "Device",
    y = "Tempo médio (s)"
  ) +
  theme_minimal()

ggsave("src/results/embeddings/graphs/g3_embedding_time_by_device.png", g3, width = 8, height = 5)