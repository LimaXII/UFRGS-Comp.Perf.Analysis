library(tidyverse)

df <- read_csv("src/results/embeddings/all_experiments_summary.csv")

model_df <- df %>%
  group_by(experiment_id, language, file_name) %>%
  summarise(
    file_chars = first(file_chars),
    chunk_size = first(chunk_size),
    language = first(language),
    batch_size = first(batch_size),
    chunks_count = first(chunks_count),
    embedding_time = mean(file_embedding_time_seconds, na.rm = TRUE),
    .groups = "drop"
  )

model <- lm(
  embedding_time ~ (file_chars + chunk_size + language)^2,
  data = model_df
)

summary(model)

grid <- expand_grid(
  file_chars = seq(min(model_df$file_chars), max(model_df$file_chars), length.out = 200),
  chunk_size = median(model_df$chunk_size),
  language = model_df$language[1]
)

grid$predicted_time <- predict(model, newdata = grid)

g <- ggplot(model_df, aes(x = file_chars)) +

  geom_point(aes(y = embedding_time), alpha = 0.4, size = 1) +

  geom_line(
    data = grid,
    aes(y = predicted_time),
    color = "red",
    linewidth = 1
  ) +

  labs(
    title = "Embedding Time × File Characters (modelo de regressão)",
    x = "File characters",
    y = "Embedding time (s)"
  ) +
  theme_minimal()

output_dir <- "src/results/graphs"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

ggsave(
  filename = file.path(output_dir, "embedding_time_vs_file_chars_model.png"),
  plot = g,
  width = 10,
  height = 6,
  dpi = 300
)