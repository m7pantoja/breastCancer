library(tidyverse)
library(ggplot2)

path <- "data/recognitionDataset"

# Obtención de las rutas de todas las imágenes del dataset
image_files <- list.files(path, recursive = TRUE, full.names = TRUE)

# Creación de un dataframe con la ruta y etiqueta de cada imagen 
df <- data.frame(
  path = image_files,
  label = basename(dirname(image_files)) 
) |> 
  mutate(label = recode(label,
                        "0_ausence" = "Ausencia",
                        "1_presence" = "Presencia"))

# Conversión de label a tipo factor
df$label <- as.factor(df$label)

# Análisis del balance de clases
class_counts <- df |>
  group_by(label) |>
  summarise(count = n()) |>
  ungroup() |>
  mutate(percentage = (count / sum(count)) * 100)

# Gráfico de barras del balance de clases
barplotClasses <- ggplot(class_counts, aes(x = label, y = count, fill = label)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = paste0(count, " (", round(percentage, 1), "%)")), vjust = -0.5) +
  labs(title = "Balance de Clases (Ausencia vs Presencia de Tumor)",
       x = "Clase", y = "Número de Imágenes") +
  theme_minimal() +
  theme(legend.position = "none")

plot(barplotClasses)