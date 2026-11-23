############################################################
# PS3 - Algorithm Comparison | Equipo 03
# Insumos: resultados YA estimados; no vuelve a entrenar.
# Ejecutar desde la raiz: source("01_code/14_compare_algorithms.R")
# Salidas: 02_outputs/figures/compare_*.pdf,
#          02_outputs/tables/compare_metricas.tex
############################################################

objetos_previos <- ls()

if (!require(pacman)) install.packages("pacman")
pacman::p_load(ggplot2, knitr)

dir.create("02_outputs/figures", recursive = TRUE, showWarnings = FALSE)
dir.create("02_outputs/tables", recursive = TRUE, showWarnings = FALSE)

kaggle <- read.csv("00_data/kaggle_scores.csv", fileEncoding = "UTF-8-BOM")
registro <- read.csv("02_outputs/tables/registro_modelos.csv", fileEncoding = "UTF-8")
pesos <- read.csv("02_outputs/tables/BLEND_m2_red_pesos.csv")
seleccion <- read.csv("02_outputs/tables/seleccion_cv.csv")
conjuntos <- read.csv("02_outputs/tables/conjuntos_predictores.csv")

# Coincidencia por nombre de envio: cada score viene de Kaggle.
score <- function(patron) {
  x <- kaggle[grepl(patron, kaggle$archivo), ]
  x <- x[which.min(x$puntaje_publico), ]
  c(archivo = x$archivo, mae = as.character(x$puntaje_publico / 1e6))
}
p <- list(
  lm_base = score("^LM_base$"),
  lm = score("^LM_completo$"),
  en = score("^EN_completo_a_"),
  en_tfidf = score("^EN_ampliado_"),
  cart = score("^CART_"),
  rf = score("^RF_"),
  lgbm = score("^LGBM_completo_lr_0_017_"),
  lgbm_tfidf = score("^LGBM_ampliado_"),
  m2 = score("^LGBM_completo_m2_lr_"),
  vecinos = score("^LGBM_completo_m2_vecinos_k10_lr_"),
  dmin200 = score("^LGBM_completo_m2_vecinos_k10_dmin200_"),
  red = score("^NN_"),
  blend_m2 = score("^BLEND_m2_red_"),
  blend = score("^BLEND_vecinos_red_")
)
valor <- function(nombre) as.numeric(p[[nombre]]["mae"])
id <- function(nombre) unname(p[[nombre]]["archivo"])
cv <- function(nombre, tipo) {
  x <- registro[registro$archivo == id(nombre) & registro$tipo_cv == tipo, ]
  if (nrow(x)) x$mae[1] / 1e6 else NA_real_
}
fmt <- function(x, dig = 2) formatC(x, digits = dig, format = "f", decimal.mark = ",")
metricas <- c(
  MAEBlend = valor("blend"), MAEVecinos = valor("vecinos"),
  MAEMdos = valor("m2"), MAELGBM = valor("lgbm"),
  MAERF = valor("rf"), MAECART = valor("cart"),
  MAERed = valor("red"), MAEEN = valor("en_tfidf"),
  MAELM = valor("lm"), MAEBase = valor("lm_base"),
  MAETFIDFLGBM = valor("lgbm_tfidf"), MAEDmin = valor("dmin200"),
  GainBlend = valor("vecinos") - valor("blend"),
  GainTotal = valor("lgbm") - valor("blend"),
  GainMdos = valor("lgbm") - valor("m2"),
  GainVecinos = valor("m2") - valor("vecinos"),
  GainMCO = valor("lm_base") - valor("lm"),
  CVBlendAleatoria = cv("blend", "aleatoria"),
  CVBlendEspacial = cv("blend", "espacial"),
  CVGapEspacial = seleccion$brecha_absoluta_pct[seleccion$elegido],
  CVGapAleatoria = seleccion$brecha_absoluta_pct[seleccion$esquema == "Aleatoria"],
  CVGapBloques = seleccion$brecha_absoluta_pct[seleccion$esquema == "Bloques de 700 m"]
)
numeros <- c(NBase = conjuntos$Base[conjuntos$Bloque == "Total"],
             NCompleto = conjuntos$Completo[conjuntos$Bloque == "Total"],
             NAmpliado = conjuntos$Ampliado[conjuntos$Bloque == "Total"])
macros <- c(
  sprintf("\\newcommand{\\%s}{%s}", names(metricas), fmt(metricas)),
  sprintf("\\newcommand{\\%s}{%s}", names(numeros),
          format(numeros, big.mark = ".", scientific = FALSE, trim = TRUE))
)
writeLines(macros, "02_outputs/tables/compare_metricas.tex", useBytes = TRUE)

azul <- "#2F5FA8"
naranja <- "#D98E51"
verde <- "#268D77"
gris <- "#77828A"
tema <- theme_minimal(base_family = "sans", base_size = 12) +
  theme(plot.title = element_text(face = "bold", size = 12, color = "#263640"),
        plot.subtitle = element_text(size = 10),
        panel.grid.minor = element_blank(),
        panel.grid.major.y = element_blank(),
        axis.title.y = element_blank(),
        legend.position = "top",
        plot.margin = margin(8, 10, 8, 6))
fig <- function(nombre, grafico, ancho = 8.5, alto = 4.1) {
  ggsave(file.path("02_outputs/figures", paste0(nombre, ".pdf")),
         grafico, width = ancho, height = alto, units = "in", device = "pdf")
}

# 5. Comparacion orientativa dentro del conjunto completo.
df <- data.frame(modelo = c("Random Forest", "LightGBM", "Red neuronal",
                            "CART", "Elastic Net", "Regresion lineal"),
                 mae = c(valor("rf"), valor("lgbm"), valor("red"),
                         valor("cart"), valor("en"), valor("lm")))
df$modelo <- factor(df$modelo, levels = rev(df$modelo))
g <- ggplot(df, aes(modelo, mae)) + geom_col(fill = azul, width = .64) +
  geom_text(aes(label = sprintf("%.1f", mae)), hjust = -.10, size = 3.4) +
  coord_flip(clip = "off") + expand_limits(y = 245) +
  labs(x = NULL, y = "MAE publico (millones COP)") + tema
fig("compare_01_familias", g)

# 6. Incremento de la informacion: no siempre mejora.
df <- data.frame(modelo = c("Lineal: base", "Lineal: completo",
                            "Elastic Net: completo", "Elastic Net: + TF-IDF",
                            "LightGBM: completo", "LightGBM: + TF-IDF"),
                 mae = c(valor("lm_base"), valor("lm"), valor("en"),
                         valor("en_tfidf"), valor("lgbm"), valor("lgbm_tfidf")))
df$modelo <- factor(df$modelo, levels = rev(df$modelo))
g <- ggplot(df, aes(modelo, mae)) + geom_col(fill = azul, width = .65) +
  geom_text(aes(label = sprintf("%.1f", mae)), hjust = -.1, size = 3.3) +
  coord_flip(clip = "off") + expand_limits(y = 325) +
  labs(y = "MAE publico (millones COP)") + tema
fig("compare_02_predictores", g)

# 7. Secuencia descriptiva de desarrollo de LightGBM.
df <- data.frame(paso = c("LightGBM completo", "Precio por m2",
                          "+ precio de vecinos", "+ 25% red neuronal"),
                 mae = c(valor("lgbm"), valor("m2"),
                         valor("vecinos"), valor("blend")))
df$paso <- factor(df$paso, levels = df$paso)
g <- ggplot(df, aes(paso, mae, group = 1)) +
  geom_line(color = azul, linewidth = .85) +
  geom_point(color = azul, size = 3) +
  geom_text(aes(label = sprintf("%.2f M", mae)), vjust = -1.1, size = 3.7) +
  coord_cartesian(ylim = c(160, 185), clip = "off") +
  labs(y = "MAE publico (millones COP)", x = NULL) +
  theme_minimal(base_family = "sans", base_size = 11) +
  theme(panel.grid.minor = element_blank(), axis.text.x = element_text(size = 9),
        plot.margin = margin(15, 15, 10, 10))
fig("compare_03_evolucion", g, 9, 3.7)

# 8. Distancia minima a los vecinos (prueba con dmin = 200 m).
df <- data.frame(modelo = c("Hibrido sin vecinos", "Con vecinos (k=10)",
                            "Con vecinos (min. 200 m)"),
                 mae = c(valor("m2"), valor("vecinos"), valor("dmin200")))
df$modelo <- factor(df$modelo, levels = rev(df$modelo))
g <- ggplot(df, aes(modelo, mae)) + geom_col(fill = azul, width = .58) +
  geom_text(aes(label = sprintf("%.2f M", mae)), hjust = -.08, size = 4) +
  coord_flip(clip = "off") + expand_limits(y = 205) +
  labs(y = "MAE publico de Kaggle") + tema
fig("compare_04_vecinos", g, 8, 3.4)

# 9. Experimento de blending sin vecinos: no confundir con validacion del BLEND final.
curvas <- rbind(data.frame(peso = 100 * pesos$peso_red,
                           mae = pesos$mae_similares,
                           grupo = "Similares a Chapinero"),
                data.frame(peso = 100 * pesos$peso_red,
                           mae = pesos$mae_toda_cv,
                           grupo = "Toda la CV espacial"))
g <- ggplot(curvas, aes(peso, mae, color = grupo)) +
  geom_line(linewidth = .95) + geom_point(size = 1.2) +
  geom_vline(xintercept = 25, linetype = "dashed", color = gris) +
  scale_color_manual(values = c("Similares a Chapinero" = azul,
                                "Toda la CV espacial" = naranja)) +
  labs(x = "Peso de la red neuronal (%)", y = "MAE (millones COP)", color = NULL) +
  tema + theme(panel.grid.major.y = element_line(color = "#ECECEC"))
fig("compare_05_peso_red", g, 8.2, 4.2)

# 10. CV aleatoria vs espacial vs Kaggle para 5 modelos registrados.
cv_nombres <- c("lgbm", "m2", "red", "vecinos", "blend")
etiquetas <- c("LightGBM completo", "Hibrido por m2", "Red neuronal",
               "LightGBM + vecinos", "BLEND vecinos + red")
df <- do.call(rbind, lapply(seq_along(cv_nombres), function(i) {
  nombre <- cv_nombres[i]
  data.frame(modelo = etiquetas[i],
             tipo = c("CV aleatoria", "CV espacial", "Kaggle publico"),
             mae = c(cv(nombre, "aleatoria"), cv(nombre, "espacial"), valor(nombre)))
}))
df$modelo <- factor(df$modelo, levels = rev(etiquetas))
g <- ggplot(df, aes(mae, modelo, color = tipo)) +
  geom_line(aes(group = modelo), color = "#CCCCCC", linewidth = .6) +
  geom_point(size = 2.7) +
  scale_color_manual(values = c("CV aleatoria" = gris, "CV espacial" = azul,
                                "Kaggle publico" = naranja)) +
  coord_cartesian(xlim = c(80, 220)) +
  labs(x = "MAE (millones COP)", y = NULL, color = NULL) + tema +
  theme(panel.grid.major.y = element_line(color = "#EEEEEE"))
fig("compare_06_validacion", g, 8.2, 4.1)

# 11. Criterio de seleccion de CV: brecha absoluta promedio.
df <- subset(seleccion, esquema %in% c("k-means 20 regiones", "Aleatoria"))
df$esquema <- factor(df$esquema, levels = c("Aleatoria", "k-means 20 regiones"))
g <- ggplot(df, aes(esquema, brecha_absoluta_pct)) +
  geom_col(fill = azul, width = .55) +
  geom_text(aes(label = paste0(fmt(brecha_absoluta_pct, 1), "%")),
            hjust = -.1, size = 4) +
  coord_flip(clip = "off") + expand_limits(y = 25) +
  labs(x = NULL, y = "Brecha absoluta media en pseudo-Chapineros (%)") + tema
fig("compare_07_cv_esquemas", g, 8, 3.4)

message("Listo: mejor modelo = BLEND vecinos + red | MAE publico = ",
        fmt(valor("blend")), " millones COP. Se generaron 7 figuras y metricas.tex.")


# Limpiamos el entorno: borramos solo los objetos que creó este script.
rm(list = setdiff(ls(), objetos_previos))