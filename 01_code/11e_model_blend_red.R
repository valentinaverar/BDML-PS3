##########################################################
# 11e_model_blend_red.R
#
# Objetivo: mezclar el LightGBM híbrido de precio por m² (script 11b) con la
# red neuronal (script 12) y generar el envío a Kaggle.
#
# Etapa del Problem Set: Etapa 2 (combinación de familias) y Etapa 4.
# Insumos:  03_temp/train_model.rds, test_model.rds (script 05);
#           03_temp/similitud_chapinero.rds (script 06);
#           03_temp/oof_lgbm_m2.rds y el envío LGBM_completo_m2_lr_....csv
#           (script 11b); 03_temp/oof_nn.rds y el envío NN_completo_....csv
#           (script 12).
# Salidas:  02_outputs/predictions/BLEND_m2_red_w_....csv (envío).
#           02_outputs/tables/BLEND_m2_red_pesos.csv: MAE de cada peso.
#           02_outputs/figures/blend_red_pesos.png: evidencia del peso
#           elegido (MAE y sesgo según el peso de la red).
#           Filas en 02_outputs/tables/registro_modelos.csv.
#           03_temp/oof_blend_red.rds (predicciones fuera de muestra).
#
# Decisiones:
#   - Por qué mezclar: los dos modelos se equivocan en direcciones opuestas
#     en las viviendas parecidas a Chapinero. El LightGBM subestima (sesgo
#     de -29 millones) porque los árboles no extrapolan; la red casi no
#     tiene sesgo (+1 millón) porque su atajo lineal sí extrapola. En
#     Kaggle quedan cerca (169,2 y 180,0). El promedio de dos modelos
#     parecidos en precisión pero con errores distintos suele equivocarse
#     menos que cada uno.
#   - No se reentrena nada: se usan las predicciones fuera de muestra de la
#     CV espacial de los dos scripts (mismos folds) y sus envíos de test.
#   - Predicción = (1 - w) x LightGBM + w x red. El peso w se elige entre 0
#     y 1 (pasos de 0,05) con el menor MAE en las viviendas similares a
#     Chapinero (CV espacial), como en la mezcla del 11c. Como se elige con
#     el mismo grupo con el que se mide, ese MAE queda un poco optimista;
#     Kaggle da la medida limpia.
#   - Evidencia del peso elegido: además del MAE en las similares (el
#     criterio), se muestra el MAE en toda la CV espacial, que no se usó
#     para elegir. Si los dos tienen su mínimo cerca del mismo peso, la
#     mejora no es producto de haber elegido con el mismo grupo con el que
#     se mide.
#   - Si el mejor peso es 0 (la red no aporta), no se genera envío.
##########################################################

# Objetos que ya existían en el entorno antes de correr este script. Al
# final se borra solo lo que este script creó, sin tocar lo demás.
objetos_previos <- ls()

# Parámetros ---------------------------------------------------------------
grilla_pesos <- seq(0, 1, by = 0.05)

# Paquetes y funciones -----------------------------------------------------
if (!require(pacman)) install.packages("pacman")
pacman::p_load(dplyr, ggplot2)

source("01_code/functions/fun_save_submission.R")
source("01_code/functions/fun_metricas.R")

# Datos --------------------------------------------------------------------
train <- readRDS("03_temp/train_model.rds")
test <- readRDS("03_temp/test_model.rds")
similitud <- readRDS("03_temp/similitud_chapinero.rds")
oof_lgbm <- readRDS("03_temp/oof_lgbm_m2.rds")
oof_red <- readRDS("03_temp/oof_nn.rds")

for (oof in list(similitud, oof_lgbm, oof_red)) {
  if (!identical(oof$property_id, train$property_id)) {
    stop("Las predicciones fuera de muestra no están en el orden de ",
         "train_model.rds. Corra de nuevo los scripts 11b y 12.")
  }
}
y <- train$price
rango_precio <- range(y)
similares <- similitud$similitud >= stats::quantile(similitud$similitud, 0.9)

fun_leer_envio <- function(.patron) {
  #' Leer un envío de Kaggle y ordenarlo como test_model.rds
  archivo <- list.files("02_outputs/predictions", pattern = .patron,
                        full.names = TRUE)
  if (length(archivo) != 1) {
    stop("Se esperaba un único envío con el patrón ", .patron, " y hay ",
         length(archivo), ".")
  }
  envio <- utils::read.csv(archivo)
  precio <- envio$price[match(test$property_id, envio$property_id)]
  if (anyNA(precio)) stop("El envío ", archivo, " no cubre todo el test.")
  message("Envío leído: ", basename(archivo))
  precio
}
test_lgbm <- fun_leer_envio("^LGBM_completo_m2_lr_.*\\.csv$")
test_red <- fun_leer_envio("^NN_completo_.*\\.csv$")

fun_mezclar <- function(.lgbm, .red, .w) {
  #' Promedio ponderado, acotado al rango de precios de train
  pmin(pmax((1 - .w) * .lgbm + .w * .red, rango_precio[1]), rango_precio[2])
}

# 1. Peso de la red con la CV espacial -------------------------------------
fun_mae <- function(.pred, .filas = TRUE) mean(abs(.pred - y)[.filas]) / 1e6
fun_sesgo <- function(.pred, .filas = TRUE) mean((.pred - y)[.filas]) / 1e6
pesos <- do.call(rbind, lapply(grilla_pesos, function(w) {
  pred <- fun_mezclar(oof_lgbm$hibrida_espacial, oof_red$NN_completo_espacial, w)
  data.frame(peso_red = w,
             mae_similares = fun_mae(pred, similares),
             sesgo_similares = fun_sesgo(pred, similares),
             mae_toda_cv = fun_mae(pred))
}))
dir.create("02_outputs/tables", recursive = TRUE, showWarnings = FALSE)
write.csv(pesos, "02_outputs/tables/BLEND_m2_red_pesos.csv", row.names = FALSE)
w <- pesos$peso_red[which.min(pesos$mae_similares)]

message("MAE en millones (CV espacial) según el peso de la red:")
print(pesos[pesos$peso_red %in% unique(c(0, w, 0.25, 0.5, 1)), ],
      row.names = FALSE, digits = 4)
message("Peso elegido para la red: ", w, ".")

# Evidencia del peso elegido: gráfica para las slides ----------------------
w_toda_cv <- pesos$peso_red[which.min(pesos$mae_toda_cv)]
message("Peso con menor MAE en toda la CV espacial (no usado para elegir): ",
        w_toda_cv, ".")
grafica <- rbind(
  data.frame(peso_red = pesos$peso_red, valor = pesos$mae_similares,
             serie = "Similares a Chapinero (criterio)",
             panel = "Error absoluto medio (millones)"),
  data.frame(peso_red = pesos$peso_red, valor = pesos$mae_toda_cv,
             serie = "Toda la CV espacial",
             panel = "Error absoluto medio (millones)"),
  data.frame(peso_red = pesos$peso_red, valor = pesos$sesgo_similares,
             serie = "Similares a Chapinero (criterio)",
             panel = "Sesgo en similares (millones)")
)
grafica$panel <- factor(grafica$panel,
                        levels = c("Error absoluto medio (millones)",
                                   "Sesgo en similares (millones)"))
figura <- ggplot(grafica, aes(x = peso_red, y = valor, colour = serie)) +
  geom_hline(data = data.frame(panel = factor(
    "Sesgo en similares (millones)", levels = levels(grafica$panel)),
    y = 0), aes(yintercept = y), colour = "grey60", linewidth = 0.4) +
  geom_vline(xintercept = w, linetype = "dashed", colour = "grey30") +
  geom_line(linewidth = 0.9) +
  geom_point(size = 1.4) +
  facet_wrap(~ panel, scales = "free_y") +
  scale_x_continuous(breaks = seq(0, 1, by = 0.25),
                     labels = function(.x) paste0(100 * .x, "%")) +
  scale_colour_manual(values = c("Similares a Chapinero (criterio)" = "#1b6ca8",
                                 "Toda la CV espacial" = "#d1495b")) +
  labs(x = "Peso de la red neuronal en la mezcla (el resto es LightGBM)",
       y = NULL, colour = NULL,
       title = paste0("Mezclar ", 100 * w, "% red y ", 100 * (1 - w),
                      "% LightGBM minimiza el error"),
       subtitle = paste0("Validación cruzada espacial. Solo LightGBM (0%): ",
                         round(pesos$mae_similares[pesos$peso_red == 0], 1),
                         " millones en similares; mezcla elegida: ",
                         round(min(pesos$mae_similares), 1), " millones.")) +
  theme_minimal(base_size = 11) +
  theme(legend.position = "bottom", panel.grid.minor = element_blank(),
        plot.title = element_text(face = "bold"))
dir.create("02_outputs/figures", recursive = TRUE, showWarnings = FALSE)
ggsave("02_outputs/figures/blend_red_pesos.png", figura,
       width = 9, height = 4.5, dpi = 300, bg = "white")

# 2. Registro, predicciones fuera de muestra y envío -----------------------
if (w == 0) {
  message("La red no mejora al LightGBM en las similares: no se genera envío.")
} else {
  nombre <- paste0("BLEND_m2_red_w_", fun_hiperparam_str(w, 2))
  descripcion <- paste0("Mezcla (1-w) x LightGBM precio m2 (11b) + w x red ",
                        "neuronal (12); w=", w,
                        " elegido con la CV espacial en las similares")
  oof_blend_red <- data.frame(property_id = train$property_id)
  for (tipo in c("espacial", "aleatoria")) {
    pred <- fun_mezclar(oof_lgbm[[paste0("hibrida_", tipo)]],
                        oof_red[[paste0("NN_completo_", tipo)]], w)
    fun_registrar_cv(
      .archivo = nombre, .algoritmo = "Mezcla LGBM-red",
      .conjunto = "completo", .n_predictores = NA_integer_,
      .hiperparametros = descripcion, .tipo_cv = tipo, .obs = y, .pred = pred
    )
    oof_blend_red[[paste0("BLEND_red_", tipo)]] <- pred
  }
  saveRDS(oof_blend_red, "03_temp/oof_blend_red.rds")
  fun_save_submission(.ids = test$property_id,
                      .pred = fun_mezclar(test_lgbm, test_red, w),
                      .nombre = nombre)
}

# Limpiamos el entorno: borramos solo los objetos que creó este script.
rm(list = setdiff(ls(), objetos_previos))
