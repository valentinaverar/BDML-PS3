##########################################################
# 07_model_linear_regression.R
#
# Objetivo: estimar el modelo de referencia (regresión lineal por MCO) con
# los conjuntos base y completo, medir su error con CV espacial y
# aleatoria, y generar sus envíos a Kaggle.
#
# Etapa del Problem Set: Etapa 2 (familia Linear Regression) y Etapa 3
# (CV espacial frente a aleatoria).
# Insumos:  03_temp/train_model.rds, test_model.rds,
#           conjuntos_predictores.rds (script 05) y cv_folds.rds (script 06).
# Salidas:  02_outputs/predictions/LM_base.csv y LM_completo.csv (envíos).
#           Filas en 02_outputs/tables/registro_modelos.csv (una por
#           variante y tipo de CV).
#           03_temp/oof_lm.rds: predicciones fuera de muestra de cada
#           variante y tipo de CV (para el análisis de errores, Etapa 4).
#
# Decisiones:
#   - Se modela log(precio): el precio es asimétrico y los efectos de las
#     características son más parecidos a porcentajes que a montos fijos.
#     La predicción se devuelve a pesos con exp(), que aproxima la mediana
#     condicional, coherente con una métrica de error absoluto (MAE).
#   - Las predicciones se acotan al rango de precios de train (300 a 1.650
#     millones): la muestra viene truncada en ese rango, así que un precio
#     por fuera de él es con seguridad un error.
#   - No hay hiperparámetros: la CV solo mide el error esperado. Es la
#     referencia contra la que se comparan las demás familias.
##########################################################

# Objetos que ya existían en el entorno antes de correr este script. Al
# final se borra solo lo que este script creó, sin tocar lo demás.
objetos_previos <- ls()

# Paquetes y funciones -----------------------------------------------------
if (!require(pacman)) install.packages("pacman")
pacman::p_load(dplyr)

source("01_code/functions/fun_save_submission.R")
source("01_code/functions/fun_metricas.R")

# Datos --------------------------------------------------------------------
train <- readRDS("03_temp/train_model.rds")
test <- readRDS("03_temp/test_model.rds")
conjuntos_predictores <- readRDS("03_temp/conjuntos_predictores.rds")
cv_folds <- readRDS("03_temp/cv_folds.rds")

if (!all(sort(unlist(cv_folds$espacial)) == seq_len(nrow(train)))) {
  stop("Los folds no coinciden con train_model.rds. Corra de nuevo los ",
       "scripts 05 y 06.")
}
rango_precio <- range(train$price)

# Función de ajuste y predicción --------------------------------------------
fun_lm_predecir <- function(.vars, .entrenar, .nuevos, .rango) {
  #' Ajustar MCO sobre log(precio) y predecir en pesos
  #'
  #' @param .vars \code{character}. Nombres de los predictores.
  #' @param .entrenar \code{data.frame}. Datos para estimar.
  #' @param .nuevos \code{data.frame}. Datos para predecir.
  #' @param .rango \code{numeric}. Precio mínimo y máximo de train.
  #'
  #' @returns \code{numeric}: precio predicho en pesos, acotado a .rango.
  formula <- reformulate(.vars, response = "log(price)")
  modelo <- lm(formula, data = .entrenar)
  # Si una raíz no aparece en un fold, su coeficiente es NA y predict()
  # avisa; esa variable simplemente no aporta en ese fold.
  pred <- exp(suppressWarnings(predict(modelo, newdata = .nuevos)))
  pmin(pmax(pred, .rango[1]), .rango[2])
}

# Estimación, validación cruzada y envíos ----------------------------------
variantes <- c(base = "LM_base", completo = "LM_completo")
oof_lm <- data.frame(property_id = train$property_id)

for (conjunto in names(variantes)) {
  nombre <- variantes[[conjunto]]
  vars <- conjuntos_predictores[[conjunto]]
  
  # Error fuera de muestra con cada tipo de CV.
  for (tipo in c("espacial", "aleatoria")) {
    oof <- rep(NA_real_, nrow(train))
    for (val in cv_folds[[tipo]]) {
      oof[val] <- fun_lm_predecir(vars, train[-val, ], train[val, ],
                                  rango_precio)
    }
    fun_registrar_cv(
      .archivo = nombre, .algoritmo = "Regresión lineal",
      .conjunto = conjunto, .n_predictores = length(vars),
      .hiperparametros = "MCO sobre log(precio)", .tipo_cv = tipo,
      .obs = train$price, .pred = oof
    )
    oof_lm[[paste0(nombre, "_", tipo)]] <- oof
    message(nombre, " | CV ", tipo, ": MAE = ",
            round(mean(abs(oof - train$price)) / 1e6, 1), " millones.")
  }
  
  # Modelo final con todo train y envío a Kaggle.
  pred_test <- fun_lm_predecir(vars, train, test, rango_precio)
  fun_save_submission(.ids = test$property_id, .pred = pred_test,
                      .nombre = nombre)
}

saveRDS(oof_lm, "03_temp/oof_lm.rds")

# Resumen ------------------------------------------------------------------
read.csv("02_outputs/tables/registro_modelos.csv") |>
  filter(algoritmo == "Regresión lineal") |>
  transmute(archivo, tipo_cv,
            mae_millones = round(mae / 1e6, 1),
            mae_similares_chapinero = round(mae_similares_chapinero / 1e6, 1),
            sesgo_millones = round(sesgo_medio / 1e6, 1),
            sesgo_similares_chapinero = round(sesgo_similares_chapinero / 1e6, 1),
            pct_sobreestima = round(pct_sobreestima, 1)) |>
  print(row.names = FALSE)

# Limpiamos el entorno: borramos solo los objetos que creó este script.
rm(list = setdiff(ls(), objetos_previos))
