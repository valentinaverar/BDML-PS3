##########################################################
# 13_model_boosting_precio_m2.R
#
# Objetivo: LightGBM que aprende el precio por metro cuadrado (y luego lo
# multiplica por la superficie) en lugar del precio total, para predecir
# mejor en Chapinero. Genera su envío a Kaggle.
#
# Etapa del Problem Set: Etapa 2 (familia Boosting) y Etapa 4 (por qué el
# mejor modelo falla en Chapinero y cómo corregirlo).
# Insumos:  03_temp/train_model.rds, test_model.rds,
#           conjuntos_predictores.rds (script 05); 03_temp/cv_folds.rds y
#           similitud_chapinero.rds (script 06);
#           02_outputs/tables/LGBM_completo_ensayos.csv (script 11).
# Salidas:  02_outputs/predictions/LGBM_completo_m2_....csv (envío).
#           Filas en 02_outputs/tables/registro_modelos.csv.
#           03_temp/oof_lgbm_m2.rds (predicciones fuera de muestra).
#
# Decisiones:
#   - Problema que se ataca: un árbol predice la mediana de las viviendas
#     de train que caen en su misma hoja, así que no extrapola. Una
#     vivienda grande y cara de Chapinero cae en hojas con viviendas más
#     pequeñas o más baratas y el modelo se queda corto (sesgo de -38
#     millones en las similares a Chapinero). El precio por m² se parece
#     más entre viviendas grandes y pequeñas de una misma zona, y al
#     multiplicarlo por la superficie la predicción sí escala con el
#     tamaño. Así se tasa la vivienda en la práctica.
#   - El precio por m² solo tiene sentido si la superficie es real. Se usa
#     la superficie construida (o, si falta, la total), de Kaggle o de la
#     descripción, y NO la imputada con la mediana (indicadores falta_ del
#     script 05). Por eso el modelo final es híbrido:
#       * viviendas con superficie conocida: precio por m² predicho x
#         superficie;
#       * viviendas sin superficie conocida: el LightGBM de precio total
#         del script 11 (mismos hiperparámetros y rondas).
#   - El modelo de precio por m² se entrena solo con las viviendas de train
#     con superficie conocida; usa los mismos predictores y los mismos
#     hiperparámetros del mejor ensayo del 11, y elige sus rondas con la
#     CV espacial.
#   - Se probó también calcular el precio por m² para todas las viviendas,
#     incluida la superficie imputada, con un solo modelo: empeora el error
#     en las similares a Chapinero (134,4 frente a 127,6 del híbrido),
#     porque mezcla precios por m² reales con otros inventados.
#   - Se compara contra el LightGBM de precio total (script 11) con las
#     mismas particiones, en toda la CV, en las similares a Chapinero y solo
#     en las viviendas con superficie conocida (donde actúa el cambio).
#     Resultado (CV espacial, similares a Chapinero): 127,6 millones frente
#     a 131,4 del LightGBM de precio total; el sesgo baja de -38 a -29.
##########################################################

# Objetos que ya existían en el entorno antes de correr este script. Al
# final se borra solo lo que este script creó, sin tocar lo demás.
objetos_previos <- ls()

# Parámetros ---------------------------------------------------------------
semilla <- 2026
hilos <- 6L
max_rondas <- 10000L
paciencia <- 100L

# Paquetes y funciones -----------------------------------------------------
if (!require(pacman)) install.packages("pacman")
pacman::p_load(dplyr, lightgbm)

source("01_code/functions/fun_save_submission.R")
source("01_code/functions/fun_metricas.R")

# Datos --------------------------------------------------------------------
train <- readRDS("03_temp/train_model.rds")
test <- readRDS("03_temp/test_model.rds")
conjuntos_predictores <- readRDS("03_temp/conjuntos_predictores.rds")
cv_folds <- readRDS("03_temp/cv_folds.rds")
similitud <- readRDS("03_temp/similitud_chapinero.rds")

if (!identical(similitud$property_id, train$property_id)) {
  stop("La similitud no está en el mismo orden que train_model.rds. ",
       "Corra de nuevo los scripts 05 y 06.")
}
if (!file.exists("02_outputs/tables/LGBM_completo_ensayos.csv")) {
  stop("Falta la tabla de ensayos del LightGBM. Corra primero el script 11.")
}
y <- train$price
rango_precio <- range(y)
similares <- similitud$similitud >= stats::quantile(similitud$similitud, 0.9)

# Matriz del conjunto completo (la misma del script 11).
fun_matriz <- function(.datos, .vars) {
  x <- .datos[.vars]
  x$property_type <- as.integer(x$property_type == "Casa")
  as.matrix(as.data.frame(lapply(x, as.numeric)))
}
x_train <- fun_matriz(train, conjuntos_predictores$completo)
x_test <- fun_matriz(test, conjuntos_predictores$completo)

# Superficie conocida (no imputada) ------------------------------------------
fun_superficie <- function(.datos) {
  #' Superficie real de cada vivienda, o NA si fue imputada
  #'
  #' @description Usa la superficie construida y, si falta, la total (las
  #' versiones "_completa" del script 05, que ya incluyen la superficie
  #' tomada de la descripción). Si las dos fueron imputadas con la mediana
  #' (indicador falta_ igual a 1), devuelve NA.
  #'
  #' @returns \code{numeric}: superficie en m², o NA.
  fun_conocida <- function(.v) {
    indicador <- paste0("falta_", .v)
    if (indicador %in% names(.datos)) .datos[[indicador]] == 0 else
      rep(TRUE, nrow(.datos))
  }
  dplyr::case_when(
    fun_conocida("surface_covered_completa") ~ .datos$surface_covered_completa,
    fun_conocida("surface_total_completa") ~ .datos$surface_total_completa,
    TRUE ~ NA_real_
  )
}
superficie_train <- fun_superficie(train)
superficie_test <- fun_superficie(test)
conocida_train <- !is.na(superficie_train)
conocida_test <- !is.na(superficie_test)
precio_m2 <- y / superficie_train
message("Superficie conocida: ", round(100 * mean(conocida_train), 1),
        "% de train, ", round(100 * mean(conocida_test), 1),
        "% de test y ", round(100 * mean(conocida_train[similares]), 1),
        "% de las similares a Chapinero.")
message("Precio por m² en train (millones): mediana ",
        round(stats::median(precio_m2, na.rm = TRUE) / 1e6, 2),
        "; percentiles 1 y 99: ",
        paste(round(stats::quantile(precio_m2, c(0.01, 0.99), na.rm = TRUE) /
                      1e6, 2), collapse = " y "), ".")

# Hiperparámetros del mejor ensayo del script 11 ----------------------------
fun_armar_params <- function(log_lr, num_leaves, max_depth, min_data_in_leaf,
                             feature_fraction, bagging_fraction, log_l1,
                             log_l2) {
  #' Convertir un ensayo en parámetros de LightGBM (igual que en el 11)
  profundidad <- as.integer(round(max_depth))
  list(
    objective = "regression_l1", metric = "l1",
    learning_rate = exp(log_lr),
    num_leaves = as.integer(min(round(num_leaves), 2^profundidad)),
    max_depth = profundidad,
    min_data_in_leaf = as.integer(round(min_data_in_leaf)),
    feature_fraction = feature_fraction,
    bagging_fraction = bagging_fraction, bagging_freq = 1L,
    lambda_l1 = expm1(log_l1), lambda_l2 = expm1(log_l2),
    num_threads = hilos, seed = semilla, verbosity = -1L
  )
}
ensayos <- read.csv("02_outputs/tables/LGBM_completo_ensayos.csv")
mejor <- ensayos[which.min(ensayos$mae), ]
params <- do.call(fun_armar_params, as.list(
  mejor[c("log_lr", "num_leaves", "max_depth", "min_data_in_leaf",
          "feature_fraction", "bagging_fraction", "log_l1", "log_l2")]))
rondas_precio <- as.integer(mejor$rondas)

# Rondas del modelo de precio por m² (CV espacial, solo superficie conocida).
filas_m2 <- which(conocida_train)
folds_m2 <- lapply(cv_folds$espacial,
                   function(.val) match(intersect(.val, filas_m2), filas_m2))
cv_m2 <- lightgbm::lgb.cv(
  params = params,
  data = lightgbm::lgb.Dataset(x_train[filas_m2, , drop = FALSE],
                               label = precio_m2[filas_m2]),
  folds = folds_m2, nrounds = max_rondas,
  early_stopping_rounds = paciencia, verbose = -1L
)
rondas_m2 <- as.integer(cv_m2$best_iter)
message("Rondas: precio total = ", rondas_precio, "; precio por m² = ",
        rondas_m2, ".")

# Funciones ----------------------------------------------------------------
fun_entrenar <- function(.filas, .y, .rondas) {
  #' Entrenar LightGBM con las filas indicadas de train
  lightgbm::lgb.train(
    params = params,
    data = lightgbm::lgb.Dataset(x_train[.filas, , drop = FALSE],
                                 label = .y[.filas]),
    nrounds = .rondas, verbose = -1L
  )
}
fun_acotar <- function(.pred) pmin(pmax(.pred, rango_precio[1]), rango_precio[2])

fun_predecir <- function(.filas_ajuste, .x_nuevo, .superficie_nueva) {
  #' Predicción de precio total y predicción híbrida (precio por m²)
  #'
  #' @param .filas_ajuste \code{integer}. Filas de train para entrenar.
  #' @param .x_nuevo Matriz de predictores donde se predice.
  #' @param .superficie_nueva \code{numeric}. Superficie conocida (o NA).
  #'
  #' @returns \code{list} con \code{precio} (modelo de precio total, como
  #' referencia) e \code{hibrida} (precio por m² x superficie donde se
  #' conoce; precio total donde no), en pesos y acotadas.
  modelo_precio <- fun_entrenar(.filas_ajuste, y, rondas_precio)
  modelo_m2 <- fun_entrenar(intersect(.filas_ajuste, filas_m2), precio_m2,
                            rondas_m2)
  pred_precio <- fun_acotar(stats::predict(modelo_precio, .x_nuevo))
  pred_m2 <- fun_acotar(stats::predict(modelo_m2, .x_nuevo) *
                          .superficie_nueva)
  list(precio = pred_precio,
       hibrida = ifelse(is.na(.superficie_nueva), pred_precio, pred_m2))
}

# Validación cruzada -------------------------------------------------------
nombre <- paste0("LGBM_completo_m2_lr_",
                 fun_hiperparam_str(params$learning_rate, 3),
                 "_hojas_", params$num_leaves, "_rondas_", rondas_m2)
descripcion <- paste0("Híbrido: precio por m2 x superficie (superficie ",
                      "conocida) y precio total (superficie imputada); ",
                      "hiperparámetros del 11; rondas m2=", rondas_m2,
                      "; rondas precio=", rondas_precio)
oof_m2 <- data.frame(property_id = train$property_id)
comparacion <- list()

for (tipo in c("espacial", "aleatoria")) {
  oof <- list(precio = rep(NA_real_, nrow(train)),
              hibrida = rep(NA_real_, nrow(train)))
  for (i in seq_along(cv_folds[[tipo]])) {
    val <- cv_folds[[tipo]][[i]]
    message("CV ", tipo, ": fold ", i, "/", length(cv_folds[[tipo]]))
    pred <- fun_predecir(setdiff(seq_len(nrow(train)), val),
                         x_train[val, , drop = FALSE],
                         superficie_train[val])
    for (modelo in names(oof)) oof[[modelo]][val] <- pred[[modelo]]
  }
  fun_registrar_cv(
    .archivo = nombre, .algoritmo = "LightGBM precio m2",
    .conjunto = "completo", .n_predictores = ncol(x_train),
    .hiperparametros = descripcion, .tipo_cv = tipo, .obs = y,
    .pred = oof$hibrida
  )
  for (modelo in names(oof)) {
    oof_m2[[paste0(modelo, "_", tipo)]] <- oof[[modelo]]
  }
  
  # Comparación con las mismas particiones (MAE en millones).
  fun_mae <- function(.pred, .filas) round(mean(abs(.pred - y)[.filas]) / 1e6, 1)
  fun_sesgo <- function(.pred, .filas) round(mean((.pred - y)[.filas]) / 1e6, 1)
  for (modelo in names(oof)) {
    pred <- oof[[modelo]]
    comparacion[[length(comparacion) + 1]] <- data.frame(
      tipo_cv = tipo, modelo = modelo,
      mae_toda_cv = fun_mae(pred, TRUE),
      mae_similares = fun_mae(pred, similares),
      sesgo_similares = fun_sesgo(pred, similares),
      mae_superficie_conocida = fun_mae(pred, conocida_train),
      mae_similares_conocida = fun_mae(pred, similares & conocida_train)
    )
  }
}
comparacion <- do.call(rbind, comparacion)
message("Comparación (millones; 'precio' = LightGBM de precio total; ",
        "'hibrida' = precio por m² donde la superficie es conocida):")
print(comparacion, row.names = FALSE)
saveRDS(oof_m2, "03_temp/oof_lgbm_m2.rds")

# Modelo final y envío -----------------------------------------------------
final <- fun_predecir(seq_len(nrow(train)), x_test, superficie_test)
fun_save_submission(.ids = test$property_id, .pred = final$hibrida,
                    .nombre = nombre)

# Limpiamos el entorno: borramos solo los objetos que creó este script.
rm(list = setdiff(ls(), objetos_previos))
