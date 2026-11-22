##########################################################
# 11d_model_boosting_vecinos.R
#
# Objetivo: agregar al LightGBM híbrido del script 11b el precio de las
# viviendas vecinas de train (rezago espacial del precio) y generar su
# envío a Kaggle.
#
# Etapa del Problem Set: Etapa 2 (familia Boosting) y Etapa 4 (por qué el
# mejor modelo falla en Chapinero y cómo corregirlo).
# Insumos:  03_temp/train_model.rds, test_model.rds,
#           conjuntos_predictores.rds (script 05); 03_temp/cv_folds.rds y
#           similitud_chapinero.rds (script 06);
#           02_outputs/tables/LGBM_completo_ensayos.csv (script 11).
# Salidas:  02_outputs/predictions/LGBM_completo_m2_vecinos_....csv (envío).
#           Filas en 02_outputs/tables/registro_modelos.csv.
#           03_temp/oof_lgbm_vecinos.rds (predicciones fuera de muestra).
#
# Decisiones:
#   - Problema que se ataca: el modelo se queda corto en Chapinero porque
#     no "sabe" cuánto vale el m² en esa zona; los árboles no extrapolan.
#     Casi todas las viviendas de Chapinero tienen un anuncio de train a
#     menos de 500 m (mediana: 90 m) y el 18% comparte coordenadas con
#     alguno (mismo edificio). El precio de esos vecinos es información
#     directa sobre el nivel de precios de la zona (rezago espacial del
#     precio, Lecture 14).
#   - Variables nuevas (3), además del conjunto completo:
#       * vec_precio_m2: mediana del precio por m² de las 10 viviendas de
#         referencia más cercanas con superficie real;
#       * vec_dist_m: distancia mediana a esas 10 viviendas (le dice al
#         modelo qué tan local es esa información);
#       * vec_precio: mediana del precio total de las 10 viviendas de
#         referencia más cercanas.
#   - Sin fuga de información: las viviendas de referencia son SOLO las de
#     entrenamiento de cada fold. Las de validación nunca aportan su
#     precio, ni siquiera a sus vecinas. Además se descartan los vecinos
#     que son el mismo anuncio (misma descripción) que la vivienda, porque
#     su precio es el mismo que se quiere predecir. En el test, las
#     viviendas de referencia son todas las de train.
#   - Advertencia para leer la CV espacial: al apartar una región entera,
#     sus viviendas quedan lejos de las de entrenamiento, mientras que en
#     Chapinero los vecinos están a unos 90 m. Por eso la CV espacial
#     subestima lo que aportan estas variables. El script imprime la
#     distancia típica a los vecinos en cada caso.
#   - Todo lo demás es igual al 11b: modelo híbrido (precio por m² x
#     superficie donde la superficie es real; precio total donde no),
#     hiperparámetros del mejor ensayo del 11 y rondas del 11b.
##########################################################

# Objetos que ya existían en el entorno antes de correr este script. Al
# final se borra solo lo que este script creó, sin tocar lo demás.
objetos_previos <- ls()

# Parámetros ---------------------------------------------------------------
semilla <- 2026
hilos <- 6L
k_vecinos <- 10L
crs_m <- 3116  # Coordenadas proyectadas en metros, como en el script 03.
max_rondas <- 10000L
paciencia <- 100L

# Paquetes y funciones -----------------------------------------------------
if (!require(pacman)) install.packages("pacman")
pacman::p_load(dplyr, lightgbm, sf, FNN)

source("01_code/functions/fun_save_submission.R")
source("01_code/functions/fun_metricas.R")
source("01_code/functions/fun_vecinos.R")

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

# Superficie conocida (no imputada), igual que en el 11b.
fun_superficie <- function(.datos) {
  #' Superficie real de cada vivienda, o NA si fue imputada
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
precio_m2 <- y / superficie_train
filas_m2 <- which(conocida_train)

# Coordenadas en metros y grupos de anuncio.
fun_xy <- function(.datos) {
  sf::st_coordinates(sf::st_transform(
    sf::st_as_sf(.datos[c("lon", "lat")], coords = c("lon", "lat"),
                 crs = 4326), crs_m))
}
xy_train <- fun_xy(train)
xy_test <- fun_xy(test)
grupo_train <- fun_grupo_anuncio(train)
grupo_test <- fun_grupo_anuncio(test)

# Variables de vecinos ------------------------------------------------------
fun_variables_vecinos <- function(.ref, .xy_obj, .grupo_obj) {
  #' Rezago espacial del precio con las viviendas de referencia .ref
  #'
  #' @param .ref \code{integer}. Filas de train que pueden aportar su precio.
  #' @param .xy_obj \code{matrix}. Coordenadas de las viviendas objetivo.
  #' @param .grupo_obj \code{character}. Grupo de anuncio de las objetivo.
  #'
  #' @returns \code{matrix} con vec_precio_m2, vec_dist_m y vec_precio.
  ref_m2 <- intersect(.ref, filas_m2)
  m2 <- fun_vecinos(xy_train[ref_m2, , drop = FALSE], precio_m2[ref_m2],
                    grupo_train[ref_m2], .xy_obj, .grupo_obj, .k = k_vecinos)
  total <- fun_vecinos(xy_train[.ref, , drop = FALSE], y[.ref],
                       grupo_train[.ref], .xy_obj, .grupo_obj, .k = k_vecinos)
  cbind(vec_precio_m2 = m2$mediana, vec_dist_m = m2$distancia,
        vec_precio = total$mediana)
}

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

# Rondas del modelo de precio por m²: igual que en el 11b (CV espacial con
# el conjunto completo, solo superficie conocida).
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

# Funciones del modelo híbrido ----------------------------------------------
fun_acotar <- function(.pred) pmin(pmax(.pred, rango_precio[1]), rango_precio[2])

fun_predecir <- function(.filas_ajuste, .x_ajuste, .x_nuevo,
                         .superficie_nueva) {
  #' Modelo híbrido (como en el 11b) con las variables de vecinos
  #'
  #' @param .filas_ajuste \code{integer}. Filas de train para entrenar.
  #' @param .x_ajuste Matriz de predictores de esas filas (con vecinos).
  #' @param .x_nuevo Matriz de predictores donde se predice (con vecinos).
  #' @param .superficie_nueva \code{numeric}. Superficie conocida (o NA).
  #'
  #' @returns \code{numeric}: predicción híbrida en pesos, acotada.
  conocidas <- which(.filas_ajuste %in% filas_m2)
  modelo_precio <- lightgbm::lgb.train(
    params = params,
    data = lightgbm::lgb.Dataset(.x_ajuste, label = y[.filas_ajuste]),
    nrounds = rondas_precio, verbose = -1L
  )
  modelo_m2 <- lightgbm::lgb.train(
    params = params,
    data = lightgbm::lgb.Dataset(.x_ajuste[conocidas, , drop = FALSE],
                                 label = precio_m2[.filas_ajuste[conocidas]]),
    nrounds = rondas_m2, verbose = -1L
  )
  pred_precio <- fun_acotar(stats::predict(modelo_precio, .x_nuevo))
  pred_m2 <- fun_acotar(stats::predict(modelo_m2, .x_nuevo) * .superficie_nueva)
  ifelse(is.na(.superficie_nueva), pred_precio, pred_m2)
}

# Validación cruzada -------------------------------------------------------
nombre <- paste0("LGBM_completo_m2_vecinos_k", k_vecinos, "_lr_",
                 fun_hiperparam_str(params$learning_rate, 3),
                 "_hojas_", params$num_leaves, "_rondas_", rondas_m2)
descripcion <- paste0("Híbrido del 11b + rezago espacial del precio (", k_vecinos,
                      " vecinos de entrenamiento: precio por m², distancia y ",
                      "precio total); hiperparámetros del 11; rondas m2=",
                      rondas_m2, "; rondas precio=", rondas_precio)
oof_vecinos <- data.frame(property_id = train$property_id)
distancias_val <- list()

for (tipo in c("espacial", "aleatoria")) {
  oof <- rep(NA_real_, nrow(train))
  for (i in seq_along(cv_folds[[tipo]])) {
    val <- cv_folds[[tipo]][[i]]
    ajuste <- setdiff(seq_len(nrow(train)), val)
    message("CV ", tipo, ": fold ", i, "/", length(cv_folds[[tipo]]))
    # Referencia = solo las filas de entrenamiento del fold.
    vec_ajuste <- fun_variables_vecinos(ajuste, xy_train[ajuste, , drop = FALSE],
                                        grupo_train[ajuste])
    vec_val <- fun_variables_vecinos(ajuste, xy_train[val, , drop = FALSE],
                                     grupo_train[val])
    distancias_val[[tipo]] <- c(distancias_val[[tipo]], vec_val[, "vec_dist_m"])
    oof[val] <- fun_predecir(ajuste,
                             cbind(x_train[ajuste, , drop = FALSE], vec_ajuste),
                             cbind(x_train[val, , drop = FALSE], vec_val),
                             superficie_train[val])
  }
  fun_registrar_cv(
    .archivo = nombre, .algoritmo = "LightGBM precio m2 + vecinos",
    .conjunto = "completo + vecinos", .n_predictores = ncol(x_train) + 3L,
    .hiperparametros = descripcion, .tipo_cv = tipo, .obs = y, .pred = oof
  )
  oof_vecinos[[paste0("LGBM_vecinos_", tipo)]] <- oof
  message("CV ", tipo, ": MAE = ", round(mean(abs(oof - y)) / 1e6, 1),
          " | similares a Chapinero = ",
          round(mean(abs(oof - y)[similares]) / 1e6, 1), " millones.")
}
saveRDS(oof_vecinos, "03_temp/oof_lgbm_vecinos.rds")

# Modelo final y envío -----------------------------------------------------
todas <- seq_len(nrow(train))
vec_train <- fun_variables_vecinos(todas, xy_train, grupo_train)
vec_test <- fun_variables_vecinos(todas, xy_test, grupo_test)
pred_test <- fun_predecir(todas, cbind(x_train, vec_train),
                          cbind(x_test, vec_test), superficie_test)
fun_save_submission(.ids = test$property_id, .pred = pred_test,
                    .nombre = nombre)

# Resumen ------------------------------------------------------------------
message("Distancia mediana a los ", k_vecinos, " vecinos con superficie ",
        "real (m): CV espacial = ",
        round(stats::median(distancias_val$espacial, na.rm = TRUE)),
        "; CV aleatoria = ",
        round(stats::median(distancias_val$aleatoria, na.rm = TRUE)),
        "; test (Chapinero) = ",
        round(stats::median(vec_test[, "vec_dist_m"], na.rm = TRUE)), ".")
read.csv("02_outputs/tables/registro_modelos.csv") |>
  filter(algoritmo %in% c("LightGBM precio m2",
                          "LightGBM precio m2 + vecinos")) |>
  transmute(archivo = substr(archivo, 1, 40), tipo_cv,
            mae_millones = round(mae / 1e6, 1),
            mae_similares_chapinero = round(mae_similares_chapinero / 1e6, 1),
            sesgo_similares_chapinero = round(sesgo_similares_chapinero / 1e6, 1)) |>
  print(row.names = FALSE)

# Limpiamos el entorno: borramos solo los objetos que creó este script.
rm(list = setdiff(ls(), objetos_previos))
