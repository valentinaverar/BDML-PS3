##########################################################
# 11_model_boosting.R
#
# Objetivo: entrenar la familia Boosting (LightGBM) con los conjuntos
# completo y ampliado, medir su error con CV espacial y aleatoria, y
# generar sus envíos a Kaggle.
#
# Etapa del Problem Set: Etapa 2 (familia Boosting) y Etapa 3.
# Insumos:  03_temp/train_model.rds, test_model.rds,
#           conjuntos_predictores.rds y tfidf_*.rds (scripts 04 y 05);
#           03_temp/cv_folds.rds y similitud_chapinero.rds (script 06).
# Salidas:  02_outputs/predictions/LGBM_<conjunto>_....csv (envíos).
#           02_outputs/tables/LGBM_completo_ensayos.csv: cada combinación
#           de hiperparámetros probada y su MAE en CV espacial.
#           02_outputs/tables/LGBM_<conjunto>_importancia.csv.
#           Filas en 02_outputs/tables/registro_modelos.csv.
#           03_temp/oof_lgbm.rds (predicciones fuera de muestra) y
#           03_temp/LGBM_<conjunto>_modelo.txt.
#
# Decisiones:
#   - LightGBM (decisión de Andrés): boosting de árboles que optimiza
#     directamente el error absoluto (objective = "regression_l1"), la
#     métrica de Kaggle, y acepta matrices dispersas (TF-IDF).
#   - Hiperparámetros elegidos con la CV espacial (k-means de 20 regiones,
#     script 06), con búsqueda bayesiana: 10 puntos iniciales + 20
#     iteraciones. Si la búsqueda bayesiana falla, se completan los 30
#     ensayos con búsqueda aleatoria en los mismos límites.
#   - La búsqueda se hace con el conjunto completo. El ampliado usa los
#     mismos hiperparámetros y solo vuelve a elegir el número de rondas: así
#     la diferencia entre los dos mide solo el aporte del TF-IDF.
#   - Las predicciones se acotan al rango de precios de train (la muestra
#     viene truncada entre 300 y 1.650 millones).
#   - Si ya existe la tabla de ensayos y buscar_hiperparametros es FALSE,
#     se reutiliza el mejor ensayo y solo se reentrena (minutos). La primera
#     vez siempre hace la búsqueda completa.
##########################################################

# Objetos que ya existían en el entorno antes de correr este script. Al
# final se borra solo lo que este script creó, sin tocar lo demás.
objetos_previos <- ls()

# Parámetros ---------------------------------------------------------------
buscar_hiperparametros <- FALSE  # TRUE: repetir la búsqueda aunque exista.
semilla <- 2026
hilos <- 6L
puntos_iniciales <- 10L
iteraciones_bayes <- 20L
max_rondas <- 5000L
paciencia <- 100L  # Rondas sin mejora antes de detener (early stopping).

limites <- list(
  log_lr = log(c(0.008, 0.12)),
  num_leaves = c(15L, 127L),
  max_depth = c(4L, 12L),
  min_data_in_leaf = c(15L, 200L),
  feature_fraction = c(0.3, 1),
  bagging_fraction = c(0.65, 1),
  log_l1 = log1p(c(0, 20)),
  log_l2 = log1p(c(0, 20))
)

# Paquetes y funciones -----------------------------------------------------
if (!require(pacman)) install.packages("pacman")
pacman::p_load(dplyr, Matrix, lightgbm, rBayesianOptimization)

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
y <- train$price
rango_precio <- range(y)

# Matrices de predictores. LightGBM necesita todo numérico: el tipo de
# vivienda pasa a un indicador de casa.
fun_matriz <- function(.datos, .vars) {
  x <- .datos[.vars]
  x$property_type <- as.integer(x$property_type == "Casa")
  as.matrix(as.data.frame(lapply(x, as.numeric)))
}
x_completo_train <- fun_matriz(train, conjuntos_predictores$completo)
x_completo_test <- fun_matriz(test, conjuntos_predictores$completo)

# Ampliado: completo + TF-IDF (matriz dispersa, mismo orden de filas). Los
# nombres de los términos se limpian porque LightGBM no acepta algunos
# caracteres en los nombres de las variables.
fun_ampliar <- function(.x, .ruta) {
  tfidf <- readRDS(.ruta)
  colnames(tfidf) <- make.unique(paste0(
    "tfidf_", gsub("[^[:alnum:]_]", "_", colnames(tfidf))))
  cbind(Matrix::Matrix(.x, sparse = TRUE), tfidf)
}
x_ampliado_train <- fun_ampliar(x_completo_train,
                                conjuntos_predictores$tfidf_ampliado[["train"]])
x_ampliado_test <- fun_ampliar(x_completo_test,
                               conjuntos_predictores$tfidf_ampliado[["test"]])
message("LightGBM: completo = ", ncol(x_completo_train),
        " predictores; ampliado = ", ncol(x_ampliado_train), ".")

# Funciones de LightGBM ------------------------------------------------------
fun_armar_params <- function(log_lr, num_leaves, max_depth, min_data_in_leaf,
                             feature_fraction, bagging_fraction, log_l1,
                             log_l2) {
  #' Convertir un punto de la búsqueda en parámetros de LightGBM
  #'
  #' @description La tasa de aprendizaje y las penalizaciones se buscan en
  #' escala logarítmica. El número de hojas se limita a 2^profundidad.
  #'
  #' @returns \code{list} de parámetros para lgb.cv() y lgb.train().
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

fun_cv_lgbm <- function(.params, .x, .folds) {
  #' MAE de CV y número óptimo de rondas (con early stopping)
  #'
  #' @param .params \code{list}. Parámetros de LightGBM.
  #' @param .x Matriz de predictores de train (densa o dispersa).
  #' @param .folds \code{list}. Índices de validación de cada fold.
  #'
  #' @returns \code{list} con mae y rondas.
  cv <- lightgbm::lgb.cv(
    params = .params, data = lightgbm::lgb.Dataset(.x, label = y),
    folds = .folds, nrounds = max_rondas,
    early_stopping_rounds = paciencia, verbose = -1L
  )
  list(mae = as.numeric(cv$best_score), rondas = as.integer(cv$best_iter))
}

fun_lgbm_predecir <- function(.params, .rondas, .x_ajuste, .y_ajuste,
                              .x_nuevo) {
  #' Entrenar LightGBM y predecir, acotado al rango de precios de train
  #'
  #' @returns \code{list} con el modelo y las predicciones en pesos.
  modelo <- lightgbm::lgb.train(
    params = .params,
    data = lightgbm::lgb.Dataset(.x_ajuste, label = .y_ajuste),
    nrounds = .rondas, verbose = -1L
  )
  pred <- stats::predict(modelo, .x_nuevo)
  list(modelo = modelo,
       pred = pmin(pmax(pred, rango_precio[1]), rango_precio[2]))
}

# 1. Búsqueda de hiperparámetros (conjunto completo, CV espacial) ----------
ruta_ensayos <- "02_outputs/tables/LGBM_completo_ensayos.csv"
dir.create("02_outputs/tables", recursive = TRUE, showWarnings = FALSE)
total_ensayos <- puntos_iniciales + iteraciones_bayes

if (buscar_hiperparametros || !file.exists(ruta_ensayos)) {
  if (file.exists(ruta_ensayos)) file.remove(ruta_ensayos)
  
  fun_puntuar <- function(log_lr, num_leaves, max_depth, min_data_in_leaf,
                          feature_fraction, bagging_fraction, log_l1,
                          log_l2) {
    #' Evaluar un punto de la búsqueda y guardarlo en la tabla de ensayos
    inicio <- proc.time()[["elapsed"]]
    params <- fun_armar_params(log_lr, num_leaves, max_depth,
                               min_data_in_leaf, feature_fraction,
                               bagging_fraction, log_l1, log_l2)
    cv <- fun_cv_lgbm(params, x_completo_train, cv_folds$espacial)
    existe <- file.exists(ruta_ensayos)
    utils::write.table(
      data.frame(mae = cv$mae, rondas = cv$rondas, log_lr, num_leaves,
                 max_depth, min_data_in_leaf, feature_fraction,
                 bagging_fraction, log_l1, log_l2),
      ruta_ensayos, sep = ",", row.names = FALSE,
      col.names = !existe, append = existe)
    message("Ensayo ", nrow(utils::read.csv(ruta_ensayos)), "/",
            total_ensayos, " | MAE espacial = ", round(cv$mae / 1e6, 1),
            " millones | rondas = ", cv$rondas, " | ",
            round(proc.time()[["elapsed"]] - inicio), " s")
    list(Score = -cv$mae / 1e6, Pred = 0)
  }
  
  set.seed(semilla)
  tipo_busqueda <- tryCatch({
    rBayesianOptimization::BayesianOptimization(
      FUN = fun_puntuar, bounds = limites, init_points = puntos_iniciales,
      n_iter = iteraciones_bayes, acq = "ei", verbose = FALSE)
    "bayesiana"
  }, error = function(e) {
    message("La búsqueda bayesiana falló (", conditionMessage(e),
            "). Se completa con búsqueda aleatoria.")
    "aleatoria"
  })
  # Respaldo: completar los ensayos que falten con búsqueda aleatoria.
  n_hechos <- if (file.exists(ruta_ensayos)) {
    nrow(utils::read.csv(ruta_ensayos))
  } else {
    0
  }
  while (n_hechos < total_ensayos) {
    candidato <- lapply(limites, function(.l) {
      if (is.integer(.l)) sample(.l[1]:.l[2], 1) else stats::runif(1, .l[1], .l[2])
    })
    do.call(fun_puntuar, candidato)
    n_hechos <- n_hechos + 1
  }
} else {
  tipo_busqueda <- "ensayos guardados"
  message("Se reutilizan los ensayos de ", ruta_ensayos,
          " (buscar_hiperparametros = FALSE).")
}

ensayos <- utils::read.csv(ruta_ensayos)
mejor <- ensayos[which.min(ensayos$mae), ]
params <- do.call(fun_armar_params, as.list(mejor[names(limites)]))
message("Mejor ensayo: MAE espacial = ", round(mejor$mae / 1e6, 1),
        " millones, lr = ", round(params$learning_rate, 4), ", hojas = ",
        params$num_leaves, ", profundidad = ", params$max_depth,
        ", rondas = ", mejor$rondas, ".")

# 2. Variantes: CV, envío e importancia -------------------------------------
variantes <- list(
  completo = list(train = x_completo_train, test = x_completo_test),
  ampliado = list(train = x_ampliado_train, test = x_ampliado_test)
)
oof_lgbm <- data.frame(property_id = train$property_id)

for (conjunto in names(variantes)) {
  x_tr <- variantes[[conjunto]]$train
  x_te <- variantes[[conjunto]]$test
  
  # Número de rondas: el del mejor ensayo en completo; en ampliado se
  # vuelve a elegir con la CV espacial (mismos hiperparámetros).
  rondas <- if (conjunto == "completo") {
    mejor$rondas
  } else {
    fun_cv_lgbm(params, x_tr, cv_folds$espacial)$rondas
  }
  nombre <- paste0("LGBM_", conjunto,
                   "_lr_", fun_hiperparam_str(params$learning_rate, 3),
                   "_hojas_", params$num_leaves,
                   "_prof_", params$max_depth,
                   "_minhoja_", params$min_data_in_leaf,
                   "_rondas_", rondas)
  descripcion <- paste0("Búsqueda ", tipo_busqueda, "; lr=",
                        round(params$learning_rate, 4), "; hojas=",
                        params$num_leaves, "; prof=", params$max_depth,
                        "; min_hoja=", params$min_data_in_leaf,
                        "; ff=", round(params$feature_fraction, 2),
                        "; rondas=", rondas)
  
  # Error fuera de muestra con cada tipo de CV.
  for (tipo in c("espacial", "aleatoria")) {
    oof <- rep(NA_real_, nrow(train))
    for (val in cv_folds[[tipo]]) {
      oof[val] <- fun_lgbm_predecir(params, rondas,
                                    x_tr[-val, , drop = FALSE], y[-val],
                                    x_tr[val, , drop = FALSE])$pred
    }
    fun_registrar_cv(
      .archivo = nombre, .algoritmo = "LightGBM", .conjunto = conjunto,
      .n_predictores = ncol(x_tr), .hiperparametros = descripcion,
      .tipo_cv = tipo, .obs = y, .pred = oof
    )
    oof_lgbm[[paste0("LGBM_", conjunto, "_", tipo)]] <- oof
    message("LGBM ", conjunto, " | CV ", tipo, ": MAE = ",
            round(mean(abs(oof - y)) / 1e6, 1), " millones.")
  }
  
  # Modelo final con todo train y envío a Kaggle.
  final <- fun_lgbm_predecir(params, rondas, x_tr, y, x_te)
  fun_save_submission(.ids = test$property_id, .pred = final$pred,
                      .nombre = nombre)
  lightgbm::lgb.save(final$modelo,
                     paste0("03_temp/LGBM_", conjunto, "_modelo.txt"))
  utils::write.csv(lightgbm::lgb.importance(final$modelo),
                   paste0("02_outputs/tables/LGBM_", conjunto,
                          "_importancia.csv"), row.names = FALSE)
}

saveRDS(oof_lgbm, "03_temp/oof_lgbm.rds")

# Resumen ------------------------------------------------------------------
read.csv("02_outputs/tables/registro_modelos.csv") |>
  filter(algoritmo == "LightGBM") |>
  transmute(archivo = substr(archivo, 1, 30), tipo_cv,
            mae_millones = round(mae / 1e6, 1),
            mae_similares_chapinero = round(mae_similares_chapinero / 1e6, 1),
            sesgo_millones = round(sesgo_medio / 1e6, 1),
            sesgo_similares_chapinero = round(sesgo_similares_chapinero / 1e6, 1),
            pct_sobreestima = round(pct_sobreestima, 1)) |>
  print(row.names = FALSE)

# Limpiamos el entorno: borramos solo los objetos que creó este script.
rm(list = setdiff(ls(), objetos_previos))
