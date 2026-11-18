##########################################################
# 11_model_boosting.R - LightGBM (variables Metrocuadrado)
# Entradas: 03_temp/train_features.rds,
#           03_temp/test_features.rds, 03_temp/cv_folds.rds
# Resultado: CV, tuning bayesiano y submission de Kaggle.
##########################################################
objetos_previos <- ls()

# Paquetes y funciones del repositorio --------------------
if (!require(pacman)) install.packages("pacman")
pacman::p_load(lightgbm, rBayesianOptimization, dplyr, tibble)
source("01_code/functions/fun_save_submission.R")
source("01_code/functions/fun_metricas.R")

set.seed(2026)
hilos <- 6L
dir.create("02_outputs/tables", recursive = TRUE, showWarnings = FALSE)

# 1. Datos y variables ------------------------------------
train <- readRDS("03_temp/train_features.rds")
test  <- readRDS("03_temp/test_features.rds")
folds <- readRDS("03_temp/cv_folds.rds")

variables <- c(
  "estrato_manzana", "txt_antiguedad_anios", "bedrooms", "bathrooms",
  "txt_terraza", "txt_balcon", "property_type", "surface_total",
  "surface_covered", "txt_parqueadero", "lat", "lon"
)

faltantes <- setdiff(variables, intersect(names(train), names(test)))
if (length(faltantes)) {
  stop("Faltan predictores en train o test: ", paste(faltantes, collapse = ", "),
       ". Ejecute primero 03_build_geo_features.R y 04_build_text_features.R.")
}
if (!"price" %in% names(train)) stop("Falta price en train_features.rds.")
# El test puede no incluir ID; la plantilla de Kaggle contiene los IDs.
plantilla <- utils::read.csv("00_data/raw/submission_template.csv")
if (nrow(plantilla) != nrow(test)) {
  stop("La plantilla de Kaggle y test_features.rds tienen distinta cantidad de filas.")
}
# Si el ID existe en test con el mismo nombre, verificar tambien su orden.
col_id <- names(plantilla)[1]
if (col_id %in% names(test) &&
    !identical(as.character(plantilla[[1]]), as.character(test[[col_id]]))) {
  stop("El orden de las propiedades no coincide con submission_template.csv.")
}
ids_test <- plantilla[[1]]

# El ID no es predictor: solo se utiliza al generar la submission.
y <- as.numeric(train$price)
if (any(!is.finite(y)) || any(y <= 0)) stop("Hay precios invalidos en train.")

# Codificacion consistente de casa/apartamento (categorica).
niveles <- unique(c(as.character(train$property_type),
                    as.character(test$property_type)))
niveles <- niveles[!is.na(niveles)]
train$property_type <- match(as.character(train$property_type), niveles) - 1L
test$property_type  <- match(as.character(test$property_type), niveles) - 1L

# Solo numeros; los NA se conservan para LightGBM.
for (v in setdiff(variables, "property_type")) {
  train[[v]] <- suppressWarnings(as.numeric(as.character(train[[v]])))
  test[[v]]  <- suppressWarnings(as.numeric(as.character(test[[v]])))
}
x_train <- as.matrix(train[, variables, drop = FALSE])
x_test  <- as.matrix(test[, variables, drop = FALSE])
x_train[!is.finite(x_train)] <- NA_real_
x_test[!is.finite(x_test)] <- NA_real_

dtrain <- lightgbm::lgb.Dataset(
  data = x_train, label = y,
  categorical_feature = "property_type"
)
message("LightGBM Metrocuadrado: ", nrow(x_train), " filas y ",
        ncol(x_train), " predictores.")

# 2. Hiperparametros --------------------------------------
armar_params <- function(log_lr, num_leaves, max_depth, min_data_in_leaf,
                         feature_fraction, bagging_fraction, log_l1, log_l2) {
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
    num_threads = hilos, seed = 2026L, verbosity = -1L
  )
}

# 3. Tuning bayesiano con CV espacial ---------------------
ruta_ensayos <- "02_outputs/tables/LGBM_metrocuadrado_ensayos.csv"
if (file.exists(ruta_ensayos)) file.remove(ruta_ensayos)

puntuar <- function(log_lr, num_leaves, max_depth, min_data_in_leaf,
                    feature_fraction, bagging_fraction, log_l1, log_l2) {
  params <- armar_params(log_lr, num_leaves, max_depth, min_data_in_leaf,
                         feature_fraction, bagging_fraction, log_l1, log_l2)
  cv <- lightgbm::lgb.cv(
    params = params, data = dtrain, folds = folds$espacial,
    nrounds = 4000L, early_stopping_rounds = 100L,
    verbose = -1L, showsd = FALSE
  )
  mae <- as.numeric(cv$best_score)
  iter <- as.integer(cv$best_iter)
  
  fila <- data.frame(
    mae = mae, iteraciones = iter,
    log_lr = log_lr, num_leaves = num_leaves, max_depth = max_depth,
    min_data_in_leaf = min_data_in_leaf,
    feature_fraction = feature_fraction, bagging_fraction = bagging_fraction,
    log_l1 = log_l1, log_l2 = log_l2
  )
  utils::write.table(
    fila, ruta_ensayos, sep = ",", row.names = FALSE,
    col.names = !file.exists(ruta_ensayos),
    append = file.exists(ruta_ensayos)
  )
  message("MAE espacial: $", round(mae), " | rondas: ", iter)
  list(Score = -mae / 1e6, Pred = 0)
}

opt <- rBayesianOptimization::BayesianOptimization(
  FUN = puntuar,
  bounds = list(
    log_lr = log(c(0.008, 0.12)),
    num_leaves = c(15L, 127L),
    max_depth = c(4L, 12L),
    min_data_in_leaf = c(15L, 200L),
    feature_fraction = c(0.65, 1),
    bagging_fraction = c(0.65, 1),
    log_l1 = log1p(c(0, 20)),
    log_l2 = log1p(c(0, 20))
  ),
  init_points = 25L, n_iter = 125L, acq = "ei", verbose = TRUE
)

# Usar el mismo ensayo para seleccionar parametros y rondas.
ensayos <- utils::read.csv(ruta_ensayos)
b <- ensayos[which.min(ensayos$mae), ]
mejores <- armar_params(b$log_lr, b$num_leaves, b$max_depth,
                        b$min_data_in_leaf, b$feature_fraction,
                        b$bagging_fraction, b$log_l1, b$log_l2)
rondas <- as.integer(b$iteraciones)
message("Mejor MAE espacial: $", round(b$mae), " | rondas: ", rondas)

saveRDS(list(params = mejores, rondas = rondas, variables = variables),
        "03_temp/LGBM_metrocuadrado_config.rds")

# 4. Diagnostico OOF: espacial vs aleatorio ---------------
nombre <- "LGBM_metrocuadrado_bayes"
for (tipo in c("espacial", "aleatoria")) {
  oof <- rep(NA_real_, length(y))
  for (val in folds[[tipo]]) {
    modelo_cv <- lightgbm::lgb.train(
      params = mejores,
      data = lightgbm::lgb.Dataset(
        x_train[-val, , drop = FALSE], label = y[-val],
        categorical_feature = "property_type"
      ),
      nrounds = rondas, verbose = -1L
    )
    oof[val] <- as.numeric(predict(modelo_cv, x_train[val, , drop = FALSE]))
  }
  fun_registrar_cv(
    .archivo = nombre, .algoritmo = "LightGBM",
    .conjunto = "metrocuadrado", .n_predictores = length(variables),
    .hiperparametros = paste0("Bayes espacial; rondas=", rondas),
    .tipo_cv = tipo, .obs = y, .pred = oof
  )
  message("MAE ", tipo, " (OOF): $", round(mean(abs(oof - y))))
}

# 5. Entrenamiento final y submission ---------------------
modelo_final <- lightgbm::lgb.train(
  params = mejores, data = dtrain, nrounds = rondas, verbose = -1L
)
lightgbm::lgb.save(modelo_final, "03_temp/LGBM_metrocuadrado_modelo.txt")
utils::write.csv(
  lightgbm::lgb.importance(modelo_final),
  "02_outputs/tables/LGBM_metrocuadrado_importancia.csv", row.names = FALSE
)
pred_test <- pmax(as.numeric(predict(modelo_final, x_test)), 1)
fun_save_submission(.ids = ids_test, .pred = pred_test, .nombre = nombre)

message("Listo: tuning, MAE, modelo y submission de Metrocuadrado.")
rm(list = setdiff(ls(), objetos_previos))
