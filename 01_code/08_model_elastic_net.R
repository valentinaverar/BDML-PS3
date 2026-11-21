##########################################################
# 08_model_elastic_net.R
#
# Objetivo: estimar la red elástica (Elastic Net) sobre log(precio) con el
# conjunto completo, elegir alpha y lambda con la CV espacial, medir el
# error también con la CV aleatoria y generar el envío a Kaggle.
#
# Etapa del Problem Set: Etapa 2 (familia Elastic Net) y Etapa 3
# (CV espacial frente a aleatoria).
# Insumos:  03_temp/train_model.rds, test_model.rds,
#           conjuntos_predictores.rds (script 05) y cv_folds.rds (script 06).
# Salidas:  02_outputs/predictions/EN_completo_a_*_l_*.csv (envío).
#           Filas en 02_outputs/tables/registro_modelos.csv.
#           03_temp/oof_en.rds: predicciones fuera de muestra.
#           03_temp/en_ajuste.rds: alpha, lambda y número de coeficientes
#           distintos de cero.
#
# Decisiones (material del curso, Lecture 15):
#   - glmnet, el mismo estimador que cv.gamlr / caret method = "glmnet"
#     de la clase de texto. La penalización encoge las raíces de la
#     descripción, que en MCO entran todas y están correlacionadas.
#   - Se modela log(precio) y se vuelve a pesos con exp(), igual que la
#     regresión lineal (script 07) y que el ejemplo de Ames de la clase
#     espacial. Las predicciones se acotan al rango de precios de train.
#   - alpha (mezcla entre ridge y lasso) y lambda se eligen por el MAE en
#     pesos en la CV espacial, no por el error cuadrático del logaritmo:
#     la métrica de Kaggle es el MAE. La grilla de alpha es 0, 0.25, 0.5,
#     0.75 y 1. La búsqueda es una grilla, como pide el enunciado.
#   - Esos hiperparámetros no se reeligien con la CV aleatoria: así las
#     dos filas del registro comparan el mismo modelo.
##########################################################

# Objetos que ya existían en el entorno antes de correr este script. Al
# final se borra solo lo que este script creó, sin tocar lo demás.
objetos_previos <- ls()

# Paquetes y funciones -----------------------------------------------------
if (!require(pacman)) install.packages("pacman")
pacman::p_load(dplyr, glmnet)

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

vars <- conjuntos_predictores$completo
rango_precio <- range(train$price)
y <- log(train$price)

# Misma matriz de diseño en train y en test (la dummy de property_type
# usa los niveles de train).
formula_x <- reformulate(vars)
x_train <- model.matrix(formula_x, data = train)[, -1, drop = FALSE]
x_test <- model.matrix(formula_x, data = test)[, -1, drop = FALSE]
if (!identical(colnames(x_train), colnames(x_test))) {
  stop("Train y test no quedaron con las mismas columnas en la matriz ",
       "de diseño.")
}

foldid_de <- function(.folds) {
  foldid <- integer(nrow(train))
  for (i in seq_along(.folds)) foldid[.folds[[i]]] <- i
  if (any(foldid == 0)) stop("Hay propiedades sin fold.")
  foldid
}

# MAE en pesos de las predicciones pre-validadas de cv.glmnet (en log).
fun_mae_por_lambda <- function(.preval) {
  precio <- exp(.preval)
  precio <- pmin(pmax(precio, rango_precio[1]), rango_precio[2])
  colMeans(abs(precio - train$price))
}

# 1. Elegir alpha y lambda con la CV espacial ------------------------------
alphas <- c(0, 0.25, 0.5, 0.75, 1)
foldid_espacial <- foldid_de(cv_folds$espacial)
busqueda <- vector("list", length(alphas))

for (i in seq_along(alphas)) {
  message("CV espacial, alpha = ", alphas[i], " ...")
  ajuste <- cv.glmnet(
    x_train, y, alpha = alphas[i], foldid = foldid_espacial,
    type.measure = "mse", keep = TRUE, standardize = TRUE
  )
  mae_lambda <- fun_mae_por_lambda(ajuste$fit.preval)
  mejor <- which.min(mae_lambda)
  busqueda[[i]] <- list(
    alpha = alphas[i],
    lambda = ajuste$lambda[mejor],
    mae = mae_lambda[mejor],
    preval = ajuste$fit.preval[, mejor]
  )
  message("  mejor lambda = ", format(ajuste$lambda[mejor], digits = 4),
          "; MAE espacial = ", round(mae_lambda[mejor] / 1e6, 1),
          " millones.")
}

elegido <- busqueda[[which.min(vapply(busqueda, function(b) b$mae, numeric(1)))]]
message("Elegido: alpha = ", elegido$alpha,
        ", lambda = ", format(elegido$lambda, digits = 4), ".")

# Coeficientes distintos de cero en el modelo final (cuántas raíces
# sobreviven a la penalización).
modelo <- glmnet(x_train, y, alpha = elegido$alpha, standardize = TRUE)
coeficientes <- as.matrix(coef(modelo, s = elegido$lambda))
n_activos <- sum(coeficientes[-1, 1] != 0)

hiperparametros <- paste0(
  "Elastic Net sobre log(precio); alpha ", elegido$alpha,
  ", lambda ", format(elegido$lambda, digits = 4, scientific = FALSE),
  "; ", n_activos, " coeficientes distintos de cero"
)
# sprintf evita la notación científica: as.character(0.0004) es "4e-04"
# y ese guion no cabe en el nombre del archivo de Kaggle.
nombre <- paste0(
  "EN_completo_a_", fun_hiperparam_str(elegido$alpha),
  "_l_", gsub(".", "_", sprintf("%.6f", elegido$lambda), fixed = TRUE)
)

# 2. Error fuera de muestra con cada tipo de CV ----------------------------
# Espacial: ya está en las predicciones pre-validadas del alpha elegido.
# Aleatoria: el mismo alpha y el mismo lambda, sin volver a elegirlos.
oof_en <- data.frame(property_id = train$property_id)

oof_espacial <- exp(elegido$preval)
oof_espacial <- pmin(pmax(oof_espacial, rango_precio[1]), rango_precio[2])
fun_registrar_cv(
  .archivo = nombre, .algoritmo = "Elastic Net",
  .conjunto = "completo", .n_predictores = length(vars),
  .hiperparametros = hiperparametros, .tipo_cv = "espacial",
  .obs = train$price, .pred = oof_espacial
)
oof_en[[paste0(nombre, "_espacial")]] <- oof_espacial
message(nombre, " | CV espacial: MAE = ",
        round(mean(abs(oof_espacial - train$price)) / 1e6, 1), " millones.")

oof_aleatoria <- rep(NA_real_, nrow(train))
for (val in cv_folds$aleatoria) {
  ajuste_fold <- glmnet(x_train[-val, , drop = FALSE], y[-val],
                        alpha = elegido$alpha, standardize = TRUE)
  pred_log <- as.numeric(predict(ajuste_fold, x_train[val, , drop = FALSE],
                                 s = elegido$lambda))
  oof_aleatoria[val] <- pmin(pmax(exp(pred_log), rango_precio[1]),
                             rango_precio[2])
}
fun_registrar_cv(
  .archivo = nombre, .algoritmo = "Elastic Net",
  .conjunto = "completo", .n_predictores = length(vars),
  .hiperparametros = hiperparametros, .tipo_cv = "aleatoria",
  .obs = train$price, .pred = oof_aleatoria
)
oof_en[[paste0(nombre, "_aleatoria")]] <- oof_aleatoria
message(nombre, " | CV aleatoria: MAE = ",
        round(mean(abs(oof_aleatoria - train$price)) / 1e6, 1), " millones.")

# 3. Modelo final con todo train y envío a Kaggle --------------------------
pred_test <- as.numeric(predict(modelo, x_test, s = elegido$lambda))
pred_test <- pmin(pmax(exp(pred_test), rango_precio[1]), rango_precio[2])
fun_save_submission(.ids = test$property_id, .pred = pred_test,
                    .nombre = nombre)

saveRDS(oof_en, "03_temp/oof_en.rds")
saveRDS(list(alpha = elegido$alpha, lambda = elegido$lambda,
             n_activos = n_activos, nombre = nombre,
             mae_busqueda = vapply(busqueda, function(b) b$mae, numeric(1)),
             alphas = alphas),
        "03_temp/en_ajuste.rds")

# Resumen ------------------------------------------------------------------
read.csv("02_outputs/tables/registro_modelos.csv") |>
  filter(algoritmo == "Elastic Net") |>
  transmute(archivo, tipo_cv,
            mae_millones = round(mae / 1e6, 1),
            mae_similares_chapinero = round(mae_similares_chapinero / 1e6, 1),
            sesgo_millones = round(sesgo_medio / 1e6, 1),
            sesgo_similares_chapinero = round(sesgo_similares_chapinero / 1e6, 1),
            pct_sobreestima = round(pct_sobreestima, 1)) |>
  print(row.names = FALSE)

message("CSV para Kaggle: 02_outputs/predictions/", nombre, ".csv")

# Limpiamos el entorno: borramos solo los objetos que creó este script.
rm(list = setdiff(ls(), objetos_previos))
