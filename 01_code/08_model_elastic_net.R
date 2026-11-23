##########################################################
# 08_model_elastic_net.R
#
# Objetivo: estimar la familia Elastic Net con los conjuntos completo y
# ampliado (experimento con TF-IDF), elegir sus hiperparámetros con la CV
# espacial, medir su error con CV espacial y aleatoria, y generar sus
# envíos a Kaggle.
#
# Etapa del Problem Set: Etapa 2 (familia Elastic Net) y Etapa 3.
# Insumos:  03_temp/train_model.rds, test_model.rds,
#           conjuntos_predictores.rds y tfidf_*.rds (scripts 04 y 05);
#           03_temp/cv_folds.rds y similitud_chapinero.rds (script 06).
# Salidas:  02_outputs/predictions/EN_<conjunto>_....csv (envíos).
#           02_outputs/tables/EN_<conjunto>_busqueda.csv: mejor lambda y su
#           MAE en CV espacial para cada alpha.
#           02_outputs/tables/EN_<conjunto>_coeficientes.csv: coeficientes
#           distintos de cero del modelo final.
#           Filas en 02_outputs/tables/registro_modelos.csv.
#           03_temp/oof_en.rds (predicciones fuera de muestra).
#
# Decisiones:
#   - Elastic Net penaliza la suma de los coeficientes en valor absoluto
#     (Lasso, alpha = 1) y la de sus cuadrados (Ridge, alpha = 0). Sirve
#     para muchos predictores, como las raíces del texto o el TF-IDF:
#     encoge los coeficientes y, con alpha > 0, deja en cero los que no
#     aportan.
#   - Se modela log(precio), como la regresión lineal (07), y se devuelve
#     a pesos con exp(). Las predicciones se acotan al rango de train.
#   - Hiperparámetros: alpha en {0; 0,25; 0,5; 0,75; 1} y, para cada uno,
#     la secuencia de lambdas de glmnet. Se elige la combinación con menor
#     MAE en pesos en la CV espacial (k-means de 20 regiones, script 06),
#     usando las predicciones fuera de muestra de cv.glmnet().
#   - Cada conjunto tiene su propia búsqueda: con 12.000 columnas más, el
#     ampliado necesita otra penalización. Es barato (minutos).
#   - glmnet estandariza los predictores antes de penalizar (por defecto),
#     así la penalización no depende de las unidades de cada variable.
##########################################################

# Objetos que ya existían en el entorno antes de correr este script. Al
# final se borra solo lo que este script creó, sin tocar lo demás.
objetos_previos <- ls()

# Parámetros ---------------------------------------------------------------
semilla <- 2026
alphas <- c(0, 0.25, 0.5, 0.75, 1)

# Paquetes y funciones -----------------------------------------------------
if (!require(pacman)) install.packages("pacman")
pacman::p_load(dplyr, Matrix, glmnet)

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
precio <- train$price
y_log <- log(precio)
rango_precio <- range(precio)

# Fold espacial de cada fila (formato foldid de cv.glmnet).
foldid_espacial <- integer(nrow(train))
for (i in seq_along(cv_folds$espacial)) {
  foldid_espacial[cv_folds$espacial[[i]]] <- i
}

# Matrices de predictores (las mismas del script 11). glmnet necesita todo
# numérico: el tipo de vivienda pasa a un indicador de casa.
fun_matriz <- function(.datos, .vars) {
  x <- .datos[.vars]
  x$property_type <- as.integer(x$property_type == "Casa")
  as.matrix(as.data.frame(lapply(x, as.numeric)))
}
fun_ampliar <- function(.x, .ruta) {
  tfidf <- readRDS(.ruta)
  colnames(tfidf) <- make.unique(paste0(
    "tfidf_", gsub("[^[:alnum:]_]", "_", colnames(tfidf))))
  cbind(Matrix::Matrix(.x, sparse = TRUE), tfidf)
}
x_completo_train <- fun_matriz(train, conjuntos_predictores$completo)
x_completo_test <- fun_matriz(test, conjuntos_predictores$completo)
x_ampliado_train <- fun_ampliar(x_completo_train,
                                conjuntos_predictores$tfidf_ampliado[["train"]])
x_ampliado_test <- fun_ampliar(x_completo_test,
                               conjuntos_predictores$tfidf_ampliado[["test"]])

# Funciones ----------------------------------------------------------------
fun_acotar <- function(.pred_log) {
  #' Pasar log(precio) a pesos y acotar al rango de precios de train
  pmin(pmax(exp(.pred_log), rango_precio[1]), rango_precio[2])
}

fun_buscar_en <- function(.x, .foldid) {
  #' Elegir lambda para cada alpha con la CV espacial
  #'
  #' @description Para cada alpha, cv.glmnet() ajusta toda la secuencia de
  #' lambdas en cada fold y guarda las predicciones fuera de muestra
  #' (keep = TRUE). Con ellas se calcula el MAE en pesos de cada lambda,
  #' que es la métrica de Kaggle (cv.glmnet solo mediría el error en log).
  #'
  #' @param .x Matriz de predictores de train (densa o dispersa).
  #' @param .foldid \code{integer}. Fold de cada fila.
  #'
  #' @returns \code{data.frame}: una fila por alpha con el mejor lambda, su
  #' MAE en CV espacial (pesos) y el número de coeficientes distintos de
  #' cero.
  resultados <- list()
  for (a in alphas) {
    inicio <- proc.time()[["elapsed"]]
    cv <- glmnet::cv.glmnet(.x, y_log, alpha = a, foldid = .foldid,
                            keep = TRUE)
    # Columnas de fit.preval: una por lambda (NA si algún fold no llegó a
    # ese lambda).
    mae <- colMeans(abs(fun_acotar(cv$fit.preval) - precio))
    k <- which.min(mae)
    resultados[[length(resultados) + 1]] <- data.frame(
      alpha = a, lambda = cv$lambda[k], mae_espacial = mae[k],
      coeficientes_no_cero = cv$nzero[k]
    )
    message("  alpha = ", a, ": MAE espacial = ", round(mae[k] / 1e6, 1),
            " millones, lambda = ", signif(cv$lambda[k], 3), ", ",
            cv$nzero[k], " coeficientes distintos de cero (",
            round(proc.time()[["elapsed"]] - inicio), " s).")
  }
  do.call(rbind, resultados)
}

fun_en_predecir <- function(.x_ajuste, .y_ajuste, .x_nuevo, .alpha,
                            .lambda) {
  #' Ajustar Elastic Net y predecir en pesos con el lambda elegido
  #'
  #' @description Se ajusta toda la secuencia de lambdas (como recomienda
  #' glmnet) y se predice en el lambda elegido.
  #'
  #' @returns \code{list} con el modelo y las predicciones en pesos.
  modelo <- glmnet::glmnet(.x_ajuste, .y_ajuste, alpha = .alpha)
  pred <- stats::predict(modelo, newx = .x_nuevo, s = .lambda)
  list(modelo = modelo, pred = fun_acotar(as.numeric(pred)))
}

# Búsqueda, validación cruzada y envíos ------------------------------------
variantes <- list(
  completo = list(train = x_completo_train, test = x_completo_test),
  ampliado = list(train = x_ampliado_train, test = x_ampliado_test)
)
oof_en <- data.frame(property_id = train$property_id)
dir.create("02_outputs/tables", recursive = TRUE, showWarnings = FALSE)

for (conjunto in names(variantes)) {
  x_tr <- variantes[[conjunto]]$train
  x_te <- variantes[[conjunto]]$test
  message("Elastic Net ", conjunto, " (", ncol(x_tr), " predictores):")
  
  # 1. Hiperparámetros con la CV espacial.
  busqueda <- fun_buscar_en(x_tr, foldid_espacial)
  write.csv(busqueda, paste0("02_outputs/tables/EN_", conjunto,
                             "_busqueda.csv"), row.names = FALSE)
  mejor <- busqueda[which.min(busqueda$mae_espacial), ]
  
  nombre <- paste0("EN_", conjunto,
                   "_alpha_", fun_hiperparam_str(mejor$alpha, 2),
                   "_lambda_", fun_hiperparam_str(mejor$lambda, 5))
  descripcion <- paste0("alpha=", mejor$alpha, "; lambda=",
                        signif(mejor$lambda, 3), "; coeficientes no cero=",
                        mejor$coeficientes_no_cero)
  
  # 2. Error fuera de muestra con cada tipo de CV.
  for (tipo in c("espacial", "aleatoria")) {
    oof <- rep(NA_real_, nrow(train))
    for (val in cv_folds[[tipo]]) {
      oof[val] <- fun_en_predecir(x_tr[-val, , drop = FALSE], y_log[-val],
                                  x_tr[val, , drop = FALSE],
                                  mejor$alpha, mejor$lambda)$pred
    }
    fun_registrar_cv(
      .archivo = nombre, .algoritmo = "Elastic Net", .conjunto = conjunto,
      .n_predictores = ncol(x_tr), .hiperparametros = descripcion,
      .tipo_cv = tipo, .obs = precio, .pred = oof
    )
    oof_en[[paste0("EN_", conjunto, "_", tipo)]] <- oof
    message("  CV ", tipo, ": MAE = ",
            round(mean(abs(oof - precio)) / 1e6, 1), " millones.")
  }
  
  # 3. Modelo final con todo train, envío y coeficientes.
  final <- fun_en_predecir(x_tr, y_log, x_te, mejor$alpha, mejor$lambda)
  fun_save_submission(.ids = test$property_id, .pred = final$pred,
                      .nombre = nombre)
  coeficientes <- as.matrix(stats::coef(final$modelo, s = mejor$lambda))
  coeficientes <- data.frame(variable = rownames(coeficientes),
                             coeficiente = coeficientes[, 1]) |>
    filter(coeficiente != 0, variable != "(Intercept)") |>
    arrange(desc(abs(coeficiente)))
  write.csv(coeficientes, paste0("02_outputs/tables/EN_", conjunto,
                                 "_coeficientes.csv"), row.names = FALSE)
}

saveRDS(oof_en, "03_temp/oof_en.rds")

# Resumen ------------------------------------------------------------------
read.csv("02_outputs/tables/registro_modelos.csv") |>
  filter(algoritmo == "Elastic Net") |>
  transmute(archivo = substr(archivo, 1, 30), tipo_cv,
            mae_millones = round(mae / 1e6, 1),
            mae_similares_chapinero = round(mae_similares_chapinero / 1e6, 1),
            sesgo_millones = round(sesgo_medio / 1e6, 1),
            sesgo_similares_chapinero = round(sesgo_similares_chapinero / 1e6, 1),
            pct_sobreestima = round(pct_sobreestima, 1)) |>
  print(row.names = FALSE)

# Limpiamos el entorno: borramos solo los objetos que creó este script.
rm(list = setdiff(ls(), objetos_previos))
