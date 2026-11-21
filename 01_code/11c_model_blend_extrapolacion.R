##########################################################
# 11c_model_blend_extrapolacion.R
#
# Objetivo: mejorar el LightGBM en Chapinero. El árbol se queda corto
# cuando la vivienda está más cerca del empleo de lo que vio en train,
# porque en el borde repite el último precio. Aquí se mezcla con un
# Elastic Net, que sí extrapola esa pendiente, y se corrige el sesgo
# medido en las viviendas parecidas a Chapinero.
#
# Etapa del Problem Set: Etapa 2 (extensión sobre Boosting y Elastic Net)
# y Etapa 4 (corregir el error en Chapinero).
# Insumos:  03_temp/train_model.rds, test_model.rds,
#           conjuntos_predictores.rds (script 05); cv_folds.rds y
#           similitud_chapinero.rds (script 06, k-means de 20 regiones);
#           02_outputs/tables/LGBM_completo_ensayos.csv (script 11).
# Salidas:  02_outputs/predictions/BLEND_completo_....csv (envío).
#           02_outputs/tables/BLEND_completo_busqueda.csv: pesos, sesgo
#           y MAE en las parecidas a Chapinero.
#           Filas en 02_outputs/tables/registro_modelos.csv.
#           03_temp/oof_blend.rds (predicciones fuera de muestra).
#
# Decisiones:
#   - Extensión. Las piezas son del curso: LightGBM L1 (script 11) y
#     Elastic Net sobre el logaritmo (script 08). La mezcla y la
#     corrección de sesgo no están en los cuadernos.
#   - Los hiperparámetros del LightGBM no se vuelven a buscar: se usa el
#     ensayo que ya está en Kaggle (lr 0,017, 66 hojas, profundidad 7,
#     mínimo 23, 4.123 rondas). Las rondas del precio por m² sí se eligen
#     con la CV espacial, como en el script 11b.
#   - Elastic Net: alpha en {0; 0,5; 1} y el lambda de glmnet. Se elige
#     con la CV espacial, minimizando el MAE en pesos. Hay uno para el
#     precio total y otro para el precio por m².
#   - Donde la superficie es real: la predicción es el precio por m²
#     (mezcla de los dos modelos) por la superficie. Donde fue imputada,
#     la mezcla es sobre el precio total. El peso de cada mezcla se elige
#     con la CV espacial, minimizando el MAE en las parecidas a Chapinero,
#     que es la columna que ordena los modelos igual que Kaggle.
#   - Corrección: en cada fold espacial, a las parecidas a Chapinero se
#     les suma la mediana del residuo (observado - predicho) de las
#     parecidas de los demás folds. Eso quita el sesgo de quedar cortos
#     sin usar las etiquetas del propio fold. En el test, que es
#     Chapinero, la corrección se aplica a todas las filas.
#   - Si el modelo va en logaritmo, se vuelve a pesos con exp() antes de
#     medir el error y de guardar. Todo se acota al rango de train.
##########################################################

# Objetos que ya existían en el entorno antes de correr este script. Al
# final se borra solo lo que este script creó, sin tocar lo demás.
objetos_previos <- ls()

# Parámetros ---------------------------------------------------------------
semilla <- 2026
hilos <- 6L
max_rondas_m2 <- 10000L
paciencia <- 100L
alphas <- c(0, 0.5, 1)
grilla_pesos <- seq(0, 1, by = 0.05)

# Paquetes y funciones -----------------------------------------------------
if (!require(pacman)) install.packages("pacman")
pacman::p_load(dplyr, lightgbm, glmnet)

source("01_code/functions/fun_save_submission.R")
source("01_code/functions/fun_metricas.R")

# Datos --------------------------------------------------------------------
train <- readRDS("03_temp/train_model.rds")
test <- readRDS("03_temp/test_model.rds")
conjuntos_predictores <- readRDS("03_temp/conjuntos_predictores.rds")
cv_folds <- readRDS("03_temp/cv_folds.rds")
similitud <- readRDS("03_temp/similitud_chapinero.rds")

if (!identical(similitud$property_id, train$property_id)) {
  stop("La similitud no está en el mismo orden que train_model.rds.")
}
if (!all(sort(unlist(cv_folds$espacial)) == seq_len(nrow(train)))) {
  stop("Los folds no coinciden con train_model.rds. Corra de nuevo los ",
       "scripts 05 y 06.")
}
if (!identical(cv_folds$esquema_espacial, "k-means 20 regiones")) {
  stop("cv_folds$espacial es '", cv_folds$esquema_espacial,
       "'. Este modelo usa k-means de 20 regiones.")
}

precio <- train$price
rango_precio <- range(precio)
n <- nrow(train)
parecidas <- similitud$similitud >= stats::quantile(similitud$similitud, 0.9)
fold_espacial <- integer(n)
for (i in seq_along(cv_folds$espacial)) {
  fold_espacial[cv_folds$espacial[[i]]] <- i
}

fun_matriz <- function(.datos, .vars) {
  x <- .datos[.vars]
  x$property_type <- as.integer(x$property_type == "Casa")
  as.matrix(as.data.frame(lapply(x, as.numeric)))
}
x_train <- fun_matriz(train, conjuntos_predictores$completo)
x_test <- fun_matriz(test, conjuntos_predictores$completo)

fun_superficie <- function(.datos) {
  #' Superficie real, o NA si las dos fueron imputadas
  conocida <- function(.v) {
    indicador <- paste0("falta_", .v)
    if (indicador %in% names(.datos)) .datos[[indicador]] == 0 else
      rep(TRUE, nrow(.datos))
  }
  dplyr::case_when(
    conocida("surface_covered_completa") &
      .datos$surface_covered_completa > 0 ~ .datos$surface_covered_completa,
    conocida("surface_total_completa") &
      .datos$surface_total_completa > 0 ~ .datos$surface_total_completa,
    TRUE ~ NA_real_
  )
}
superficie_train <- fun_superficie(train)
superficie_test <- fun_superficie(test)
conocida <- !is.na(superficie_train)
filas_m2 <- which(conocida)
message("Superficie conocida: ", round(100 * mean(conocida), 1),
        "% de train y ", round(100 * mean(!is.na(superficie_test)), 1),
        "% de test.")

fun_acotar <- function(.pred) {
  pmin(pmax(.pred, rango_precio[1]), rango_precio[2])
}

# LightGBM: el ensayo que ya está en Kaggle --------------------------------
ensayos <- read.csv("02_outputs/tables/LGBM_completo_ensayos.csv")
ganador <- ensayos[ensayos$rondas == 4123 & ensayos$num_leaves == 66, ]
if (nrow(ganador) != 1) {
  stop("No encontré el ensayo del LightGBM de Kaggle (66 hojas, 4123 rondas).")
}
fun_armar_params <- function(.fila) {
  list(
    objective = "regression_l1", metric = "l1",
    learning_rate = exp(.fila$log_lr),
    num_leaves = as.integer(.fila$num_leaves),
    max_depth = as.integer(.fila$max_depth),
    min_data_in_leaf = as.integer(.fila$min_data_in_leaf),
    feature_fraction = .fila$feature_fraction,
    bagging_fraction = .fila$bagging_fraction, bagging_freq = 1L,
    lambda_l1 = expm1(.fila$log_l1), lambda_l2 = expm1(.fila$log_l2),
    num_threads = hilos, seed = semilla, verbosity = -1L
  )
}
params <- fun_armar_params(ganador)
rondas_precio <- as.integer(ganador$rondas)

# 1. Predicciones fuera de muestra del LightGBM ----------------------------
ruta_lgbm <- "03_temp/blend_lgbm_oof.rds"
if (file.exists(ruta_lgbm)) {
  lgbm <- readRDS(ruta_lgbm)
  rondas_m2 <- lgbm$rondas_m2
  message("LightGBM: se reutilizan las predicciones guardadas (",
          rondas_m2, " rondas de precio por m²).")
} else {
  folds_m2 <- lapply(cv_folds$espacial, function(.val) {
    match(intersect(.val, filas_m2), filas_m2)
  })
  cv_m2 <- lightgbm::lgb.cv(
    params = params,
    data = lightgbm::lgb.Dataset(x_train[filas_m2, , drop = FALSE],
                                 label = precio[filas_m2] /
                                   superficie_train[filas_m2]),
    folds = folds_m2, nrounds = max_rondas_m2,
    early_stopping_rounds = paciencia, verbose = -1L
  )
  rondas_m2 <- as.integer(cv_m2$best_iter)
  message("Rondas del precio por m²: ", rondas_m2, ".")

  fun_oof_lgbm <- function(.y, .filas, .folds, .rondas) {
    #' Precio predicho fuera de muestra. NA fuera de .filas.
    oof <- rep(NA_real_, n)
    for (val in .folds) {
      val_uso <- intersect(val, .filas)
      ajuste <- setdiff(.filas, val)
      modelo <- lightgbm::lgb.train(
        params = params,
        data = lightgbm::lgb.Dataset(x_train[ajuste, , drop = FALSE],
                                     label = .y[ajuste]),
        nrounds = .rondas, verbose = -1L
      )
      oof[val_uso] <- stats::predict(modelo, x_train[val_uso, , drop = FALSE])
    }
    oof
  }
  y_m2 <- rep(NA_real_, n)
  y_m2[filas_m2] <- precio[filas_m2] / superficie_train[filas_m2]
  lgbm <- list(
    precio_espacial = fun_acotar(fun_oof_lgbm(precio, seq_len(n),
                                              cv_folds$espacial, rondas_precio)),
    precio_aleatoria = fun_acotar(fun_oof_lgbm(precio, seq_len(n),
                                               cv_folds$aleatoria, rondas_precio)),
    m2_espacial = fun_oof_lgbm(y_m2, filas_m2, cv_folds$espacial, rondas_m2),
    m2_aleatoria = fun_oof_lgbm(y_m2, filas_m2, cv_folds$aleatoria, rondas_m2),
    rondas_m2 = rondas_m2
  )
  # El precio por m² pasa a pesos con la superficie real.
  lgbm$m2_espacial <- fun_acotar(lgbm$m2_espacial * superficie_train)
  lgbm$m2_aleatoria <- fun_acotar(lgbm$m2_aleatoria * superficie_train)
  saveRDS(lgbm, ruta_lgbm)
  message("LightGBM fuera de muestra guardado.")
}

# 2. Elastic Net: log(precio) y log(precio por m²) -------------------------
ruta_en <- "03_temp/blend_en_oof.rds"
if (file.exists(ruta_en)) {
  en <- readRDS(ruta_en)
  message("Elastic Net: se reutilizan las predicciones guardadas.")
} else {
  fun_buscar_en <- function(.x, .y_log, .a_pesos, .foldid) {
    #' Alpha y lambda con menor MAE en pesos, y la predicción espacial
    #'
    #' @param .a_pesos Función que pasa la predicción en logaritmo a pesos.
    #' @returns \code{list} con alpha, lambda y el vector fuera de muestra.
    mejor <- NULL
    for (a in alphas) {
      cv <- glmnet::cv.glmnet(.x, .y_log, alpha = a, foldid = .foldid,
                              keep = TRUE)
      pesos <- .a_pesos(cv$fit.preval)
      mae <- apply(pesos, 2, function(p) mean(abs(p - attr(.a_pesos, "obs"))))
      k <- which.min(mae)
      if (is.null(mejor) || mae[k] < mejor$mae) {
        mejor <- list(alpha = a, lambda = cv$lambda[k], mae = mae[k],
                      oof = pesos[, k])
      }
      message("  alpha ", a, ": MAE ", round(mae[k] / 1e6, 1),
              " millones, lambda ", signif(cv$lambda[k], 3), ".")
    }
    mejor
  }
  fun_oof_en_fijo <- function(.x, .y_log, .folds, .alpha, .lambda, .a_pesos) {
    #' Misma penalización, ahora con la otra CV (no se vuelve a elegir)
    oof <- rep(NA_real_, length(.y_log))
    for (val in .folds) {
      modelo <- glmnet::glmnet(.x[-val, , drop = FALSE], .y_log[-val],
                               alpha = .alpha)
      log_pred <- as.numeric(stats::predict(modelo, newx = .x[val, , drop = FALSE],
                                            s = .lambda))
      oof[val] <- .a_pesos(log_pred)
    }
    oof
  }

  message("Elastic Net del precio total:")
  a_precio <- function(.log) fun_acotar(exp(.log))
  attr(a_precio, "obs") <- precio
  en_precio <- fun_buscar_en(x_train, log(precio), a_precio, fold_espacial)

  message("Elastic Net del precio por m²:")
  a_m2 <- function(.log) {
    fun_acotar(exp(.log) * superficie_train[filas_m2])
  }
  attr(a_m2, "obs") <- precio[filas_m2]
  en_m2 <- fun_buscar_en(
    x_train[filas_m2, , drop = FALSE],
    log(precio[filas_m2] / superficie_train[filas_m2]),
    a_m2, fold_espacial[filas_m2]
  )
  oof_m2_esp <- rep(NA_real_, n)
  oof_m2_esp[filas_m2] <- en_m2$oof

  # CV aleatoria con el alpha y el lambda ya elegidos.
  a_precio_fila <- function(.log) fun_acotar(exp(.log))
  oof_precio_al <- fun_oof_en_fijo(
    x_train, log(precio), cv_folds$aleatoria,
    en_precio$alpha, en_precio$lambda, a_precio_fila
  )
  oof_m2_al <- rep(NA_real_, n)
  for (val in cv_folds$aleatoria) {
    ajuste <- setdiff(filas_m2, val)
    val_uso <- intersect(val, filas_m2)
    modelo <- glmnet::glmnet(
      x_train[ajuste, , drop = FALSE],
      log(precio[ajuste] / superficie_train[ajuste]),
      alpha = en_m2$alpha
    )
    log_m2 <- as.numeric(stats::predict(
      modelo, newx = x_train[val_uso, , drop = FALSE], s = en_m2$lambda
    ))
    oof_m2_al[val_uso] <- fun_acotar(exp(log_m2) * superficie_train[val_uso])
  }

  en <- list(
    precio_espacial = en_precio$oof, precio_aleatoria = oof_precio_al,
    m2_espacial = oof_m2_esp, m2_aleatoria = oof_m2_al,
    alpha_precio = en_precio$alpha, lambda_precio = en_precio$lambda,
    alpha_m2 = en_m2$alpha, lambda_m2 = en_m2$lambda
  )
  saveRDS(en, ruta_en)
  message("Elastic Net fuera de muestra guardado.")
}

# 3. Pesos de la mezcla, con las parecidas a Chapinero ---------------------
fun_mezclar <- function(.lgbm_precio, .en_precio, .lgbm_m2, .en_m2,
                        .w_m2, .w_precio) {
  #' Mezcla en pesos, ya acotada al rango de train
  pred <- rep(NA_real_, n)
  pred[conocida] <- (1 - .w_m2) * .lgbm_m2[conocida] +
    .w_m2 * .en_m2[conocida]
  pred[!conocida] <- (1 - .w_precio) * .lgbm_precio[!conocida] +
    .w_precio * .en_precio[!conocida]
  fun_acotar(pred)
}
fun_mae_en <- function(.pred, .filas) {
  mean(abs(.pred[.filas] - precio[.filas]))
}
filas_m2_parecidas <- which(parecidas & conocida)
filas_precio_parecidas <- which(parecidas & !conocida)

mae_m2 <- vapply(grilla_pesos, function(w) {
  pred <- fun_mezclar(lgbm$precio_espacial, en$precio_espacial,
                      lgbm$m2_espacial, en$m2_espacial, w, 0)
  fun_mae_en(pred, filas_m2_parecidas)
}, numeric(1))
w_m2 <- grilla_pesos[which.min(mae_m2)]

mae_precio <- vapply(grilla_pesos, function(w) {
  pred <- fun_mezclar(lgbm$precio_espacial, en$precio_espacial,
                      lgbm$m2_espacial, en$m2_espacial, w_m2, w)
  fun_mae_en(pred, filas_precio_parecidas)
}, numeric(1))
w_precio <- grilla_pesos[which.min(mae_precio)]
message("Pesos del Elastic Net: precio por m² ", w_m2,
        ", precio total ", w_precio, ".")

# 4. Corrección del sesgo, sin usar el fold que se evalúa ------------------
fun_corregir <- function(.pred, .folds) {
  #' Sumar, solo en las parecidas, la mediana del residuo de los otros folds
  #'
  #' @returns \code{list} con la predicción corregida y el sesgo aplicado
  #' en el test (mediana sobre todas las parecidas).
  corregida <- .pred
  for (val in .folds) {
    otros <- parecidas
    otros[val] <- FALSE
    ajuste <- stats::median(precio[otros] - .pred[otros])
    filas <- val[parecidas[val]]
    corregida[filas] <- .pred[filas] + ajuste
  }
  list(pred = fun_acotar(corregida),
       sesgo = stats::median(precio[parecidas] - .pred[parecidas]))
}

pred_espacial <- fun_mezclar(lgbm$precio_espacial, en$precio_espacial,
                             lgbm$m2_espacial, en$m2_espacial,
                             w_m2, w_precio)
corr_espacial <- fun_corregir(pred_espacial, cv_folds$espacial)

pred_aleatoria <- fun_mezclar(lgbm$precio_aleatoria, en$precio_aleatoria,
                              lgbm$m2_aleatoria, en$m2_aleatoria,
                              w_m2, w_precio)
# El sesgo es el de la CV espacial. En la aleatoria se aplica el mismo
# número, solo a las parecidas a Chapinero.
pred_aleatoria_corr <- pred_aleatoria
pred_aleatoria_corr[parecidas] <- pred_aleatoria[parecidas] + corr_espacial$sesgo
pred_aleatoria_corr <- fun_acotar(pred_aleatoria_corr)

fun_texto <- function(.x, .decimales) {
  gsub(".", "_", sprintf(paste0("%.", .decimales, "f"), .x), fixed = TRUE)
}
sesgo_millones <- round(corr_espacial$sesgo / 1e6)
nombre <- paste0("BLEND_completo_wm2_", fun_texto(w_m2, 2),
                 "_wp_", fun_texto(w_precio, 2),
                 "_sesgo_", sesgo_millones)
descripcion <- paste0(
  "Extension: mezcla LGBM y Elastic Net; w m2=", w_m2,
  "; w precio=", w_precio,
  "; EN precio alpha=", en$alpha_precio, " lambda=", signif(en$lambda_precio, 3),
  "; EN m2 alpha=", en$alpha_m2, " lambda=", signif(en$lambda_m2, 3),
  "; sesgo parecidas=", round(corr_espacial$sesgo / 1e6, 1), " millones",
  "; rondas m2=", lgbm$rondas_m2
)

# 5. Registro con las dos CV -----------------------------------------------
oof_blend <- data.frame(property_id = train$property_id)
for (tipo in c("espacial", "aleatoria")) {
  oof <- if (tipo == "espacial") corr_espacial$pred else pred_aleatoria_corr
  fun_registrar_cv(
    .archivo = nombre, .algoritmo = "Híbrido LGBM-EN",
    .conjunto = "completo", .n_predictores = ncol(x_train),
    .hiperparametros = descripcion, .tipo_cv = tipo,
    .obs = precio, .pred = oof
  )
  oof_blend[[paste0("BLEND_completo_", tipo)]] <- oof
  message("CV ", tipo, ": MAE = ",
          round(mean(abs(oof - precio)) / 1e6, 1), " millones.")
}
saveRDS(oof_blend, "03_temp/oof_blend.rds")

busqueda <- data.frame(
  w_m2 = w_m2, w_precio = w_precio,
  sesgo_millones = corr_espacial$sesgo / 1e6,
  alpha_precio = en$alpha_precio, lambda_precio = en$lambda_precio,
  alpha_m2 = en$alpha_m2, lambda_m2 = en$lambda_m2,
  rondas_m2 = lgbm$rondas_m2,
  mae_parecidas_sin_correccion = fun_mae_en(pred_espacial, which(parecidas)),
  mae_parecidas_con_correccion = fun_mae_en(corr_espacial$pred, which(parecidas))
)
dir.create("02_outputs/tables", recursive = TRUE, showWarnings = FALSE)
write.csv(busqueda, "02_outputs/tables/BLEND_completo_busqueda.csv",
          row.names = FALSE)

# 6. Modelo final y envío --------------------------------------------------
modelo_precio <- lightgbm::lgb.train(
  params = params,
  data = lightgbm::lgb.Dataset(x_train, label = precio),
  nrounds = rondas_precio, verbose = -1L
)
modelo_m2 <- lightgbm::lgb.train(
  params = params,
  data = lightgbm::lgb.Dataset(x_train[filas_m2, , drop = FALSE],
                               label = precio[filas_m2] /
                                 superficie_train[filas_m2]),
  nrounds = lgbm$rondas_m2, verbose = -1L
)
modelo_en_precio <- glmnet::glmnet(x_train, log(precio), alpha = en$alpha_precio)
modelo_en_m2 <- glmnet::glmnet(
  x_train[filas_m2, , drop = FALSE],
  log(precio[filas_m2] / superficie_train[filas_m2]),
  alpha = en$alpha_m2
)
pred_lgbm_precio <- fun_acotar(stats::predict(modelo_precio, x_test))
pred_en_precio <- fun_acotar(exp(as.numeric(stats::predict(
  modelo_en_precio, newx = x_test, s = en$lambda_precio
))))
pred_lgbm_m2 <- fun_acotar(stats::predict(modelo_m2, x_test) * superficie_test)
pred_en_m2 <- fun_acotar(exp(as.numeric(stats::predict(
  modelo_en_m2, newx = x_test, s = en$lambda_m2
))) * superficie_test)
conocida_test <- !is.na(superficie_test)
pred_test <- rep(NA_real_, nrow(test))
pred_test[conocida_test] <- (1 - w_m2) * pred_lgbm_m2[conocida_test] +
  w_m2 * pred_en_m2[conocida_test]
pred_test[!conocida_test] <- (1 - w_precio) * pred_lgbm_precio[!conocida_test] +
  w_precio * pred_en_precio[!conocida_test]
# El test es Chapinero: la corrección aplica a todas las filas.
pred_test <- fun_acotar(pred_test + corr_espacial$sesgo)
fun_save_submission(.ids = test$property_id, .pred = pred_test, .nombre = nombre)

# Resumen ------------------------------------------------------------------
registro <- read.csv("02_outputs/tables/registro_modelos.csv")
registro |>
  filter(algoritmo == "Híbrido LGBM-EN") |>
  transmute(archivo = substr(archivo, 1, 42), tipo_cv,
            mae_millones = round(mae / 1e6, 1),
            mae_similares_chapinero = round(mae_similares_chapinero / 1e6, 1),
            sesgo_millones = round(sesgo_medio / 1e6, 1),
            sesgo_similares_chapinero = round(sesgo_similares_chapinero / 1e6, 1)) |>
  print(row.names = FALSE)
referencia <- registro$mae_similares_chapinero[
  registro$archivo == "LGBM_completo_lr_0_017_hojas_66_prof_7_minhoja_23_rondas_4123" &
    registro$tipo_cv == "espacial"]
hibrido <- registro$mae_similares_chapinero[
  registro$archivo == nombre & registro$tipo_cv == "espacial"]
if (length(referencia) == 1 && length(hibrido) == 1) {
  message("MAE en similares a Chapinero (CV espacial): híbrido ",
          round(hibrido / 1e6, 1), " millones; LightGBM ",
          round(referencia / 1e6, 1), " millones.")
}

# Limpiamos el entorno: borramos solo los objetos que creó este script.
rm(list = setdiff(ls(), objetos_previos))
