##########################################################
# 12_model_neural_network.R
#
# Objetivo: estimar una red neuronal sobre el conjunto completo, elegir su
# arquitectura con la CV espacial, medir el error con CV espacial y
# aleatoria, y generar el envío a Kaggle.
#
# Etapa del Problem Set: Etapa 2 (familia Neural Networks) y Etapa 3.
# Insumos:  03_temp/train_model.rds, test_model.rds,
#           conjuntos_predictores.rds (script 05) y cv_folds.rds (script 06,
#           esquema k-means de 20 regiones).
# Salidas:  02_outputs/predictions/NN_completo_....csv (envío).
#           02_outputs/tables/NN_completo_busqueda.csv: MAE en CV espacial
#           de cada arquitectura.
#           Filas en 02_outputs/tables/registro_modelos.csv.
#           03_temp/oof_nn.rds (predicciones fuera de muestra).
#
# Decisiones:
#   - Molde del curso (cuaderno 02_NN_example_regression): keras secuencial,
#     capas densas y la tarea del cuaderno de probar mean_absolute_error
#     además del MSE. El ejemplo del cuaderno es una red de juguete (2
#     neuronas sigmoides, SGD, el lote completo) con 2 predictores. Con 174
#     predictores esa red no aprende; lo que sigue es la extensión.
#   - Extensión: perceptrón con bloques densos (inicialización He, batch
#     normalization, ReLU y dropout), optimizador Adam y un atajo lineal
#     desde la entrada hasta la salida. El atajo puede reproducir un modelo
#     lineal (el de la familia 07) y las capas ocultas aprenden lo que ese
#     modelo no captura. La pérdida es el error absoluto.
#   - Se predice el precio en pesos, como el LightGBM de referencia
#     (objective regression_l1), no el logaritmo. El precio se estandariza
#     dentro de cada ajuste para que la tasa de aprendizaje no dependa de
#     los millones; al predecir se deshace esa estandarización. No hace
#     falta exp().
#   - Los predictores se estandarizan con la media y la desviación del
#     ajuste de ese fold (no con las de todo train).
#   - Hiperparámetros (profundidad, ancho y dropout) elegidos con la CV
#     espacial, minimizando el MAE en pesos. La tasa de aprendizaje queda
#     fija en 0,001. Durante la búsqueda, el número de épocas se decide con
#     early stopping sobre un 15% del ajuste, agrupado por coordenadas para
#     no partir un mismo edificio. Ese 15% no es el fold espacial: el fold
#     espacial solo se usa para medir el MAE. Para el registro y el envío se
#     reentrena, con las épocas ya elegidas, sobre todo el fold.
#   - Las predicciones se acotan al rango de precios de train (300 a 1.650
#     millones).
#   - Si NN_completo_busqueda.csv ya tiene todas las arquitecturas de la
#     grilla, no se repite la búsqueda.
##########################################################

# Objetos que ya existían en el entorno antes de correr este script. Al
# final se borra solo lo que este script creó, sin tocar lo demás.
objetos_previos <- ls()

# Parámetros ---------------------------------------------------------------
semilla <- 2026
tasa_aprendizaje <- 0.001
batch <- 512L
max_epocas <- 60L
paciencia <- 8L
epocas_minimas <- 8L
prop_monitoreo <- 0.15
buscar_hiperparametros <- TRUE

# Grilla: ancho de cada capa oculta y dropout. La tasa de aprendizaje no
# entra a la grilla (Adam a 0,001 sobre datos estandarizados).
grilla <- list(
  list(capas = c(128L, 64L), dropout = 0.10),
  list(capas = c(256L, 128L, 64L), dropout = 0.10),
  list(capas = c(256L, 128L, 64L), dropout = 0.30),
  list(capas = c(256L, 128L, 64L, 32L), dropout = 0.20)
)

# Paquetes y funciones -----------------------------------------------------
# Keras usa Python a través de reticulate. Si R no encuentra el entorno
# con TensorFlow, indique su ruta antes de correr el script, por ejemplo:
# Sys.setenv(RETICULATE_PYTHON = "~/.virtualenvs/r-keras/bin/python")
Sys.setenv(KERAS_BACKEND = "tensorflow", TF_CPP_MIN_LOG_LEVEL = "2")

if (!require(pacman)) install.packages("pacman")
pacman::p_load(dplyr, keras3)

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
if (!identical(cv_folds$esquema_espacial, "k-means 20 regiones")) {
  stop("cv_folds$espacial es '", cv_folds$esquema_espacial,
       "', y el LightGBM de referencia se afinó con k-means de 20 regiones. ",
       "Vuelva a crear los folds con ese esquema (script 06).")
}

vars <- conjuntos_predictores$completo
fun_matriz <- function(.datos, .vars) {
  x <- .datos[.vars]
  x$property_type <- as.integer(x$property_type == "Casa")
  as.matrix(as.data.frame(lapply(x, as.numeric)))
}
x_train <- fun_matriz(train, vars)
x_test <- fun_matriz(test, vars)
precio <- train$price
rango_precio <- range(precio)
message("Red neuronal: conjunto completo, ", ncol(x_train), " predictores.")

# Funciones ----------------------------------------------------------------
fun_id_arquitectura <- function(.capas, .dropout) {
  #' Identificador corto de una arquitectura, para la tabla y el archivo
  paste0("c", paste(.capas, collapse = "_"), "_d",
         sprintf("%.2f", .dropout))
}

fun_texto_archivo <- function(.x, .decimales) {
  #' Número sin punto decimal, para el nombre del envío
  gsub(".", "_", sprintf(paste0("%.", .decimales, "f"), .x), fixed = TRUE)
}

fun_estadisticos <- function(.x, .y) {
  #' Media y desviación para estandarizar, calculadas solo en el ajuste
  #'
  #' @returns \code{list} con centro y escala de predictores y del precio.
  escala_x <- apply(.x, 2, stats::sd)
  escala_x[!is.finite(escala_x) | escala_x < 1e-8] <- 1
  list(centro_x = colMeans(.x), escala_x = escala_x,
       centro_y = mean(.y), escala_y = stats::sd(.y))
}

fun_aplicar_escala <- function(.x, .estadisticos) {
  #' Estandarizar predictores con estadísticas ya calculadas
  sweep(sweep(.x, 2, .estadisticos$centro_x, "-"), 2,
        .estadisticos$escala_x, "/")
}

fun_a_pesos <- function(.pred_z, .estadisticos) {
  #' Deshacer la estandarización del precio y acotar al rango de train
  pred <- .pred_z * .estadisticos$escala_y + .estadisticos$centro_y
  pmin(pmax(pred, rango_precio[1]), rango_precio[2])
}

fun_partir_monitoreo <- function(.filas, .prop, .semilla) {
  #' Apartar grupos de coordenadas para el early stopping
  #'
  #' @description El monitoreo del early stopping sale del propio ajuste,
  #' nunca del fold espacial. Las propiedades con la misma coordenada (el
  #' mismo edificio) van juntas, para que la red no vea en el ajuste el
  #' precio de un anuncio y lo "prediga" en el monitoreo.
  #'
  #' @param .filas \code{integer}. Filas disponibles para ajustar.
  #' @param .prop \code{numeric}. Fracción de grupos que va a monitoreo.
  #' @param .semilla \code{integer}.
  #'
  #' @returns \code{list} con índices de ajuste y de monitoreo.
  clave <- paste(round(train$lat[.filas], 5), round(train$lon[.filas], 5))
  grupos <- unique(clave)
  set.seed(.semilla)
  n_monitoreo <- max(1L, floor(.prop * length(grupos)))
  grupos_monitoreo <- sample(grupos, n_monitoreo)
  es_monitoreo <- clave %in% grupos_monitoreo
  list(ajuste = .filas[!es_monitoreo], monitoreo = .filas[es_monitoreo])
}

fun_crear_red <- function(.p, .capas, .dropout) {
  #' Armar la red: bloques densos más un atajo lineal
  #'
  #' @description Cada bloque es una capa densa, batch normalization, ReLU
  #' y dropout. El atajo lineal suma a la salida una función lineal de la
  #' entrada, así la red puede quedarse cerca de un modelo lineal si las
  #' capas ocultas no aportan.
  #'
  #' @param .p \code{integer}. Número de predictores.
  #' @param .capas \code{integer}. Unidades de cada capa oculta.
  #' @param .dropout \code{numeric}. Fracción de unidades apagadas.
  #'
  #' @returns Modelo de Keras, sin compilar.
  entrada <- layer_input(shape = c(.p))
  h <- entrada
  for (unidades in .capas) {
    h <- h |>
      layer_dense(units = unidades, kernel_initializer = "he_normal") |>
      layer_batch_normalization() |>
      layer_activation_relu() |>
      layer_dropout(rate = .dropout)
  }
  cabeza <- h |> layer_dense(units = 1)
  atajo <- entrada |> layer_dense(units = 1)
  keras_model(inputs = entrada, outputs = layer_add(list(cabeza, atajo)))
}

fun_mejor_epoca <- function(.historia) {
  #' Época (desde 1) con menor error absoluto en el monitoreo
  #'
  #' @description No se miran las épocas anteriores a \code{epocas_minimas}:
  #' el early stopping tampoco las usa para detenerse. Sin monitoreo (el
  #' reentrenamiento final) no hay val_loss y se devuelven las épocas pedidas.
  metricas <- .historia$metrics
  if (is.null(metricas) || is.null(metricas$val_loss)) return(epocas_minimas)
  perdidas <- as.numeric(metricas$val_loss)
  if (!length(perdidas) || all(!is.finite(perdidas))) return(epocas_minimas)
  desde <- min(epocas_minimas, length(perdidas))
  desde - 1L + which.min(perdidas[desde:length(perdidas)])
}

fun_entrenar_predecir <- function(.x_ajuste, .y_ajuste, .x_nuevo, .capas,
                                  .dropout, .epocas, .semilla,
                                  .x_monitoreo = NULL, .y_monitoreo = NULL) {
  #' Entrenar la red y predecir en pesos
  #'
  #' @param .x_ajuste,.x_nuevo Matrices numéricas ya estandarizadas.
  #' @param .y_ajuste \code{numeric}. Precio estandarizado.
  #' @param .epocas \code{integer}. Máximo de épocas. Si hay monitoreo, el
  #' early stopping puede devolver una época anterior.
  #' @param .x_monitoreo,.y_monitoreo Datos para el early stopping. NULL en
  #' el reentrenamiento final, que ya trae las épocas elegidas.
  #'
  #' @returns \code{list} con predicciones estandarizadas, la época usada y
  #' el modelo.
  set_random_seed(.semilla)
  clear_session()
  modelo <- fun_crear_red(ncol(.x_ajuste), .capas, .dropout)
  modelo |> compile(
    optimizer = optimizer_adam(learning_rate = tasa_aprendizaje),
    loss = "mean_absolute_error",
    jit_compile = FALSE
  )
  argumentos <- list(
    object = modelo, x = .x_ajuste, y = .y_ajuste,
    epochs = as.integer(.epocas), batch_size = batch, verbose = 0
  )
  if (!is.null(.x_monitoreo)) {
    argumentos$validation_data <- list(.x_monitoreo, .y_monitoreo)
    argumentos$callbacks <- list(callback_early_stopping(
      monitor = "val_loss", patience = paciencia,
      restore_best_weights = TRUE, start_from_epoch = epocas_minimas,
      verbose = 0
    ))
  }
  historia <- do.call(fit, argumentos)
  pred_z <- as.numeric(as.array(predict(modelo, .x_nuevo, verbose = 0)))
  list(pred_z = pred_z, epoca = fun_mejor_epoca(historia), modelo = modelo)
}

# 1. Búsqueda de arquitectura (CV espacial) --------------------------------
ruta_busqueda <- "02_outputs/tables/NN_completo_busqueda.csv"
dir.create("02_outputs/tables", recursive = TRUE, showWarnings = FALSE)

busqueda <- if (file.exists(ruta_busqueda)) {
  read.csv(ruta_busqueda, stringsAsFactors = FALSE)
} else {
  data.frame()
}

if (buscar_hiperparametros) {
  for (candidato in grilla) {
    id <- fun_id_arquitectura(candidato$capas, candidato$dropout)
    if (nrow(busqueda) > 0 && id %in% busqueda$id) {
      message("Arquitectura ", id, " ya está en la búsqueda. Se omite.")
      next
    }
    message("Arquitectura ", id, ":")
    mae_folds <- numeric(length(cv_folds$espacial))
    epocas_folds <- integer(length(cv_folds$espacial))
    inicio <- proc.time()[["elapsed"]]
    for (i in seq_along(cv_folds$espacial)) {
      val <- cv_folds$espacial[[i]]
      partes <- fun_partir_monitoreo(setdiff(seq_len(nrow(train)), val),
                                     prop_monitoreo, semilla + i)
      estad <- fun_estadisticos(x_train[partes$ajuste, , drop = FALSE],
                                precio[partes$ajuste])
      ajuste <- fun_entrenar_predecir(
        fun_aplicar_escala(x_train[partes$ajuste, , drop = FALSE], estad),
        (precio[partes$ajuste] - estad$centro_y) / estad$escala_y,
        fun_aplicar_escala(x_train[val, , drop = FALSE], estad),
        candidato$capas, candidato$dropout, max_epocas, semilla,
        fun_aplicar_escala(x_train[partes$monitoreo, , drop = FALSE], estad),
        (precio[partes$monitoreo] - estad$centro_y) / estad$escala_y
      )
      pred <- fun_a_pesos(ajuste$pred_z, estad)
      mae_folds[i] <- mean(abs(pred - precio[val]))
      epocas_folds[i] <- ajuste$epoca
      message("  fold ", i, ": MAE ", round(mae_folds[i] / 1e6, 1),
              " millones, época ", epocas_folds[i], ".")
      rm(ajuste)
      gc(verbose = FALSE)
    }
    fila <- data.frame(
      id = id,
      capas = paste(candidato$capas, collapse = "-"),
      dropout = candidato$dropout,
      tasa_aprendizaje = tasa_aprendizaje,
      mae_espacial = mean(mae_folds),
      epocas = as.integer(round(median(epocas_folds))),
      mae_folds = paste(round(mae_folds), collapse = ";"),
      epocas_folds = paste(epocas_folds, collapse = ";"),
      stringsAsFactors = FALSE
    )
    busqueda <- bind_rows(busqueda, fila)
    write.csv(busqueda, ruta_busqueda, row.names = FALSE)
    message("  MAE espacial = ", round(fila$mae_espacial / 1e6, 1),
            " millones; épocas (mediana) = ", fila$epocas, " (",
            round(proc.time()[["elapsed"]] - inicio), " s).")
  }
}

if (nrow(busqueda) == 0) {
  stop("No hay búsqueda de arquitecturas. Ponga buscar_hiperparametros ",
       "en TRUE o restaure NN_completo_busqueda.csv.")
}
ids_grilla <- vapply(grilla, function(g) {
  fun_id_arquitectura(g$capas, g$dropout)
}, character(1))
busqueda_grilla <- busqueda[busqueda$id %in% ids_grilla, ]
mejor <- busqueda_grilla[which.min(busqueda_grilla$mae_espacial), ]
capas <- as.integer(strsplit(mejor$capas, "-", fixed = TRUE)[[1]])
dropout <- mejor$dropout
epocas <- as.integer(mejor$epocas)
message("Arquitectura elegida: ", mejor$id, ", ", epocas,
        " épocas. MAE espacial de la búsqueda = ",
        round(mejor$mae_espacial / 1e6, 1), " millones.")

nombre <- paste0("NN_completo_capas_", paste(capas, collapse = "_"),
                 "_drop_", fun_texto_archivo(dropout, 2),
                 "_lr_", fun_texto_archivo(tasa_aprendizaje, 3),
                 "_ep_", epocas)
descripcion <- paste0(
  "MLP + atajo lineal; capas=", paste(capas, collapse = "-"),
  "; dropout=", dropout, "; lr=", tasa_aprendizaje,
  "; epocas=", epocas, "; perdida=MAE; precio en pesos estandarizado"
)

# 2. Error fuera de muestra con cada tipo de CV ----------------------------
# Mismos hiperparámetros, incluidas las épocas. Aquí sí se entrena con
# todo el complemento del fold (el 15% de monitoreo solo sirvió para
# elegir las épocas).
oof_nn <- data.frame(property_id = train$property_id)
for (tipo in c("espacial", "aleatoria")) {
  oof <- rep(NA_real_, nrow(train))
  for (i in seq_along(cv_folds[[tipo]])) {
    val <- cv_folds[[tipo]][[i]]
    ajuste_filas <- setdiff(seq_len(nrow(train)), val)
    estad <- fun_estadisticos(x_train[ajuste_filas, , drop = FALSE],
                              precio[ajuste_filas])
    pred_z <- fun_entrenar_predecir(
      fun_aplicar_escala(x_train[ajuste_filas, , drop = FALSE], estad),
      (precio[ajuste_filas] - estad$centro_y) / estad$escala_y,
      fun_aplicar_escala(x_train[val, , drop = FALSE], estad),
      capas, dropout, epocas, semilla
    )$pred_z
    oof[val] <- fun_a_pesos(pred_z, estad)
    message("CV ", tipo, ": fold ", i, "/", length(cv_folds[[tipo]]),
            " listo.")
    gc(verbose = FALSE)
  }
  fun_registrar_cv(
    .archivo = nombre, .algoritmo = "Red neuronal", .conjunto = "completo",
    .n_predictores = ncol(x_train), .hiperparametros = descripcion,
    .tipo_cv = tipo, .obs = precio, .pred = oof
  )
  oof_nn[[paste0("NN_completo_", tipo)]] <- oof
  message("CV ", tipo, ": MAE = ",
          round(mean(abs(oof - precio)) / 1e6, 1), " millones.")
}
saveRDS(oof_nn, "03_temp/oof_nn.rds")

# 3. Modelo final con todo train y envío ----------------------------------
estad_final <- fun_estadisticos(x_train, precio)
pred_test <- fun_a_pesos(fun_entrenar_predecir(
  fun_aplicar_escala(x_train, estad_final),
  (precio - estad_final$centro_y) / estad_final$escala_y,
  fun_aplicar_escala(x_test, estad_final),
  capas, dropout, epocas, semilla
)$pred_z, estad_final)
fun_save_submission(.ids = test$property_id, .pred = pred_test,
                    .nombre = nombre)

# Resumen ------------------------------------------------------------------
registro <- read.csv("02_outputs/tables/registro_modelos.csv")
registro |>
  filter(algoritmo == "Red neuronal") |>
  transmute(archivo = substr(archivo, 1, 42), tipo_cv,
            mae_millones = round(mae / 1e6, 1),
            mae_similares_chapinero = round(mae_similares_chapinero / 1e6, 1),
            sesgo_millones = round(sesgo_medio / 1e6, 1),
            sesgo_similares_chapinero = round(sesgo_similares_chapinero / 1e6, 1),
            pct_sobreestima = round(pct_sobreestima, 1)) |>
  print(row.names = FALSE)

referencia <- registro$mae_similares_chapinero[
  registro$archivo == "LGBM_completo_lr_0_017_hojas_66_prof_7_minhoja_23_rondas_4123" &
    registro$tipo_cv == "espacial"]
nn_similares <- registro$mae_similares_chapinero[
  registro$archivo == nombre & registro$tipo_cv == "espacial"]
if (length(referencia) == 1 && length(nn_similares) == 1) {
  message("MAE en similares a Chapinero (CV espacial): red ",
          round(nn_similares / 1e6, 1), " millones; LightGBM de referencia ",
          round(referencia / 1e6, 1), " millones.")
}

# Limpiamos el entorno: borramos solo los objetos que creó este script.
rm(list = setdiff(ls(), objetos_previos))
