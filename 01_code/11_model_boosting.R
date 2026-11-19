##########################################################
# 11_model_boosting.R
#
# Objetivo: entrenar un modelo de la familia Boosting (LightGBM) para
# predecir el precio de las viviendas y generar su envío a Kaggle.
#
# Etapa del Problem Set: Etapa 2: Training and Comparing Algorithms.
# Insumos:  03_temp/train_features.rds, 03_temp/test_features.rds,
#           03_temp/cv_folds.rds (script 05) y la plantilla de Kaggle.
# Salidas:  02_outputs/predictions/<nombre>.csv (envío a Kaggle).
#           02_outputs/tables/LGBM_metrocuadrado_ensayos.csv (todos los
#           ensayos de hiperparámetros y su MAE en CV espacial).
#           02_outputs/tables/LGBM_metrocuadrado_importancia.csv.
#           Fila en 02_outputs/tables/registro_modelos.csv por cada tipo
#           de CV (espacial y aleatoria).
#           03_temp/: configuración y modelo final.
#
# Decisiones:
#   - LightGBM: implementación de boosting de árboles que optimiza
#     directamente el error absoluto (objective = "regression_l1"), la
#     misma métrica de Kaggle, y maneja los valores faltantes sin imputar.
#   - Hiperparámetros elegidos con CV espacial (decisión del equipo): es la
#     que se parece al problema de predecir en una zona no vista.
#   - Búsqueda: optimización bayesiana (25 puntos iniciales + 125
#     iteraciones); si falla, búsqueda aleatoria con los mismos límites.
#   - La búsqueda tarda horas. Por eso, si ya existe la tabla de ensayos y
#     buscar_hiperparametros es FALSE, se reutiliza el mejor ensayo guardado
#     y solo se reentrena el modelo final (reproducible en minutos).
#
# Uso: LGBM_SOLO_VERIFICAR=1 ejecuta una CV y un ajuste cortos sin escribir
# resultados (prueba rápida de que todo funciona).
##########################################################

# Objetos que ya existían en el entorno antes de correr este script. Al
# final se borra solo lo que este script creó, sin tocar lo demás.
objetos_previos <- ls()

# TRUE: repite la búsqueda de hiperparámetros (tarda horas).
# FALSE: usa el mejor ensayo de la tabla guardada, si existe.
buscar_hiperparametros <- FALSE

local({
  fallar <- function(...) stop(..., call. = FALSE)
  solo_verificar <- identical(Sys.getenv("LGBM_SOLO_VERIFICAR"), "1")
  prueba_interna <- isTRUE(getOption("ps3.lgbm.prueba", FALSE))
  
  rutas <- c(
    train = "03_temp/train_features.rds",
    test = "03_temp/test_features.rds",
    folds = "03_temp/cv_folds.rds",
    plantilla = "00_data/raw/submission_template.csv",
    submission = "01_code/functions/fun_save_submission.R",
    metricas = "01_code/functions/fun_metricas.R"
  )
  archivos_faltantes <- rutas[!file.exists(rutas)]
  if (length(archivos_faltantes)) {
    fallar("Desde la raiz de BDML-PS3, faltan: ",
           paste(archivos_faltantes, collapse = ", "), ". Directorio: ", getwd())
  }
  
  paquetes <- c("lightgbm", "rBayesianOptimization", "dplyr", "tibble")
  for (paquete in paquetes) {
    tryCatch(loadNamespace(paquete), error = function(e) {
      fallar("No se pudo cargar ", paquete, ": ", conditionMessage(e),
             ". Revise la libreria de la sesion de R antes de entrenar.")
    })
  }
  # Los helpers del repositorio resuelven estas funciones sin namespace.
  filter <- dplyr::filter
  bind_rows <- dplyr::bind_rows
  bind_cols <- dplyr::bind_cols
  tibble <- tibble::tibble
  sys.source(rutas[["submission"]], envir = environment())
  sys.source(rutas[["metricas"]], envir = environment())
  
  # GPfit carga lhs al ajustar el primer proceso gaussiano.
  error_lhs <- tryCatch({
    loadNamespace("lhs")
    NULL
  }, error = function(e) conditionMessage(e))
  if (!is.null(error_lhs)) {
    message("No se pudo cargar lhs para Bayes: ", error_lhs,
            ". El entrenamiento completo usara busqueda aleatoria.")
  }
  
  set.seed(2026)
  hilos <- 6L
  train <- as.data.frame(readRDS(rutas[["train"]]))
  test <- as.data.frame(readRDS(rutas[["test"]]))
  folds <- readRDS(rutas[["folds"]])
  
  variables <- c(
    "estrato_manzana", "txt_antiguedad_anios", "bedrooms", "bathrooms",
    "txt_terraza", "txt_balcon", "property_type", "surface_total",
    "surface_covered", "txt_parqueadero", "lat", "lon"
  )
  for (base in c("train", "test")) {
    datos <- if (base == "train") train else test
    faltantes <- setdiff(c(variables, if (base == "train") "price"), names(datos))
    if (!nrow(datos) || length(faltantes)) {
      fallar("Revise ", rutas[[base]], ". Filas: ", nrow(datos),
             ". Columnas faltantes: ", paste(faltantes, collapse = ", "))
    }
  }
  
  a_numero <- function(x, etiqueta) {
    if (is.numeric(x) || is.logical(x)) return(as.numeric(x))
    texto <- trimws(as.character(x))
    res <- suppressWarnings(as.numeric(texto))
    if (any(is.na(res) & !is.na(texto) & nzchar(texto) &
            !texto %in% c("NA", "NaN"))) {
      fallar("Valores no numericos en ", etiqueta, ".")
    }
    res
  }
  
  y <- a_numero(train$price, "train$price")
  if (any(!is.finite(y) | y <= 0)) fallar("Hay precios invalidos en train.")
  
  plantilla <- utils::read.csv(rutas[["plantilla"]], colClasses = "character")
  if (ncol(plantilla) != 2L || names(plantilla)[2L] != "price" ||
      nrow(plantilla) != nrow(test)) {
    fallar("La plantilla de Kaggle debe tener ID, price y una fila por propiedad.")
  }
  col_id <- names(plantilla)[1L]
  pos_id <- intersect(c(col_id, "property_id", "id"), names(test))
  if (!length(pos_id)) {
    fallar("Falta property_id (o id) en test_features.rds. Regenere el RDS ",
           "con los pasos de preparacion, conservando property_id de test.csv.")
  }
  ids_test <- as.character(test[[pos_id[1L]]])
  ids_kaggle <- plantilla[[col_id]]
  validar_ids <- function(ids, etiqueta) {
    if (anyNA(ids) || any(!nzchar(trimws(ids))) || anyDuplicated(ids)) {
      fallar("Hay IDs vacios o duplicados en ", etiqueta, ".")
    }
  }
  validar_ids(ids_test, "test_features.rds")
  validar_ids(ids_kaggle, "submission_template.csv")
  orden_submission <- match(ids_kaggle, ids_test)
  if (anyNA(orden_submission)) fallar("Los IDs de test y Kaggle no coinciden.")
  
  for (tipo in c("espacial", "aleatoria")) {
    partes <- folds[[tipo]]
    indices_validos <- function(idx) {
      is.numeric(idx) && length(idx) > 0L &&
        all(is.finite(idx) & idx == trunc(idx) & idx >= 1L & idx <= length(y))
    }
    if (!is.list(partes) || length(partes) < 2L ||
        !all(vapply(partes, indices_validos, logical(1L))) ||
        !identical(sort(as.integer(unlist(partes))), seq_along(y))) {
      fallar("Los folds de CV ", tipo, " deben cubrir cada fila de train una vez. ",
             "Revise 05_create_cv_folds.R.")
    }
    folds[[tipo]] <- lapply(partes, function(idx) sort(as.integer(idx)))
  }
  
  niveles <- sort(unique(as.character(train$property_type)))
  niveles <- niveles[!is.na(niveles) & nzchar(niveles)]
  if (!length(niveles)) fallar("property_type no tiene valores en train.")
  train$property_type <- match(as.character(train$property_type), niveles) - 1L
  test$property_type <- match(as.character(test$property_type), niveles) - 1L
  for (v in setdiff(variables, "property_type")) {
    train[[v]] <- a_numero(train[[v]], paste0("train$", v))
    test[[v]] <- a_numero(test[[v]], paste0("test$", v))
  }
  x_train <- as.matrix(train[, variables, drop = FALSE])
  x_test <- as.matrix(test[, variables, drop = FALSE])
  x_train[!is.finite(x_train)] <- NA_real_
  x_test[!is.finite(x_test)] <- NA_real_
  if (any(colSums(!is.na(x_train)) == 0L)) {
    fallar("Predictores sin datos en train: ",
           paste(variables[colSums(!is.na(x_train)) == 0L], collapse = ", "))
  }
  rm(train, test, datos)
  message("LightGBM Metrocuadrado: ", nrow(x_train), " train, ", nrow(x_test),
          " test, ", ncol(x_train), " predictores.")
  
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
  
  crear_dataset <- function(x, obs) {
    lightgbm::lgb.Dataset(x, label = obs, categorical_feature = "property_type")
  }
  ejecutar_cv <- function(params, nrounds, paciencia) {
    cv <- lightgbm::lgb.cv(
      params = params, data = crear_dataset(x_train, y),
      folds = folds$espacial, nrounds = nrounds,
      early_stopping_rounds = paciencia, verbose = -1L, showsd = FALSE
    )
    mae <- as.numeric(cv$best_score)
    ronda <- as.integer(cv$best_iter)
    if (length(mae) != 1L || !is.finite(mae) ||
        length(ronda) != 1L || is.na(ronda) || ronda < 1L || ronda > nrounds) {
      fallar("LightGBM no devolvio un MAE y una ronda validos.")
    }
    list(mae = mae, ronda = ronda)
  }
  
  predecir <- function(modelo, x) {
    pred <- as.numeric(stats::predict(modelo, x))
    if (length(pred) != nrow(x) || any(!is.finite(pred))) {
      fallar("LightGBM devolvio predicciones invalidas.")
    }
    pred
  }
  
  if (solo_verificar) {
    inicio <- proc.time()[["elapsed"]]
    params <- armar_params(log(0.04), 31L, 8L, 40L, 0.85, 0.85, 0, 0)
    cv <- ejecutar_cv(params, nrounds = 30L, paciencia = 10L)
    modelo <- lightgbm::lgb.train(params, crear_dataset(x_train, y),
                                  nrounds = cv$ronda, verbose = -1L)
    pred <- predecir(modelo, x_test)[orden_submission]
    message("Verificacion OK: MAE espacial = ", round(cv$mae),
            ", predicciones = ", length(pred),
            ", segundos = ", round(proc.time()[["elapsed"]] - inicio, 1L), ".")
  } else {
    nombre <- "LGBM_metrocuadrado_bayes"
    ruta_ensayos <- "02_outputs/tables/LGBM_metrocuadrado_ensayos.csv"
    ruta_registro <- "02_outputs/tables/registro_modelos.csv"
    ruta_importancia <- "02_outputs/tables/LGBM_metrocuadrado_importancia.csv"
    ruta_pred <- file.path("02_outputs/predictions", paste0(nombre, ".csv"))
    ruta_config <- "03_temp/LGBM_metrocuadrado_config.rds"
    ruta_modelo <- "03_temp/LGBM_metrocuadrado_modelo.txt"
    salidas <- c(ruta_ensayos, ruta_registro, ruta_importancia,
                 ruta_pred, ruta_config, ruta_modelo)
    for (carpeta in unique(dirname(salidas))) {
      dir.create(carpeta, recursive = TRUE, showWarnings = FALSE)
      temporal <- tempfile(tmpdir = carpeta)
      writeLines("test", temporal)
      unlink(temporal)
    }
    existentes <- salidas[file.exists(salidas)]
    for (ruta in existentes) {
      tryCatch({
        archivo <- file(ruta, "r+b")
        close(archivo)
      }, error = function(e) {
        fallar("No se puede escribir en ", ruta, ": ", conditionMessage(e))
      })
    }
    reutilizar <- !buscar_hiperparametros && file.exists(ruta_ensayos)
    if (reutilizar) {
      message("Se reutilizan los hiperparámetros de ", ruta_ensayos,
              " (buscar_hiperparametros = FALSE).")
    }
    if (file.exists(ruta_registro)) {
      campos <- names(utils::read.csv(ruta_registro, nrows = 1L))
      if (!all(c("archivo", "tipo_cv") %in% campos)) {
        fallar("El registro de modelos existente no tiene archivo y tipo_cv.")
      }
    }
    # Límites de búsqueda de cada hiperparámetro.
    limites <- list(
      log_lr = log(c(0.008, 0.12)),
      num_leaves = c(15L, 127L),
      max_depth = c(4L, 12L),
      min_data_in_leaf = c(15L, 200L),
      feature_fraction = c(0.65, 1),
      bagging_fraction = c(0.65, 1),
      log_l1 = log1p(c(0, 20)),
      log_l2 = log1p(c(0, 20))
    )
    
    error_bayes <- NULL
    if (!reutilizar) {
      if (file.exists(ruta_ensayos)) {
        respaldo <- tempfile("LGBM_metrocuadrado_ensayos_anterior_",
                             tmpdir = dirname(ruta_ensayos), fileext = ".csv")
        if (!file.rename(ruta_ensayos, respaldo)) {
          fallar("Cierre el CSV de ensayos anterior para poder iniciar otra corrida.")
        }
        message("Ensayos anteriores: ", respaldo)
      }
      
      puntos_iniciales <- if (prueba_interna) 10L else 25L
      iteraciones_bayes <- if (prueba_interna) 1L else 125L
      max_rondas <- if (prueba_interna) 30L else 4000L
      paciencia <- if (prueba_interna) 10L else 100L
      total_ensayos <- puntos_iniciales + iteraciones_bayes
      n_ensayos <- 0L
      error_ensayo <- FALSE
      
      puntuar <- function(log_lr, num_leaves, max_depth, min_data_in_leaf,
                          feature_fraction, bagging_fraction, log_l1, log_l2) {
        ronda_ensayo <- n_ensayos + 1L
        message("Ensayo ", ronda_ensayo, "/", total_ensayos)
        inicio <- proc.time()[["elapsed"]]
        withCallingHandlers({
          params <- armar_params(log_lr, num_leaves, max_depth, min_data_in_leaf,
                                 feature_fraction, bagging_fraction, log_l1, log_l2)
          cv <- ejecutar_cv(params, max_rondas, paciencia)
          fila <- data.frame(
            mae = cv$mae, iteraciones = cv$ronda,
            log_lr = log_lr, num_leaves = num_leaves, max_depth = max_depth,
            min_data_in_leaf = min_data_in_leaf, feature_fraction = feature_fraction,
            bagging_fraction = bagging_fraction, log_l1 = log_l1, log_l2 = log_l2
          )
          utils::write.table(fila, ruta_ensayos, sep = ",", row.names = FALSE,
                             col.names = n_ensayos == 0L, append = n_ensayos > 0L)
          n_ensayos <<- ronda_ensayo
          message("MAE espacial: $", round(cv$mae), " | rondas: ", cv$ronda,
                  " | segundos: ", round(proc.time()[["elapsed"]] - inicio, 1L))
          list(Score = -cv$mae / 1e6, Pred = 0)
        }, error = function(e) error_ensayo <<- TRUE)
      }
      
      error_bayes <- error_lhs
      if (is.null(error_bayes)) {
        error_bayes <- tryCatch({
          rBayesianOptimization::BayesianOptimization(
            FUN = puntuar, bounds = limites, init_points = puntos_iniciales,
            n_iter = iteraciones_bayes, acq = "ei", verbose = TRUE
          )
          NULL
        }, error = function(e) {
          if (error_ensayo) stop(e)
          conditionMessage(e)
        })
      }
      if (!is.null(error_bayes)) {
        message("Busqueda aleatoria para completar ", total_ensayos - n_ensayos,
                " ensayos. Motivo: ", error_bayes)
        while (n_ensayos < total_ensayos) {
          candidato <- lapply(limites, function(limite) {
            if (is.integer(limite)) {
              return(limite[1L] - 1L + sample.int(diff(limite) + 1L, 1L))
            }
            stats::runif(1L, limite[1L], limite[2L])
          })
          do.call(puntuar, candidato)
        }
      }
      
    }
    
    ensayos <- utils::read.csv(ruta_ensayos)
    if (reutilizar) n_ensayos <- nrow(ensayos)
    b <- ensayos[which.min(ensayos$mae), ]
    mejores <- do.call(armar_params, as.list(b[names(limites)]))
    rondas <- as.integer(b$iteraciones)
    busqueda <- if (reutilizar) {
      "Ensayos guardados (CV espacial)"
    } else if (is.null(error_bayes)) {
      "Bayes espacial"
    } else {
      "CV espacial con respaldo aleatorio"
    }
    message("Mejor MAE espacial: $", round(b$mae), " | rondas: ", rondas)
    saveRDS(list(params = mejores, rondas = rondas, variables = variables,
                 niveles_property_type = niveles, ensayos = n_ensayos,
                 error_bayes = error_bayes), ruta_config)
    
    # Guardar la submission antes de los diagnosticos OOF.
    modelo_final <- lightgbm::lgb.train(
      params = mejores, data = crear_dataset(x_train, y),
      nrounds = rondas, verbose = -1L
    )
    pred_test <- pmax(predecir(modelo_final, x_test), 1)
    fun_save_submission(.ids = ids_kaggle, .pred = pred_test[orden_submission],
                        .nombre = nombre, .sample = rutas[["plantilla"]])
    lightgbm::lgb.save(modelo_final, ruta_modelo)
    utils::write.csv(lightgbm::lgb.importance(modelo_final), ruta_importancia,
                     row.names = FALSE)
    
    for (tipo in c("espacial", "aleatoria")) {
      partes <- folds[[tipo]]
      oof <- rep(NA_real_, length(y))
      for (i in seq_along(partes)) {
        val <- partes[[i]]
        message("CV ", tipo, ": fold ", i, "/", length(partes))
        modelo_cv <- lightgbm::lgb.train(
          params = mejores,
          data = crear_dataset(x_train[-val, , drop = FALSE], y[-val]),
          nrounds = rondas, verbose = -1L
        )
        oof[val] <- predecir(modelo_cv, x_train[val, , drop = FALSE])
      }
      fun_registrar_cv(
        .archivo = nombre, .algoritmo = "LightGBM", .conjunto = "metrocuadrado",
        .n_predictores = length(variables),
        .hiperparametros = paste0(busqueda, "; rondas=", rondas),
        .tipo_cv = tipo, .obs = y, .pred = oof
      )
      message("MAE ", tipo, " (OOF): $", round(mean(abs(oof - y))))
    }
    message("Listo: ", n_ensayos, " ensayos, submission y diagnosticos OOF.")
  }
})

# Limpiamos el entorno: borramos solo los objetos que creó este script.
rm(list = setdiff(ls(), objetos_previos))
