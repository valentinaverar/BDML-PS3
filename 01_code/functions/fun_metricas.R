##########################################################
# fun_metricas.R
#
# Funciones de evaluación que comparten todos los modelos: el error en
# validación cruzada (MAE, que es la métrica de Kaggle, y RMSE), la dirección
# de los errores (sobreestimación frente a subestimación) y el registro de
# cada variante. Requiere que tidyverse esté cargado.
#
# Adaptado del PS2. Diferencia principal: aquí el problema es de regresión y,
# como en el caso de Zillow, sobreestimar un precio es más costoso que
# subestimarlo. Por eso, además del MAE, se registra cómo se reparte el error
# entre sobreestimaciones y subestimaciones.
##########################################################

fun_metricas_regresion <- function(.obs, .pred) {
  #' Métricas de error de una predicción de precios
  #'
  #' @description \code{fun_metricas_regresion()} calcula el MAE (métrica de
  #' la competencia), el RMSE y la dirección de los errores. El error se
  #' define como predicho - observado, de modo que un error positivo es una
  #' sobreestimación: el modelo dice que la propiedad vale más de lo que
  #' vale y la empresa pagaría de más.
  #'
  #' @param .obs \code{numeric}. Precio observado.
  #' @param .pred \code{numeric}. Precio predicho, en la misma escala.
  #'
  #' @returns \code{tibble} de una fila con: \code{mae}, \code{rmse},
  #' \code{sesgo_medio} (error promedio; positivo indica que el modelo
  #' sobreestima en promedio), \code{pct_sobreestima} (porcentaje de
  #' propiedades sobreestimadas) y \code{mae_sobreestima} (error absoluto
  #' medio solo entre las propiedades sobreestimadas).
  error <- .pred - .obs
  tibble(
    mae = mean(abs(error)),
    rmse = sqrt(mean(error^2)),
    sesgo_medio = mean(error),
    pct_sobreestima = 100 * mean(error > 0),
    mae_sobreestima = if (any(error > 0)) mean(error[error > 0]) else 0
  )
}

fun_registrar_modelo <- function(.fila,
                                 .ruta = "02_outputs/tables/registro_modelos.csv") {
  #' Registrar una variante en la tabla de modelos
  #'
  #' @description \code{fun_registrar_modelo()} agrega la fila de una
  #' variante a la tabla de registro. Si ya existe una fila con el mismo
  #' archivo y el mismo tipo de validación cruzada (porque el script se
  #' volvió a correr), la reemplaza en lugar de duplicarla. Así una misma
  #' variante puede tener una fila con CV aleatoria y otra con CV espacial. El puntaje público de Kaggle se anota aparte, en
  #' \code{00_data/kaggle_scores.csv}, y las dos tablas se unen por
  #' \code{archivo}.
  #'
  #' @param .fila \code{data.frame} de una fila. Debe tener las columnas
  #' \code{archivo} y \code{tipo_cv}.
  #' @param .ruta \code{character}. Ruta de la tabla de registro.
  #'
  #' @returns La tabla de registro actualizada (invisible). Como efecto
  #' secundario, la guarda en \code{.ruta}.
  if (file.exists(.ruta)) {
    registro <- read.csv(.ruta) |>
      filter(!(archivo == .fila$archivo & tipo_cv == .fila$tipo_cv)) |>
      bind_rows(.fila)
  } else {
    dir.create(dirname(.ruta), recursive = TRUE, showWarnings = FALSE)
    registro <- .fila
  }
  write.csv(registro, .ruta, row.names = FALSE)
  invisible(registro)
}

fun_registrar_cv <- function(.archivo, .algoritmo, .conjunto, .n_predictores,
                             .hiperparametros, .tipo_cv, .obs, .pred,
                             .prop_similares = 0.10,
                             .ruta_similitud = "03_temp/similitud_chapinero.rds") {
  #' Registrar una variante a partir de sus predicciones fuera de muestra
  #'
  #' @description \code{fun_registrar_cv()} calcula las métricas de
  #' validación cruzada de una variante con \code{fun_metricas_regresion()}
  #' y la agrega a la tabla de registro con \code{fun_registrar_modelo()}.
  #' Así todos los algoritmos quedan en la misma tabla y se pueden comparar,
  #' tanto entre sí como contra su puntaje en Kaggle.
  #'
  #' Además calcula el MAE y el sesgo solo en las propiedades de train más
  #' similares a Chapinero (similitud del script 06, un propensity score que
  #' no usa coordenadas, nombres de barrios ni el precio). El MAE de toda la
  #' CV promedia el error de todo Bogotá, donde hay muchas viviendas más
  #' baratas que en Chapinero; como el MAE se mide en pesos, ese promedio
  #' subestima el error en Chapinero. El MAE en las similares es una mejor
  #' aproximación (aunque todavía un límite inferior) del error en el test.
  #'
  #' @param .archivo \code{character}. Nombre del envío (sin .csv).
  #' @param .algoritmo \code{character}. Nombre del algoritmo.
  #' @param .conjunto \code{character}. Conjunto de predictores.
  #' @param .n_predictores \code{integer}. Número de predictores.
  #' @param .hiperparametros \code{character}. Descripción de los
  #' hiperparámetros.
  #' @param .tipo_cv \code{character}. "aleatoria" o "espacial", para
  #' comparar ambos esquemas en la Etapa 3.
  #' @param .obs \code{numeric}. Precio observado (en el orden de train).
  #' @param .pred \code{numeric}. Precio predicho fuera de muestra.
  #' @param .prop_similares \code{numeric}. Fracción de train más similar a
  #' Chapinero en la que se mide el error adicional.
  #' @param .ruta_similitud \code{character}. Archivo con la similitud de
  #' cada propiedad de train (script 06). Si no existe, las columnas
  #' adicionales quedan en NA.
  #'
  #' @returns La tabla de registro actualizada (invisible).
  metricas <- fun_metricas_regresion(.obs, .pred)
  
  # Error en las propiedades más similares a Chapinero.
  metricas_similares <- tibble(mae_similares_chapinero = NA_real_,
                               sesgo_similares_chapinero = NA_real_)
  if (file.exists(.ruta_similitud)) {
    similitud <- readRDS(.ruta_similitud)$similitud
    if (length(similitud) != length(.obs)) {
      stop("La similitud tiene ", length(similitud), " filas y las ",
           "predicciones ", length(.obs), ". Corra de nuevo el script 06.",
           call. = FALSE)
    }
    similares <- similitud >= stats::quantile(similitud, 1 - .prop_similares)
    error_similares <- .pred[similares] - .obs[similares]
    metricas_similares <- tibble(
      mae_similares_chapinero = mean(abs(error_similares)),
      sesgo_similares_chapinero = mean(error_similares)
    )
  }
  
  fun_registrar_modelo(bind_cols(
    tibble(
      archivo = .archivo,
      algoritmo = .algoritmo,
      conjunto = .conjunto,
      n_predictores = .n_predictores,
      hiperparametros = .hiperparametros,
      tipo_cv = .tipo_cv
    ),
    metricas,
    metricas_similares,
    tibble(fecha = as.character(Sys.Date()))
  ))
}
