##########################################################
# fun_vecinos.R
#
# Función para construir el rezago espacial del precio: el valor típico de
# las viviendas vecinas de train (Lecture 14, datos espaciales). La usa
# 11d_model_boosting_vecinos.R.
#
# Cuidado con la fuga de información: el precio de los vecinos solo puede
# salir de viviendas cuyo precio el modelo tiene permitido ver. Por eso la
# función recibe por separado las viviendas de referencia (las que aportan
# su precio) y las viviendas para las que se calcula la variable.
##########################################################

fun_vecinos <- function(.xy_ref, .valor_ref, .grupo_ref, .xy_obj, .grupo_obj,
                        .k = 10L, .extra = 40L) {
  #' Mediana del valor de los k vecinos de referencia más cercanos
  #'
  #' @description Para cada vivienda de .xy_obj busca las viviendas de
  #' referencia más cercanas (distancia en línea recta, en metros) y
  #' devuelve la mediana de su valor (por ejemplo, el precio por m²) y la
  #' mediana de la distancia a ellas. Se descartan los vecinos del mismo
  #' grupo que la vivienda (el mismo anuncio publicado varias veces, y la
  #' propia vivienda cuando también es de referencia): su precio es el
  #' mismo que se quiere predecir.
  #'
  #' @param .xy_ref \code{matrix}. Coordenadas en metros de las viviendas
  #' de referencia (dos columnas).
  #' @param .valor_ref \code{numeric}. Valor de cada vivienda de referencia.
  #' @param .grupo_ref,.grupo_obj \code{character}. Grupo de anuncio de cada
  #' vivienda de referencia y de cada vivienda objetivo.
  #' @param .xy_obj \code{matrix}. Coordenadas en metros de las viviendas
  #' para las que se calcula la variable.
  #' @param .k \code{integer}. Número de vecinos.
  #' @param .extra \code{integer}. Vecinos adicionales que se buscan para
  #' poder descartar los del mismo grupo y aun así tener .k.
  #'
  #' @returns \code{data.frame} con \code{mediana} (valor típico de los
  #' vecinos) y \code{distancia} (distancia mediana a ellos, en metros).
  n_buscar <- min(nrow(.xy_ref), .k + .extra)
  cercanos <- FNN::get.knnx(.xy_ref, .xy_obj, k = n_buscar)
  resultado <- vapply(seq_len(nrow(.xy_obj)), function(i) {
    indices <- cercanos$nn.index[i, ]
    distancias <- cercanos$nn.dist[i, ]
    validos <- .grupo_ref[indices] != .grupo_obj[i]
    indices <- utils::head(indices[validos], .k)
    distancias <- utils::head(distancias[validos], .k)
    if (!length(indices)) return(c(NA_real_, NA_real_))
    c(stats::median(.valor_ref[indices]), stats::median(distancias))
  }, numeric(2))
  data.frame(mediana = resultado[1, ], distancia = resultado[2, ])
}

fun_grupo_anuncio <- function(.datos) {
  #' Identificador de anuncio para detectar publicaciones repetidas
  #'
  #' @description Dos viviendas son el mismo anuncio si tienen la misma
  #' descripción (sin mayúsculas ni espacios repetidos). Las descripciones
  #' muy cortas (menos de 50 caracteres) no se comparan: cada vivienda queda
  #' en su propio grupo.
  #'
  #' @returns \code{character}: un grupo por vivienda.
  texto <- gsub("\\s+", " ", trimws(tolower(.datos$description)))
  ifelse(!is.na(texto) & nchar(texto) >= 50, texto,
         paste0("id_", .datos$property_id))
}
