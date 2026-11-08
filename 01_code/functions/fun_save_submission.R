##########################################################
# fun_save_submission.R
#
# Funciones para convertir los precios predichos por un modelo en un archivo
# listo para subir a Kaggle. Reúnen los pasos que se repiten en todos los
# envíos: armar la base, nombrar el archivo, verificar su formato y
# guardarlo.
#
# Adaptado del PS2. Diferencia principal: aquí el problema es de regresión
# (se predice un precio), así que no hay umbral de clasificación.
##########################################################

fun_hiperparam_str <- function(.x, .digits = 4) {
  #' Convertir un hiperparámetro numérico en texto para nombres de archivo
  #'
  #' @description \code{fun_hiperparam_str()} redondea un número y reemplaza
  #' el punto decimal por un guion bajo, para que los nombres de archivo no
  #' tengan puntos. Por ejemplo, 0.0123 se convierte en "0_0123".
  #'
  #' @param .x \code{numeric}. Valor del hiperparámetro.
  #' @param .digits \code{integer}. Número de decimales al redondear.
  #'
  #' @returns \code{character}. El valor como texto, sin puntos.
  gsub(pattern = ".", replacement = "_",
       x = as.character(round(.x, .digits)), fixed = TRUE)
}

fun_save_submission <- function(.ids, .pred, .nombre,
                                .dir = "02_outputs/predictions",
                                .sample = NULL) {
  #' Guardar una predicción en el formato de envío de Kaggle
  #'
  #' @description \code{fun_save_submission()} arma la base de envío con el
  #' identificador de cada propiedad y su precio predicho, verifica que
  #' cumpla el formato de la competencia y la guarda como .csv. Los nombres
  #' de las columnas se toman del archivo de ejemplo de Kaggle, así que no
  #' hay que escribirlos a mano. Recibe predicciones (y no el modelo) para
  #' servir con cualquier algoritmo.
  #'
  #' @param .ids Identificador de cada propiedad del test, en el mismo orden
  #' que \code{.pred}.
  #' @param .pred \code{numeric}. Precio predicho para cada propiedad, en la
  #' escala original (pesos). Si el modelo se entrenó con log(precio), hay
  #' que transformar de vuelta antes de llamar esta función.
  #' @param .nombre \code{character}. Nombre del archivo, sin la extensión
  #' .csv. Debe indicar el algoritmo y los hiperparámetros, por ejemplo
  #' "RF_mtry_6_trees_500".
  #' @param .dir \code{character}. Carpeta de destino.
  #' @param .sample \code{character}. Ruta al archivo de envío de ejemplo de
  #' Kaggle. Si es NULL, se busca en 00_data/raw/ el único .csv cuyo nombre
  #' contenga "submission".
  #'
  #' @returns \code{data.frame} (invisible) con las dos columnas del envío.
  #' Como efecto secundario, guarda el archivo en \code{.dir}.

  # Archivo de ejemplo de la competencia ----------------------------------
  if (is.null(.sample)) {
    candidatos <- list.files("00_data/raw", pattern = "submission.*\\.csv$",
                             full.names = TRUE, ignore.case = TRUE)
    if (length(candidatos) != 1) {
      stop("No se pudo identificar un único archivo de envío de ejemplo en ",
           "00_data/raw/ (se encontraron ", length(candidatos), "). ",
           "Indique la ruta con el argumento .sample.", call. = FALSE)
    }
    .sample <- candidatos
  }
  db_sample <- utils::read.csv(.sample)
  col_id <- names(db_sample)[1]
  col_pred <- names(db_sample)[2]

  # Validación de los insumos ---------------------------------------------
  # Detenerse aquí es mejor que gastar un envío diario de Kaggle en un
  # archivo con errores.
  if (length(.pred) != length(.ids)) {
    stop("Hay ", length(.pred), " predicciones para ", length(.ids),
         " propiedades. Deben coincidir.", call. = FALSE)
  }
  if (anyNA(.pred)) {
    stop("Hay ", sum(is.na(.pred)), " predicciones faltantes (NA). ",
         "Revise los valores faltantes en las variables del test.",
         call. = FALSE)
  }
  if (any(.pred <= 0)) {
    stop("Hay ", sum(.pred <= 0), " precios predichos menores o iguales ",
         "a cero.", call. = FALSE)
  }
  if (!grepl("^[A-Za-z0-9_]+$", .nombre)) {
    stop("El nombre '", .nombre, "' solo puede tener letras, números y ",
         "guiones bajos (sin puntos ni espacios, y sin la extensión .csv).",
         call. = FALSE)
  }

  # Armar la base de envío -------------------------------------------------
  db_submission <- data.frame(.ids, .pred)
  names(db_submission) <- c(col_id, col_pred)

  # Verificación contra el formato de la competencia -----------------------
  if (anyDuplicated(db_submission[[col_id]]) > 0) {
    stop("Hay propiedades repetidas en la predicción.", call. = FALSE)
  }
  if (!setequal(db_submission[[col_id]], db_sample[[col_id]])) {
    stop("Los identificadores de la predicción no coinciden con los de ",
         basename(.sample), ".", call. = FALSE)
  }

  # Guardar ---------------------------------------------------------------
  dir.create(.dir, recursive = TRUE, showWarnings = FALSE)
  ruta <- file.path(.dir, paste0(.nombre, ".csv"))
  if (file.exists(ruta)) {
    warning("Se sobrescribió el archivo existente ", ruta, call. = FALSE)
  }
  utils::write.csv(db_submission, ruta, row.names = FALSE)

  message("Guardado: ", ruta, "\n",
          "  Propiedades: ", nrow(db_submission), "\n",
          "  Precio predicho: mediana ",
          format(stats::median(.pred), big.mark = ",", scientific = FALSE),
          ", rango [", format(min(.pred), big.mark = ",", scientific = FALSE),
          ", ", format(max(.pred), big.mark = ",", scientific = FALSE), "]")

  invisible(db_submission)
}
