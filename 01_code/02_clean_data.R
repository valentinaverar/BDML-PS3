##########################################################
# 02_clean_data.R
#
# Objetivo: dejar train y test en la misma estructura, corrigiendo solo
# errores de medición que no pueden ser el tamaño real de una vivienda.
# No se imputan faltantes: la mayoría de los anuncios no trae superficie
# y rellenarla aquí contaminaría la validación cruzada de los modelos.
#
# Etapa del Problem Set: Etapa 1.
# Insumos:  00_data/raw/train.csv y 00_data/raw/test.csv.
# Salidas:  03_temp/train_clean.rds, 03_temp/test_clean.rds y
#           03_temp/limpieza_conteos.csv.
##########################################################

# Paquetes -----------------------------------------------------------------
if (!require(pacman)) install.packages("pacman")
pacman::p_load(tidyverse)

# Lectura ------------------------------------------------------------------
# Los CSV vienen en UTF-8 (Bogotá, descripciones con tildes).
db_train_raw <- read.csv("00_data/raw/train.csv",
                         fileEncoding = "UTF-8", stringsAsFactors = FALSE)
db_test_raw  <- read.csv("00_data/raw/test.csv",
                         fileEncoding = "UTF-8", stringsAsFactors = FALSE)

# Limpieza -----------------------------------------------------------------
# Reglas, aplicadas igual en train y en test:
#   1. bedrooms == 0 pasa a faltante. Esas filas tampoco traen cuartos ni
#      superficie, y el título es genérico ("casa en bogota"). Un cero ahí
#      es un código de "no reportado", no un apartamento tipo estudio.
#   2. surface_covered < 12 m2 pasa a faltante. Hay anuncios de 2 a 10 m2
#      con 3 o más cuartos: el área no corresponde a la vivienda.
#   3. surface_total pasa a faltante cuando es inverosímil como área de la
#      vivienda: más de 2.000 m2 sin área cubierta, o más de 1.000 m2 y más
#      de diez veces el área cubierta (un lote o un error de digitación,
#      como 17.137 m2 en Fontibón). Se conserva el área cubierta.
#   4. Si el área cubierta supera a la total, se conservan las dos y se
#      marca superficie_inconsistente. En Properati la total a veces es el
#      lote y la cubierta es el área construida; no se borra el 14 % de
#      los pares completos por esa diferencia.
#   5. superficie_m2 usa el área cubierta y, si falta, la total. Es el
#      tamaño que entra a los modelos. Los originales se conservan.
#   6. Título y descripción faltantes quedan en texto vacío, para que el
#      script de texto no pierda la fila.
# El precio del test viene vacío en las 10.286 filas: se elimina. El de
# train se conserva. No se elimina ninguna propiedad.
fun_limpiar_propiedades <- function(.db, .es_train) {
  .db <- .db |>
    mutate(
      bedrooms = na_if(as.numeric(bedrooms), 0),
      surface_covered = if_else(
        !is.na(surface_covered) & surface_covered < 12,
        NA_real_, as.numeric(surface_covered)
      ),
      surface_total = case_when(
        !is.na(surface_total) & !is.na(surface_covered) &
          surface_total > 1000 &
          surface_total > 10 * surface_covered ~ NA_real_,
        !is.na(surface_total) & is.na(surface_covered) &
          surface_total > 2000 ~ NA_real_,
        TRUE ~ as.numeric(surface_total)
      ),
      superficie_inconsistente = !is.na(surface_covered) &
        !is.na(surface_total) & surface_covered > surface_total,
      superficie_m2 = coalesce(surface_covered, surface_total),
      title = if_else(is.na(title), "", title),
      description = if_else(is.na(description), "", description),
      fecha = as.Date(sprintf("%d-%02d-01", year, month)),
      property_type = factor(property_type, levels = c("Apartamento", "Casa"))
    )
  if (!.es_train) {
    .db <- .db |> select(-price)
  }
  .db
}

# Conteos de la limpieza, antes de modificar los valores ------------------
fun_contar_reglas <- function(.db, .conjunto) {
  tibble(
    conjunto = .conjunto,
    n = nrow(.db),
    bedrooms_cero = sum(.db$bedrooms == 0, na.rm = TRUE),
    cubierta_menor_12 = sum(.db$surface_covered < 12, na.rm = TRUE),
    total_inverosimil = sum(
      (!is.na(.db$surface_total) & !is.na(.db$surface_covered) &
         .db$surface_total > 1000 &
         .db$surface_total > 10 * .db$surface_covered) |
        (!is.na(.db$surface_total) & is.na(.db$surface_covered) &
           .db$surface_total > 2000),
      na.rm = TRUE
    )
  )
}

conteos <- bind_rows(
  fun_contar_reglas(db_train_raw, "train"),
  fun_contar_reglas(db_test_raw, "test")
)

db_train <- fun_limpiar_propiedades(db_train_raw, TRUE)
db_test  <- fun_limpiar_propiedades(db_test_raw, FALSE)

# Verificaciones -----------------------------------------------------------
# Bogotá cabe, con holgura, en este rectángulo. Un punto fuera es una
# coordenada corrida, no una vivienda de otra ciudad.
fuera_de_bogota <- function(.db) {
  .db$lat < 4.45 | .db$lat > 4.85 | .db$lon < -74.25 | .db$lon > -73.99
}

if (any(fuera_de_bogota(db_train)) || any(fuera_de_bogota(db_test))) {
  stop("Hay coordenadas fuera del rectángulo de Bogotá.", call. = FALSE)
}
if (anyDuplicated(db_train$property_id) > 0 ||
    anyDuplicated(db_test$property_id) > 0) {
  stop("Hay property_id repetidos.", call. = FALSE)
}
if (length(intersect(db_train$property_id, db_test$property_id)) > 0) {
  stop("Hay property_id que están en train y en test.", call. = FALSE)
}
if (anyNA(db_train$price) || any(db_train$price <= 0)) {
  stop("El precio de train tiene faltantes o valores no positivos.",
       call. = FALSE)
}
if (nrow(db_train) != nrow(db_train_raw) || nrow(db_test) != nrow(db_test_raw)) {
  stop("La limpieza eliminó propiedades. No debe hacerlo.", call. = FALSE)
}

# Train y test comparten predictores. Train además trae el precio, que es
# la variable a predecir. property_id se conserva en las dos: en test es la
# llave del envío a Kaggle y en train permite unir errores con el anuncio.
# Ningún modelo debe usarlo como predictor.
predictores_train <- setdiff(names(db_train), "price")
if (!setequal(predictores_train, names(db_test))) {
  stop("Train y test no tienen las mismas columnas de predictores.",
       call. = FALSE)
}

# Guardar ------------------------------------------------------------------
dir.create("03_temp", showWarnings = FALSE)
saveRDS(db_train, "03_temp/train_clean.rds")
saveRDS(db_test, "03_temp/test_clean.rds")
write.csv(conteos, "03_temp/limpieza_conteos.csv", row.names = FALSE)

message(
  "Bases guardadas en 03_temp/train_clean.rds (", nrow(db_train),
  " propiedades) y 03_temp/test_clean.rds (", nrow(db_test), ").\n",
  "Correcciones: bedrooms en cero -> NA (train ",
  conteos$bedrooms_cero[1], ", test ", conteos$bedrooms_cero[2],
  "); cubierta < 12 m2 -> NA (train ", conteos$cubierta_menor_12[1],
  ", test ", conteos$cubierta_menor_12[2],
  "); total inverosímil -> NA (train ", conteos$total_inverosimil[1],
  ", test ", conteos$total_inverosimil[2], ")."
)

rm(db_train_raw, db_test_raw, db_train, db_test, conteos,
   fun_limpiar_propiedades, fun_contar_reglas, fuera_de_bogota,
   predictores_train)
