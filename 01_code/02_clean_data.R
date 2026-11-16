##########################################################
# 02_clean_data.R
#
# Objetivo: limpieza estructural de las bases de propiedades (train y test),
# sin imputar ni eliminar observaciones.
#
# Etapa del Problem Set: Etapa 1: Understanding and Expanding the Data.
# Insumos:  00_data/raw/train.csv y 00_data/raw/test.csv.
# Salidas:  03_temp/train_clean.rds y 03_temp/test_clean.rds.
#
# Qué hace:
#   1. Reemplaza los títulos y descripciones faltantes por texto vacío, para
#      que el script de variables de texto no falle con NA.
#   2. Crea tres indicadores de problemas en la superficie: sin información,
#      inconsistente (construida mayor que total) y valores extremos.
#   3. Elimina city y operation_type, que son constantes en train y test
#      ("Bogotá D.C" y "Venta"): no aportan información al modelo.
#
# Decisión: no se eliminan observaciones. En test hay que predecir todas
# las propiedades, así que los problemas se marcan con indicadores en lugar
# de borrarse, y los modelos deciden cómo usarlos.
##########################################################

# Paquetes ----------------------------------------------------------------
if (!require(pacman)) install.packages("pacman")
pacman::p_load(readr, dplyr)

dir.create("03_temp", showWarnings = FALSE)

fun_limpiar <- function(datos) {
  datos %>%
    mutate(
      price = as.numeric(price),  # En test todos los precios son NA.
      title = trimws(coalesce(title, "")),
      description = trimws(coalesce(description, "")),
      flag_sin_superficie = as.integer(is.na(surface_total) & is.na(surface_covered)),
      flag_superficie_inconsistente = as.integer(
        !is.na(surface_total) & !is.na(surface_covered) &
          surface_covered > surface_total
      ),
      flag_superficie_extrema = as.integer(
        coalesce(surface_total < 15 | surface_total > 2000, FALSE) |
          coalesce(surface_covered < 15 | surface_covered > 1000, FALSE)
      )
    ) %>%
    select(-city, -operation_type)  # Son constantes en train y test.
}

train <- fun_limpiar(read_csv("00_data/raw/train.csv", show_col_types = FALSE))
test  <- fun_limpiar(read_csv("00_data/raw/test.csv", show_col_types = FALSE))

saveRDS(train, "03_temp/train_clean.rds")
saveRDS(test, "03_temp/test_clean.rds")

message("Bases limpias guardadas en 03_temp/. Filas: train = ", nrow(train),
        "; test = ", nrow(test), ".")

rm(train, test, fun_limpiar)
