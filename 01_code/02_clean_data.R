# PS3 - Limpieza estructural ----------------------------------------------
# Entrada: 00_data/raw/{train,test}.csv
# Salida:  03_temp/{train_clean,test_clean}.rds

paquetes <- c("readr", "dplyr")
for (paquete in paquetes) {
  if (!require(paquete, character.only = TRUE)) {
    install.packages(paquete)
    library(paquete, character.only = TRUE)
  }
}

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

rm(train, test, fun_limpiar, paquetes, paquete)
