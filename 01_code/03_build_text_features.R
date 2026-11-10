# PS3 - Features textuales -------------------------------------------------
# Ejecutar desde la raiz del repositorio:
# source("01_code/04_build_text_features.R")
# Usa las bases con geofeatures si existen; si no, usa los CSV originales.
# Salida: 03_temp/{train_features,test_features}.rds

# Paquetes ----------------------------------------------------------------
paquetes <- c("readr", "stringr", "stringi")
for (paquete in paquetes) {
  if (!require(paquete, character.only = TRUE)) {
    install.packages(paquete)
    library(paquete, character.only = TRUE)
  }
}

# Cargar datos -------------------------------------------------------------
dir_temp <- "03_temp"
dir_raw <- file.path("00_data", "raw")
dir.create(dir_temp, recursive = TRUE, showWarnings = FALSE)

archivos_geo <- file.path(dir_temp, c("train_geo.rds", "test_geo.rds"))
if (all(file.exists(archivos_geo))) {
  train <- readRDS(archivos_geo[1])
  test <- readRDS(archivos_geo[2])
  message("Trabajando sobre las bases con geofeatures.")
} else {
  train <- readr::read_csv(file.path(dir_raw, "train.csv"), show_col_types = FALSE)
  test <- readr::read_csv(file.path(dir_raw, "test.csv"), show_col_types = FALSE)
  message("Trabajando sobre las bases originales.")
}

# Patrones: 1 si la caracteristica se menciona; 0 si no se menciona. -------
patrones <- c(
  txt_balcon_terraza = "\\b(?:balcon(?:es)?|terraza(?:s)?)\\b",
  txt_remodelado = "\\b(?:remodelad[oa]s?|renovad[oa]s?|reformad[oa]s?|restaurad[oa]s?)\\b",
  txt_vista = "\\b(?:vista\\s+(?:panoramica|a la ciudad|exterior|espectacular|privilegiada|abierta)|excelente\\s+vista|hermosa\\s+vista)\\b",
  txt_seguridad = "\\b(?:vigilancia|porteria|conjunto cerrado|seguridad 24|control de acceso|recepcion 24)\\b",
  txt_iluminacion = "\\b(?:iluminacion natural|luz natural|muy iluminad[oa]|bien iluminad[oa]|ventanal(?:es)?|grandes ventanas)\\b",
  txt_amenidades = "\\b(?:gimnasio|gym|piscina|sauna|jacuzzi|coworking|co working|salon social|zona bbq|cancha de tenis|cancha de squash|turco)\\b",
  txt_parqueadero = "\\b(?:parqueader[oa]s?|garajes?|estacionamientos?)\\b",
  txt_ascensor = "\\b(?:ascensor(?:es)?|elevador(?:es)?)\\b"
)

# Negaciones directas: p. ej. "sin ascensor" o "no tiene parqueadero".
# No pretende resolver todas las negaciones del lenguaje natural.
negables <- paste0(
  "(?:balcon(?:es)?|terraza(?:s)?|vigilancia|porteria|gimnasio|piscina|",
  "sauna|jacuzzi|parqueader[oa]s?|garajes?|estacionamientos?|",
  "ascensor(?:es)?|elevador(?:es)?)"
)
patron_negacion <- paste0(
  "\\b(?:sin|no tiene|no cuenta con|carece de|no dispone de|no hay)\\s+",
  "(?:(?:un|una|de)\\s+)?", negables,
  "(?:\\s+(?:ni|o)\\s+", negables, ")*\\b"
)

# Construir las ocho variables --------------------------------------------
fun_features_texto <- function(datos) {
  titulo <- ifelse(is.na(datos$title), "", datos$title)
  descripcion <- ifelse(is.na(datos$description), "", datos$description)
  
  texto <- paste(titulo, descripcion)
  texto <- stringi::stri_trans_general(tolower(texto), "Latin-ASCII")
  texto <- stringr::str_squish(stringr::str_replace_all(texto, "[^a-z0-9]+", " "))
  texto <- stringr::str_replace_all(texto, patron_negacion, " ")
  
  nuevas <- as.data.frame(lapply(patrones, function(patron) {
    as.integer(stringr::str_detect(texto, patron))
  }))
  names(nuevas) <- names(patrones)
  cbind(datos, nuevas)
}

train_features <- fun_features_texto(train)
test_features <- fun_features_texto(test)

# Guardar y diagnosticar ---------------------------------------------------
saveRDS(train_features, file.path(dir_temp, "train_features.rds"))
saveRDS(test_features, file.path(dir_temp, "test_features.rds"))

variables <- names(patrones)
diagnostico <- data.frame(
  variable = variables,
  train_pct = round(100 * colMeans(train_features[variables]), 2),
  test_pct = round(100 * colMeans(test_features[variables]), 2)
)
print(diagnostico, row.names = FALSE)
message("Guardados: 03_temp/train_features.rds y 03_temp/test_features.rds")
