##########################################################
# 04_build_text_features.R
#
# Objetivo: construir predictores a partir del título y la descripción de
# cada anuncio (requisito: al menos cuatro).
#
# Etapa del Problem Set: Etapa 1: Understanding and Expanding the Data.
# Insumos:  03_temp/train_geo.rds y 03_temp/test_geo.rds.
# Salidas:  03_temp/train_features.rds y 03_temp/test_features.rds.
#
# Variables generadas (8): indicadoras que valen 1 si el anuncio menciona
# la característica y 0 si no: txt_balcon_terraza, txt_remodelado,
# txt_vista, txt_seguridad, txt_iluminacion, txt_amenidades,
# txt_parqueadero y txt_ascensor.
#
# Decisiones:
#   - El texto se pasa a minúsculas, sin tildes ni signos de puntuación,
#     para que "Balcón" y "balcon" cuenten igual.
#   - Se eliminan las negaciones directas ("sin ascensor", "no tiene
#     parqueadero ni garaje") antes de buscar las palabras, para no marcar
#     como presente una característica que el anuncio dice que no tiene.
##########################################################

# Objetos que ya existían en el entorno antes de correr este script. Al
# final se borra solo lo que este script creó, sin tocar lo demás.
objetos_previos <- ls()

# Paquetes ----------------------------------------------------------------
paquetes <- c("stringr", "stringi")
for (paquete in paquetes) {
  if (!require(paquete, character.only = TRUE)) {
    install.packages(paquete)
    library(paquete, character.only = TRUE)
  }
}

# Cargar datos -------------------------------------------------------------
dir_temp <- "03_temp"

train <- readRDS(file.path(dir_temp, "train_geo.rds"))
test  <- readRDS(file.path(dir_temp, "test_geo.rds"))

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


# Limpiamos el entorno: borramos solo los objetos que creó este script.
# Todo lo necesario quedó guardado en 03_temp/.
rm(list = setdiff(ls(), objetos_previos))
rm(objetos_previos)