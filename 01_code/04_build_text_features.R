##########################################################
# 04_build_text_features.R
#
# Objetivo: crear variables a partir del titulo y la descripcion.
# Insumos:  03_temp/train_geo.rds y 03_temp/test_geo.rds.
# Salidas:  03_temp/train_features.rds y 03_temp/test_features.rds.
#
# Se mantienen los ocho indicadores originales y se agregan:
#   txt_balcon, txt_terraza, txt_antiguedad_anios.
# txt_parqueadero ya existia.
##########################################################
objetos_previos <- ls()

# Paquetes ----------------------------------------------------------------
if (!require(pacman)) install.packages("pacman")
pacman::p_load(stringr, stringi)

# Cargar datos ------------------------------------------------------------
dir_temp <- "03_temp"
train <- readRDS(file.path(dir_temp, "train_geo.rds"))
test  <- readRDS(file.path(dir_temp, "test_geo.rds"))

# Patrones: 1 si se menciona la caracteristica, 0 si no --------------------
patrones <- c(
  txt_balcon_terraza = "\\b(?:balcon(?:es)?|terraza(?:s)?)\\b",
  txt_balcon = "\\b(?:balcon(?:es)?)\\b",
  txt_terraza = "\\b(?:terraza(?:s)?)\\b",
  txt_remodelado = "\\b(?:remodelad[oa]s?|renovad[oa]s?|reformad[oa]s?|restaurad[oa]s?)\\b",
  txt_vista = "\\b(?:vista\\s+(?:panoramica|a la ciudad|exterior|espectacular|privilegiada|abierta)|excelente\\s+vista|hermosa\\s+vista)\\b",
  txt_seguridad = "\\b(?:vigilancia|porteria|conjunto cerrado|seguridad 24|control de acceso|recepcion 24)\\b",
  txt_iluminacion = "\\b(?:iluminacion natural|luz natural|muy iluminad[oa]|bien iluminad[oa]|ventanal(?:es)?|grandes ventanas)\\b",
  txt_amenidades = "\\b(?:gimnasio|gym|piscina|sauna|jacuzzi|coworking|co working|salon social|zona bbq|cancha de tenis|cancha de squash|turco)\\b",
  txt_parqueadero = "\\b(?:parqueader[oa]s?|garajes?|estacionamientos?)\\b",
  txt_ascensor = "\\b(?:ascensor(?:es)?|elevador(?:es)?)\\b"
)

# Quitar negaciones directas antes de buscar menciones positivas ----------
negables <- paste0(
  "(?:balcon(?:es)?|terraza(?:s)?|vigilancia|porteria|gimnasio|piscina|",
  "sauna|jacuzzi|parqueader[oa]s?|garajes?|estacionamientos?|",
  "ascensor(?:es)?|elevador(?:es)?)"
)
patron_negacion <- paste0(
  "\\b(?:sin|no tiene|no cuenta con|no incluye|carece de|no dispone de|no hay)\\s+",
  "(?:(?:un|una|de)\\s+)?", negables,
  "(?:\\s+(?:ni|o|y)\\s+", negables, ")*\\b"
)

# Construir variables -----------------------------------------------------
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
  
  # Antiguedad explicita: "10 anos de antiguedad", "antiguedad de 10 anos"
  # o "construido hace 10 anos". "Para estrenar" equivale a 0.
  edad1 <- stringr::str_match(texto,
                              "\\b(?:antiguedad\\s*(?:de\\s*)?|construid[oa]\\s+hace\\s*)(\\d{1,3})\\s*anos?\\b")[, 2]
  edad2 <- stringr::str_match(texto,
                              "\\b(\\d{1,3})\\s*anos?\\s*(?:de\\s*)?antiguedad\\b")[, 2]
  edad <- as.integer(ifelse(!is.na(edad1), edad1, edad2))
  edad[is.na(edad) & stringr::str_detect(texto, "\\b(?:para|a) estrenar\\b")] <- 0L
  nuevas$txt_antiguedad_anios <- edad
  
  cbind(datos, nuevas)
}

train_features <- fun_features_texto(train)
test_features <- fun_features_texto(test)

# Guardar y diagnosticar --------------------------------------------------
saveRDS(train_features, file.path(dir_temp, "train_features.rds"))
saveRDS(test_features, file.path(dir_temp, "test_features.rds"))

variables <- names(patrones)
diagnostico <- data.frame(
  variable = variables,
  train_pct = round(100 * colMeans(train_features[variables]), 2),
  test_pct = round(100 * colMeans(test_features[variables]), 2)
)
print(diagnostico, row.names = FALSE)
message("Cobertura de antiguedad: train ",
        round(mean(!is.na(train_features$txt_antiguedad_anios)) * 100, 1),
        "%, test ", round(mean(!is.na(test_features$txt_antiguedad_anios)) * 100, 1), "%")
message("Guardados: 03_temp/train_features.rds y 03_temp/test_features.rds")

rm(list = setdiff(ls(), objetos_previos))

