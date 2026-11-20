##########################################################
# 04_build_text_features.R
#
# Objetivo: construir predictores a partir de la descripción de cada
# anuncio (requisito: al menos cuatro), con el método de la clase de texto
# como datos (Lecture 15): limpieza, bolsa de palabras (bag of words),
# stemming y matriz documento-término.
#
# Etapa del Problem Set: Etapa 1: Understanding and Expanding the Data.
# Insumos:  03_temp/train_geo.rds y 03_temp/test_geo.rds.
# Salidas:  03_temp/train_features.rds y 03_temp/test_features.rds.
#           03_temp/vocabulario_texto.rds: raíces de cada conjunto de
#           predictores (lo usa 05_build_model_data.R).
#           03_temp/tfidf_train.rds y 03_temp/tfidf_test.rds: matrices
#           TF-IDF (dispersas) de la descripción, para el conjunto ampliado.
#
# Variables generadas:
#   - raiz_<raíz>: 1 si la descripción usa alguna palabra con esa raíz y 0
#     si no (p. ej. raiz_chimene, raiz_parqueader, raiz_remodel). Se crean
#     para las raíces presentes en al menos el 1% de los anuncios.
#   - txt_area_m2: área en m² mencionada en la descripción.
#   - txt_banos, txt_parqueaderos, txt_piso: número de baños, de
#     parqueaderos y piso mencionados en la descripción.
#   Las cuatro numéricas valen NA si la descripción no las menciona.
#   - Matriz TF-IDF de unigramas y bigramas (hasta 12.000 términos), con
#     fun_construir_tfidf() de Andrés. Se guarda aparte porque solo la
#     pueden usar los modelos que manejan matrices dispersas grandes
#     (Elastic Net y LightGBM).
#
# Decisiones:
#   - Solo se usa la descripción (recomendación del profesor): el título
#     es corto (mediana de 5 palabras) y repite la descripción.
#   - Limpieza común con fun_preprocesar_texto() (functions/): minúsculas,
#     sin tildes ni signos, restos de HTML corregidos ("ba ntilde o" ->
#     "bano"), palabras que perdieron la ñ o la tilde ("bao" -> "bano",
#     "ubicacin" -> "ubicacion") y sin frases publicitarias de las
#     inmobiliarias ("contactanos", "visita nuestra pagina web").
#   - Se eliminan las negaciones directas ("sin ascensor", "no tiene
#     parqueadero ni garaje") antes de contar palabras, para no marcar
#     como presente lo que el anuncio dice que no tiene.
#   - Stemming con el algoritmo Snowball para español (SnowballC), que
#     junta género y número (remodelado, remodelada -> remodel). Como el
#     texto viene sin tildes, las palabras en "-cion" se pasan al plural
#     antes del stemming para que singular y plural den la misma raíz
#     (habitacion y habitaciones -> habit).
#   - Se quitan las stopwords del español (lista Snowball) y las palabras
#     de dos letras o menos.
#   - Indicadores (1/0) en lugar de conteos o TF-IDF, como en Gentzkow y
#     Shapiro (2010, Lecture 15): mencionar algo importa más que cuántas
#     veces. En una prueba, 149 indicadores de raíz predijeron log(precio)
#     mucho mejor que 30 componentes de PCA sobre TF-IDF (R² fuera de
#     muestra 0,32 frente a 0,15).
#   - El vocabulario (raíces en al menos 1% o 5% de los anuncios) se
#     calcula con train y test juntos. No usa el precio, así que no filtra
#     información de la variable a predecir.
#   - Las cantidades se extraen con reglas de plausibilidad (p. ej. área
#     entre 20 y 2.000 m², sin tomar "terraza de 30 m" ni "a 300 metros";
#     baños como suma de las menciones; piso con ordinales).
#     Donde la base trae la superficie construida, el área extraída
#     coincide con ella (±10%) en el 85% de los casos.
#   - TF-IDF: vocabulario e IDF aprendidos solo con train y aplicados a
#     test (mismas columnas). Conserva las stopwords para que bigramas
#     como "sin_ascensor" capturen las negaciones. Con validación cruzada
#     espacial, solo el texto predice mejor con el TF-IDF que con las
#     raíces (R² 0,41 frente a 0,29), aunque con muchas más columnas.
##########################################################

# Objetos que ya existían en el entorno antes de correr este script. Al
# final se borra solo lo que este script creó, sin tocar lo demás.
objetos_previos <- ls()

# Paquetes ----------------------------------------------------------------
if (!require(pacman)) install.packages("pacman")
pacman::p_load(dplyr, stringr, stringi, tidytext, SnowballC, stopwords,
               text2vec, Matrix)

# Funciones de texto (limpieza común y TF-IDF).
source("01_code/functions/fun_preprocesar_texto.R")
source("01_code/functions/fun_construir_tfidf.R")

# Parámetros ---------------------------------------------------------------
# Porcentaje mínimo de anuncios en los que debe aparecer una raíz para
# entrar a cada conjunto de predictores (ver 05_build_model_data.R).
umbral_completo <- 5
umbral_ampliado <- 1

# Cargar datos -------------------------------------------------------------
dir_temp <- "03_temp"
train <- readRDS(file.path(dir_temp, "train_geo.rds"))
test  <- readRDS(file.path(dir_temp, "test_geo.rds"))

# 1. Limpieza del texto -----------------------------------------------------
# Se usa fun_preprocesar_texto() (functions/fun_preprocesar_texto.R).

# 2. Negaciones directas ---------------------------------------------------
# No pretende resolver todas las negaciones del lenguaje natural.
negables <- paste0(
  "(?:balcon(?:es)?|terrazas?|vigilancia|porteria|gimnasio|piscina|",
  "sauna|jacuzzi|parqueader[oa]s?|garajes?|estacionamientos?|",
  "ascensor(?:es)?|elevador(?:es)?|chimenea|estudio|deposito)"
)
patron_negacion <- paste0(
  "\\b(?:sin|no tiene|no cuenta con|no incluye|carece de|no dispone de|",
  "no hay)\\s+(?:(?:un|una|de)\\s+)?", negables,
  "(?:\\s+(?:ni|o|y)\\s+", negables, ")*\\b"
)

# 3. Cantidades mencionadas en el texto -------------------------------------
numeros_palabra <- c("\\bun\\b" = "1", "\\buna\\b" = "1", "\\buno\\b" = "1",
                     "\\bdos\\b" = "2", "\\btres\\b" = "3",
                     "\\bcuatro\\b" = "4", "\\bcinco\\b" = "5",
                     "\\bseis\\b" = "6")
unidad_area <- paste0("(?:m2|mt2|mts2|mtr2|mtrs2|m 2|mts 2|metros cuadrados|",
                      "metros2|mts|mtrs|metros|mt)")
patron_area <- paste0("\\b(\\d{2,4})\\s?", unidad_area, "\\b")
# Contextos en los que el número no es el área de la vivienda.
contexto_excluido <- paste0(
  "\\b(?:terraz\\w*|balcon\\w*|patio\\w*|deposito\\w*|jardin\\w*|a|",
  "a unos|a solo|a tan solo)\\s*(?:de\\s*)?(?:aprox\\w*\\s*)?$"
)

fun_area <- function(.x) {
  pos <- str_locate_all(.x, patron_area)[[1]]
  for (k in seq_len(nrow(pos))) {
    antes <- substr(.x, max(1, pos[k, 1] - 30), pos[k, 1] - 1)
    if (str_detect(antes, contexto_excluido)) next
    valor <- as.numeric(str_match(substr(.x, pos[k, 1], pos[k, 2]),
                                  patron_area)[, 2])
    if (valor >= 20 && valor <= 2000) return(valor)
  }
  NA_real_
}

fun_numero <- function(.x, .patron, .min, .max) {
  valor <- as.numeric(str_match(.x, .patron)[, 2])
  valor[valor < .min | valor > .max] <- NA
  valor
}

# 4. Construir las variables -----------------------------------------------
datos <- bind_rows(train = train, test = test, .id = "base")
datos$doc <- seq_len(nrow(datos))
texto <- fun_preprocesar_texto(datos$description)
texto <- str_replace_all(texto, patron_negacion, " ")

# Cantidades: se convierten "dos", "tres"... a dígitos.
texto_num <- str_replace_all(texto, numeros_palabra)
datos$txt_area_m2 <- vapply(texto_num, fun_area, numeric(1),
                            USE.NAMES = FALSE)
# Baños: se suman las menciones ("1 bano social y 2 banos" = 3), porque los
# anuncios suelen listarlos por separado. Coincide con bathrooms (donde hay
# dato) más a menudo que tomar solo la primera mención (74% frente a 71%).
datos$txt_banos <- vapply(
  str_match_all(texto_num, "\\b(\\d)\\s?banos?\\b"),
  function(m) if (nrow(m)) sum(as.numeric(m[, 2])) else NA_real_,
  numeric(1)
)
datos$txt_banos[datos$txt_banos < 1 | datos$txt_banos > 8] <- NA
datos$txt_parqueaderos <- fun_numero(
  texto_num, "\\b(\\d)\\s?(?:parqueaderos?|garajes?|estacionamientos?)\\b",
  1, 6)
# Piso: se usa el texto sin convertir "un", "dos"... a dígitos (si no,
# "piso un parqueadero" se leería como piso 1) y se traducen los ordinales
# ("segundo piso" = piso 2). "Cuarto piso" no se traduce porque "cuarto"
# también significa habitación. Se descartan números seguidos de otra cosa
# ("piso 1 parqueadero", "piso 2 banos").
ordinales_piso <- c(
  "\\bprimer(?:o)?\\s+piso\\b" = "piso 1", "\\bsegundo\\s+piso\\b" = "piso 2",
  "\\btercer(?:o)?\\s+piso\\b" = "piso 3", "\\bquinto\\s+piso\\b" = "piso 5",
  "\\bsexto\\s+piso\\b" = "piso 6", "\\bseptimo\\s+piso\\b" = "piso 7",
  "\\boctavo\\s+piso\\b" = "piso 8", "\\bnoveno\\s+piso\\b" = "piso 9",
  "\\bdecimo\\s+piso\\b" = "piso 10"
)
texto_piso <- str_replace_all(texto, ordinales_piso)
datos$txt_piso <- coalesce(
  fun_numero(texto_piso, paste0("\\bpiso\\s?(\\d{1,2})\\b(?!\\s?(?:parqueader|",
                                "garaj|bano|habitac|alcob|deposit|m2|mts|metros))"),
             1, 40),
  fun_numero(texto_piso, "\\b(\\d{1,2})\\s?piso\\b", 1, 40)
)

# Bolsa de palabras: tokens, sin números, stopwords ni palabras cortas;
# "-cion" al plural para que el stemming junte singular y plural.
tokens <- tibble(doc = datos$doc, texto = texto) |>
  unnest_tokens(palabra, texto) |>
  filter(!str_detect(palabra, "^[0-9]+$"), nchar(palabra) > 2,
         !palabra %in% stopwords::stopwords("es", source = "snowball")) |>
  mutate(palabra = str_replace(palabra, "cion$", "ciones"),
         raiz = wordStem(palabra, language = "spanish")) |>
  distinct(doc, raiz)

# Frecuencia de cada raíz: porcentaje de anuncios que la usan.
frecuencia <- tokens |>
  count(raiz, name = "n_anuncios") |>
  mutate(pct_anuncios = 100 * n_anuncios / nrow(datos)) |>
  arrange(desc(n_anuncios))

vocabulario <- frecuencia$raiz[frecuencia$pct_anuncios >= umbral_ampliado]

# Indicadores 1/0 por raíz (matriz documento-término binaria).
indicadores <- matrix(0L, nrow = nrow(datos), ncol = length(vocabulario),
                      dimnames = list(NULL, paste0("raiz_", vocabulario)))
presentes <- tokens |> filter(raiz %in% vocabulario)
indicadores[cbind(presentes$doc, match(presentes$raiz, vocabulario))] <- 1L
datos <- bind_cols(datos, as.data.frame(indicadores))

# Vocabulario de cada conjunto, para 05_build_model_data.R.
vocabulario_texto <- list(
  completo = paste0("raiz_", frecuencia$raiz[
    frecuencia$pct_anuncios >= umbral_completo]),
  ampliado = paste0("raiz_", vocabulario),
  frecuencia = frecuencia
)

# 5. Matriz TF-IDF de la descripción ---------------------------------------
# fun_construir_tfidf() limpia el texto con fun_preprocesar_texto(), aprende
# el vocabulario y el IDF con train y aplica lo mismo a test.
tfidf <- fun_construir_tfidf(train$description, test$description)
if (nrow(tfidf$train) != nrow(train) || nrow(tfidf$nuevo) != nrow(test) ||
    !identical(colnames(tfidf$train), colnames(tfidf$nuevo))) {
  stop("Las matrices TF-IDF no coinciden con las filas o columnas de entrada.")
}
saveRDS(tfidf$train, file.path(dir_temp, "tfidf_train.rds"))
saveRDS(tfidf$nuevo, file.path(dir_temp, "tfidf_test.rds"))
message("TF-IDF: ", ncol(tfidf$train), " términos (unigramas y bigramas).")

# 6. Guardar y diagnosticar -------------------------------------------------
train_features <- datos |> filter(base == "train") |> select(-base, -doc)
test_features  <- datos |> filter(base == "test") |> select(-base, -doc)
saveRDS(train_features, file.path(dir_temp, "train_features.rds"))
saveRDS(test_features, file.path(dir_temp, "test_features.rds"))
saveRDS(vocabulario_texto, file.path(dir_temp, "vocabulario_texto.rds"))

message("Raíces con indicador: ", length(vocabulario),
        " (en >= ", umbral_ampliado, "% de anuncios); de ellas ",
        length(vocabulario_texto$completo), " en >= ", umbral_completo, "%.")
variables_num <- c("txt_area_m2", "txt_banos", "txt_parqueaderos", "txt_piso")
print(data.frame(
  variable = variables_num,
  pct_con_dato_train = round(100 * colMeans(!is.na(train_features[variables_num])), 1),
  pct_con_dato_test = round(100 * colMeans(!is.na(test_features[variables_num])), 1)
), row.names = FALSE)
message("Guardados en 03_temp/: train_features.rds, test_features.rds, ",
        "vocabulario_texto.rds, tfidf_train.rds y tfidf_test.rds")

# Limpiamos el entorno: borramos solo los objetos que creó este script.
rm(list = setdiff(ls(), objetos_previos))
