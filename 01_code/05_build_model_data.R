##########################################################
# 05_build_model_data.R
#
# Objetivo: dejar las bases listas para los modelos: tratar los valores
# faltantes y definir los conjuntos de predictores que comparten todos los
# scripts de modelos (common ground).
#
# Etapa del Problem Set: Etapa 1 (preparación) y Etapa 2 (conjuntos).
# Insumos:  03_temp/train_features.rds, 03_temp/test_features.rds y
#           03_temp/vocabulario_texto.rds (script 04).
# Salidas:  03_temp/train_model.rds y 03_temp/test_model.rds (mismo orden
#           de filas que train_features.rds, el de los folds del 05).
#           03_temp/conjuntos_predictores.rds: lista con los nombres de las
#           variables de cada conjunto (base, completo y ampliado) y las
#           rutas de la matriz TF-IDF que se suma en el conjunto ampliado.
#           02_outputs/tables/conjuntos_predictores.tex (y .csv): número de
#           predictores de cada bloque en cada conjunto, para las slides.
#
# Conjuntos de predictores:
#   - base (17): solo la información que trae Kaggle (características de
#     la vivienda, ubicación en coordenadas y fecha), con sus faltantes
#     imputados. Referencia: ¿cuánto se logra solo con los datos dados?
#   - completo: base + 10 variables externas (entorno, estrato y distancia
#     al empleo) + texto: superficie y baños completados con la
#     descripción, parqueaderos y piso mencionados, e indicadores de las
#     raíces presentes en al menos 5% de los anuncios. Es el conjunto con
#     el que se comparan las seis familias de algoritmos.
#   - ampliado: completo + la matriz TF-IDF de la descripción (hasta
#     12.000 unigramas y bigramas, script 04). Solo para los modelos que
#     manejan matrices dispersas grandes (Elastic Net y LightGBM): las
#     columnas de completo se convierten en matriz y se les pega la
#     matriz TF-IDF (tfidf_train.rds / tfidf_test.rds), en el mismo orden
#     de filas.
#
# Decisiones:
#   - Las superficies marcadas como extremas en el 02 se tratan como
#     faltantes (son errores de digitación, no información).
#   - Superficie y baños tienen dos versiones. La original (base) solo usa
#     las columnas de Kaggle. La "_completa" (completo y ampliado) usa
#     primero lo que dice la descripción (txt_area_m2, txt_banos) cuando
#     Kaggle no trae el dato; así, unas 10.800 propiedades tienen su
#     superficie real en lugar de la mediana. La superficie es el predictor
#     principal en un modelo hedónico y, si falta, el modelo tiende a dar
#     un precio "promedio", lo que sobreestima las viviendas pequeñas.
#   - Lo que sigue faltando se imputa con la mediana de su tipo de
#     vivienda (apartamento o casa), calculada solo con train, y se crea un
#     indicador falta_<variable> para que el modelo distinga un valor
#     imputado de uno observado (no reportar un dato puede ser
#     informativo en sí mismo).
#   - Simplificación aceptada: la mediana se calcula con todo train y no
#     dentro de cada fold de la validación cruzada. Con medianas por tipo
#     de vivienda, sobre miles de observaciones, el efecto es despreciable.
##########################################################

# Objetos que ya existían en el entorno antes de correr este script. Al
# final se borra solo lo que este script creó, sin tocar lo demás.
objetos_previos <- ls()

# Paquetes ----------------------------------------------------------------
if (!require(pacman)) install.packages("pacman")
pacman::p_load(dplyr, Matrix, knitr)

# Cargar datos -------------------------------------------------------------
train <- readRDS("03_temp/train_features.rds")
test  <- readRDS("03_temp/test_features.rds")
vocabulario_texto <- readRDS("03_temp/vocabulario_texto.rds")

# Nombres de los predictores por bloque ------------------------------------
vars_estructurales <- c("surface_total", "surface_covered", "rooms",
                        "bedrooms", "bathrooms", "property_type",
                        "lat", "lon", "month", "year")
vars_calidad <- c("flag_sin_superficie", "flag_superficie_inconsistente",
                  "flag_superficie_extrema")
vars_externas <- c("dist_transmilenio_m", "dist_via_principal_m",
                   "dist_centro_comercial_m", "dist_parque_m",
                   "dist_universidad_m", "num_restaurantes_cafes_500m",
                   "pct_zona_verde_500m", "pendiente_terreno_pct",
                   "estrato_manzana", "dist_centro_empleo_m")
vars_texto_cantidades <- c("txt_parqueaderos", "txt_piso")
raices_completo <- vocabulario_texto$completo

# Verificamos que existan todas las variables esperadas.
esperadas <- c(vars_estructurales, vars_calidad, vars_externas,
               vars_texto_cantidades, "txt_area_m2", "txt_banos",
               raices_completo)
faltan <- setdiff(esperadas, intersect(names(train), names(test)))
if (length(faltan) > 0) {
  stop("No se encontraron en train_features.rds o test_features.rds: ",
       paste(head(faltan, 10), collapse = ", "),
       ". Corra primero los scripts 02 a 04.")
}

# 1. Superficies extremas como faltantes ----------------------------------
# Mismos umbrales que flag_superficie_extrema en 02_clean_data.R. Se aplica
# por variable: si solo la superficie total es extrema, la construida se
# conserva. El indicador flag_superficie_extrema se mantiene como predictor.
fun_superficie_extrema_na <- function(.datos) {
  .datos |>
    mutate(
      surface_total = if_else(surface_total < 20 | surface_total > 2000,
                              NA_real_, surface_total),
      surface_covered = if_else(surface_covered < 20 | surface_covered > 1000,
                                NA_real_, surface_covered)
    )
}

# 2. Superficie y baños completados con la descripción ---------------------
fun_completar_con_texto <- function(.datos) {
  .datos |>
    mutate(
      surface_total_completa = coalesce(surface_total, txt_area_m2),
      surface_covered_completa = coalesce(surface_covered, txt_area_m2),
      bathrooms_completa = coalesce(bathrooms, txt_banos),
      superficie_desde_texto = as.integer(
        is.na(surface_total) & is.na(surface_covered) & !is.na(txt_area_m2))
    )
}

train <- train |> fun_superficie_extrema_na() |> fun_completar_con_texto()
test  <- test  |> fun_superficie_extrema_na() |> fun_completar_con_texto()

# 3. Indicadores de faltante e imputación por mediana ----------------------
vars_completas <- c("surface_total_completa", "surface_covered_completa",
                    "bathrooms_completa")
vars_numericas <- c(setdiff(vars_estructurales, "property_type"),
                    vars_completas, vars_externas, vars_texto_cantidades)
# Solo se imputan las que tienen algún faltante en train o en test.
vars_con_na <- vars_numericas[
  sapply(vars_numericas, function(v) anyNA(train[[v]]) || anyNA(test[[v]]))
]

# Medianas por tipo de vivienda, calculadas solo en train.
medianas <- train |>
  group_by(property_type) |>
  summarise(across(all_of(vars_con_na), ~ median(.x, na.rm = TRUE)),
            .groups = "drop")

fun_imputar <- function(.datos) {
  for (v in vars_con_na) {
    .datos[[paste0("falta_", v)]] <- as.integer(is.na(.datos[[v]]))
    mediana_fila <- medianas[[v]][match(.datos$property_type,
                                        medianas$property_type)]
    .datos[[v]] <- if_else(is.na(.datos[[v]]), mediana_fila, .datos[[v]])
  }
  .datos
}
train <- fun_imputar(train)
test  <- fun_imputar(test)

# Tipo de vivienda como factor, con los mismos niveles en train y test.
train$property_type <- factor(train$property_type)
test$property_type  <- factor(test$property_type,
                              levels = levels(train$property_type))

# 4. Conjuntos de predictores ----------------------------------------------
fun_con_faltantes <- function(.vars) {
  c(.vars, intersect(paste0("falta_", .vars), paste0("falta_", vars_con_na)))
}

# En completo, la superficie y los baños originales se reemplazan por sus
# versiones completadas con la descripción.
vars_estructurales_completas <- c(
  setdiff(vars_estructurales,
          c("surface_total", "surface_covered", "bathrooms")),
  vars_completas
)

conjuntos_predictores <- list(
  base = c(fun_con_faltantes(vars_estructurales), vars_calidad),
  completo = c(fun_con_faltantes(vars_estructurales_completas), vars_calidad,
               "superficie_desde_texto",
               fun_con_faltantes(vars_externas),
               fun_con_faltantes(vars_texto_cantidades),
               raices_completo),
  # Ampliado usa las mismas columnas que completo más la matriz TF-IDF.
  ampliado = NULL,
  tfidf_ampliado = c(train = "03_temp/tfidf_train.rds",
                     test = "03_temp/tfidf_test.rds")
)
conjuntos_predictores$ampliado <- conjuntos_predictores$completo

# Verificación: la matriz TF-IDF debe tener las mismas filas que las bases.
tfidf_train <- readRDS(conjuntos_predictores$tfidf_ampliado["train"])
filas_tfidf <- c(nrow(tfidf_train),
                 nrow(readRDS(conjuntos_predictores$tfidf_ampliado["test"])))
if (!identical(filas_tfidf, c(nrow(train), nrow(test)))) {
  stop("La matriz TF-IDF no tiene las mismas filas que train/test. ",
       "Corra de nuevo 04_build_text_features.R.")
}

# Verificación final: ningún predictor puede quedar con NA.
todos <- unique(c(conjuntos_predictores$base,
                  conjuntos_predictores$completo))
na_restantes <- c(colSums(is.na(train[todos])), colSums(is.na(test[todos])))
if (any(na_restantes > 0)) {
  stop("Quedaron faltantes en: ",
       paste(unique(names(na_restantes)[na_restantes > 0]), collapse = ", "))
}

# 5. Tabla de conjuntos para las slides -------------------------------------
# Cuenta cuántos predictores de cada bloque tiene cada conjunto. Los números
# salen de conjuntos_predictores, así que se actualizan solos si cambia algo.
fun_bloque <- function(.vars) {
  case_when(
    startsWith(.vars, "raiz_") ~ "Texto: raíces",
    .vars %in% vars_texto_cantidades ~ "Texto: cantidades",
    .vars %in% vars_externas ~ "Ubicación y entorno",
    .vars %in% c(vars_calidad, "superficie_desde_texto") |
      startsWith(.vars, "falta_") ~ "Calidad del dato",
    TRUE ~ "Vivienda"
  )
}
bloques <- tibble(
  Bloque = c("Vivienda", "Calidad del dato", "Ubicación y entorno",
             "Texto: cantidades", "Texto: raíces", "Texto: TF-IDF"),
  Contenido = c(
    "Superficie, baños, ambientes, habitaciones, tipo, coordenadas, fecha",
    "Indicadores de superficie problemática y de dato imputado",
    "Estrato, distancia al empleo, TransMilenio, vías, servicios, verde, pendiente",
    "Parqueaderos y piso mencionados en la descripción",
    "Raíces presentes en al menos 5% de los anuncios",
    "Unigramas y bigramas de la descripción"
  )
)
fun_contar <- function(.vars) {
  as.integer(table(factor(fun_bloque(.vars), levels = bloques$Bloque)))
}
tabla_conjuntos <- bloques |>
  mutate(
    Base = fun_contar(conjuntos_predictores$base),
    Completo = fun_contar(conjuntos_predictores$completo),
    Ampliado = fun_contar(conjuntos_predictores$ampliado) +
      if_else(Bloque == "Texto: TF-IDF", ncol(tfidf_train), 0L)
  )
tabla_conjuntos <- bind_rows(
  tabla_conjuntos,
  tibble(Bloque = "Total", Contenido = "",
         Base = sum(tabla_conjuntos$Base),
         Completo = sum(tabla_conjuntos$Completo),
         Ampliado = sum(tabla_conjuntos$Ampliado))
)

dir.create("02_outputs/tables", recursive = TRUE, showWarnings = FALSE)
write.csv(tabla_conjuntos, "02_outputs/tables/conjuntos_predictores.csv",
          row.names = FALSE)
fun_formato <- function(.x) {
  if_else(.x == 0, "---", format(.x, big.mark = ".", decimal.mark = ",",
                                 trim = TRUE))
}
writeLines(kable(tabla_conjuntos |>
                   mutate(across(c(Base, Completo, Ampliado), fun_formato)),
                 format = "latex", booktabs = TRUE, linesep = "",
                 align = c("l", "l", "r", "r", "r")),
           "02_outputs/tables/conjuntos_predictores.tex")

# Guardar ------------------------------------------------------------------
saveRDS(train, "03_temp/train_model.rds")
saveRDS(test, "03_temp/test_model.rds")
saveRDS(conjuntos_predictores, "03_temp/conjuntos_predictores.rds")

message("Bases para modelar guardadas en 03_temp/. Número de predictores: ",
        "base = ", length(conjuntos_predictores$base),
        "; completo = ", length(conjuntos_predictores$completo),
        "; ampliado = completo + matriz TF-IDF.")
message("Propiedades con superficie tomada de la descripción: train = ",
        sum(train$superficie_desde_texto), "; test = ",
        sum(test$superficie_desde_texto), ".")
print(data.frame(
  variable = vars_con_na,
  pct_imputado_train = round(100 * colMeans(train[paste0("falta_", vars_con_na)]), 1),
  pct_imputado_test = round(100 * colMeans(test[paste0("falta_", vars_con_na)]), 1)
), row.names = FALSE)

# Limpiamos el entorno: borramos solo los objetos que creó este script.
rm(list = setdiff(ls(), objetos_previos))
