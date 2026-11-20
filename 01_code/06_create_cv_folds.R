##########################################################
# 06_create_cv_folds.R
#
# Objetivo: crear las particiones de validación cruzada que comparten
# todos los modelos (common ground), eligiendo con evidencia el esquema de
# CV espacial que mejor anticipa el error en Chapinero.
#
# Etapa del Problem Set: Etapa 2 (elegir hiperparámetros con CV espacial)
# y Etapa 3 (comparar CV espacial frente a CV aleatoria).
# Insumos:  03_temp/train_features.rds y 03_temp/test_features.rds;
#           03_temp/train_model.rds y conjuntos_predictores.rds (script 05).
# Salidas:  03_temp/cv_folds.rds: lista con los índices de VALIDACIÓN de
#           cada fold, en el mismo orden de filas de train: "espacial" (el
#           esquema elegido), "aleatoria" y "espacial_k5" (comparación,
#           Etapa 3). Sirve tal cual para lightgbm::lgb.cv() y, con su
#           complemento, para trainControl(index = ...) de caret.
#           03_temp/similitud_chapinero.rds: similitud de cada propiedad de
#           train con las de Chapinero (propensity score).
#           02_outputs/tables/zonas_similitud_chapinero.csv: zonas de train,
#           su nombre y su similitud con Chapinero.
#           02_outputs/tables/seleccion_cv.csv y .tex: comparación de
#           esquemas de CV (resumen); seleccion_cv_detalle.csv (detalle).
#
# Decisiones:
#   - Similitud con Chapinero: regresión logística que distingue test
#     (Chapinero) de train con las características de la vivienda y de su
#     entorno, sin coordenadas ni nombres de barrios ("adversarial
#     validation"). Se calcula fuera de muestra y no usa el precio.
#   - Elección del esquema de CV espacial: la mejor CV no es la que da
#     menos error, sino la que mejor anticipa el error real en Chapinero.
#     Como ese error no se conoce, se simula: se apartan de train las 3
#     zonas más parecidas a Chapinero ("pseudo-Chapineros"), dejando 12% de
#     cada una en train para que queden tan cerca de train como Chapinero
#     (a unos 90 m de la propiedad de train más cercana). Se compara el
#     error que promete cada esquema con el error verdadero en la zona,
#     con un modelo flexible (Random Forest) y uno simple (regresión
#     lineal) sobre el conjunto completo. Gana el esquema espacial con la
#     menor brecha absoluta promedio.
#   - CV aleatoria: 5 folds equilibrados por rangos de precio, para que
#     cada fold tenga precios bajos, medios y altos. Se guarda para la
#     comparación de la Etapa 3.
#   - En los esquemas espaciales, las propiedades con las mismas
#     coordenadas (el mismo edificio o el mismo anuncio repetido) quedan
#     siempre en el mismo fold.
#   - La simulación tarda (unos 20 a 40 minutos): entrena 246 modelos de
#     prueba.
##########################################################

# Objetos que ya existían en el entorno antes de correr este script. Al
# final se borra solo lo que este script creó, sin tocar lo demás.
objetos_previos <- ls()

# Parametros ---------------------------------------------------------------
k <- 5
semilla <- 2026
crs_m <- 3116  # Coordenadas proyectadas en metros, como en el script 03.
n_zonas <- 20           # Zonas de train para buscar pseudo-Chapineros.
n_pseudo <- 3           # Número de pseudo-Chapineros.
min_tamano_zona <- 1000 # Tamaño mínimo de una zona para usarla.
prop_conservar <- 0.12  # Fracción de cada pseudo-Chapinero que queda en train.

# Datos --------------------------------------------------------------------
if (!require(pacman)) install.packages("pacman")
pacman::p_load(sf, dplyr, ranger, tidytext, knitr)

train <- readRDS("03_temp/train_features.rds")

# 1. Similitud con Chapinero (propensity score) ----------------------------
# Probabilidad de que cada propiedad "parezca" de Chapinero según sus
# características (sin coordenadas ni nombres de barrios). Sirve para
# elegir zonas de train parecidas a Chapinero al comparar esquemas de CV y,
# en la Etapa 4, para medir el error en propiedades parecidas a las que la
# empresa quiere comprar. No usa el precio.
source("01_code/functions/fun_similitud_chapinero.R")
test <- readRDS("03_temp/test_features.rds")

fun_preparar_similitud <- function(.datos) {
  .datos |>
    mutate(
      apartamento = as.integer(property_type == "Apartamento"),
      # Superficie: Kaggle y, si falta, la mencionada en la descripción.
      log_superficie = log(coalesce(surface_covered, surface_total,
                                    txt_area_m2)),
      banos = coalesce(bathrooms, txt_banos),
      across(any_of(c("dist_centro_empleo_m", "dist_transmilenio_m",
                      "dist_via_principal_m", "dist_centro_comercial_m",
                      "dist_parque_m", "dist_universidad_m",
                      "num_restaurantes_cafes_500m")),
             log1p, .names = "log_{.col}")
    )
}
# Raíces que describen la vivienda (no lugares).
raices_vivienda <- paste0("raiz_", c(
  "chimene", "estudi", "deposit", "conjunt", "ascensor", "exterior",
  "vigil", "remodel", "gimnasi", "bbq", "terraz", "balcon", "duplex",
  "closet", "vesti", "mader", "lamin", "piscin", "cubiert", "comunal",
  "infantil", "independient", "integral"))
vars_similitud <- c(
  "apartamento", "log_superficie", "bedrooms", "banos", "rooms",
  "txt_parqueaderos", "txt_piso", "estrato_manzana",
  "log_dist_centro_empleo_m", "log_dist_transmilenio_m",
  "log_dist_via_principal_m", "log_dist_centro_comercial_m",
  "log_dist_parque_m", "log_dist_universidad_m",
  "log_num_restaurantes_cafes_500m", "pct_zona_verde_500m",
  "pendiente_terreno_pct", raices_vivienda)

base_similitud_train <- fun_preparar_similitud(train)
base_similitud_test <- fun_preparar_similitud(test)
faltan_similitud <- setdiff(vars_similitud, names(base_similitud_train))
if (length(faltan_similitud) > 0) {
  stop("Faltan variables para la similitud: ",
       paste(faltan_similitud, collapse = ", "),
       ". Corra primero los scripts 02 a 04.")
}
similitud <- fun_similitud_chapinero(base_similitud_train,
                                     base_similitud_test, vars_similitud,
                                     .semilla = semilla)
saveRDS(data.frame(property_id = train$property_id,
                   similitud = similitud$train),
        "03_temp/similitud_chapinero.rds")

message("Similitud con Chapinero: AUC fuera de muestra = ",
        round(similitud$auc, 3), "; media en test = ",
        round(mean(similitud$test), 3), "; media en train = ",
        round(mean(similitud$train), 3), ".")
message("Variables que más distinguen a Chapinero (coeficiente estandarizado):")
print(head(similitud$coeficientes, 8), row.names = FALSE)

# 2. Zonas de train y su similitud con Chapinero ---------------------------
# Se divide train en zonas compactas (k-means sobre las coordenadas en
# metros), se les pone nombre con los barrios que más se repiten en los
# títulos de sus anuncios y se calcula su similitud promedio con Chapinero.
puntos <- sf::st_as_sf(train, coords = c("lon", "lat"), crs = 4326)
xy <- sf::st_coordinates(sf::st_transform(puntos, crs_m))

set.seed(semilla)
zona <- suppressWarnings(stats::kmeans(xy, centers = n_zonas, nstart = 10,
                                       iter.max = 50)$cluster)

# Nombre de cada zona: pares de palabras del título más característicos de
# la zona (TF-IDF por zona), sin palabras genéricas de los anuncios.
palabras_genericas <- c(
  "apartamento", "apartamentos", "apto", "casa", "venta", "vende", "vendo",
  "se", "en", "de", "del", "la", "el", "los", "las", "y", "a", "al", "con",
  "para", "bogota", "d", "c", "barrio", "sector", "norte", "sur", "lindo",
  "linda", "hermoso", "hermosa", "excelente", "amplio", "amplia",
  "remodelado", "oportunidad", "penthouse", "duplex", "ph", "m2", "piso",
  "alcobas", "habitaciones", "parqueadero", "conjunto", "cerrado",
  "edificio", "exterior", "estrenar", "ubicado", "ubicacion", "vista",
  "calle", "carrera", "cra", "cl", "no", "cod", "id")
nombres_zona <- tibble(zona = zona,
                       titulo = stringi::stri_trans_general(
                         tolower(coalesce(train$title, "")), "Latin-ASCII")) |>
  unnest_tokens(par, titulo, token = "ngrams", n = 2) |>
  filter(!is.na(par), !grepl("[0-9]", par)) |>
  filter(!vapply(strsplit(par, " "),
                 function(p) any(p %in% palabras_genericas), logical(1))) |>
  count(zona, par) |>
  bind_tf_idf(par, zona, n) |>
  group_by(zona) |>
  slice_max(tf_idf, n = 2, with_ties = FALSE) |>
  summarise(nombre = paste(par, collapse = " / "), .groups = "drop")

zonas <- tibble(zona = zona, similitud = similitud$train,
                apartamento = train$property_type == "Apartamento",
                precio = train$price,
                dist_empleo = train$dist_centro_empleo_m) |>
  group_by(zona) |>
  summarise(propiedades = n(),
            similitud_media = mean(similitud),
            pct_apartamentos = 100 * mean(apartamento),
            precio_mediano_millones = median(precio) / 1e6,
            dist_empleo_km = median(dist_empleo) / 1000,
            .groups = "drop") |>
  left_join(nombres_zona, by = "zona") |>
  arrange(desc(similitud_media))

dir.create("02_outputs/tables", recursive = TRUE, showWarnings = FALSE)
write.csv(zonas, "02_outputs/tables/zonas_similitud_chapinero.csv",
          row.names = FALSE)

# Pseudo-Chapineros: las zonas más parecidas, con un tamaño mínimo.
pseudo <- zonas |> filter(propiedades >= min_tamano_zona) |>
  slice_head(n = n_pseudo)
message("Pseudo-Chapineros elegidos:")
print(as.data.frame(pseudo), row.names = FALSE)

# 3. Elegir el esquema de CV espacial (simulación) ---------------------------
# Datos del conjunto completo (mismas variables que usan las seis familias).
source("01_code/functions/fun_simular_cv.R")
train_model <- readRDS("03_temp/train_model.rds")
conjuntos_predictores <- readRDS("03_temp/conjuntos_predictores.rds")
if (!identical(train_model$property_id, train$property_id)) {
  stop("train_model.rds y train_features.rds no tienen el mismo orden de ",
       "filas. Corra de nuevo 05_build_model_data.R.")
}
x_simulacion <- train_model[conjuntos_predictores$completo]
x_simulacion$property_type <- as.integer(x_simulacion$property_type == "Casa")
x_simulacion <- as.data.frame(lapply(x_simulacion, as.numeric))
y_simulacion <- log(train_model$price)

# Verificación: ¿los pseudo-Chapineros quedan tan cerca de train como
# Chapinero? (distancia de cada ubicación a la de train más cercana).
fun_distancia_vecino <- function(.desde, .hacia) {
  desde <- sf::st_as_sf(as.data.frame(unique(.desde)), coords = 1:2,
                        crs = crs_m)
  hacia <- sf::st_as_sf(as.data.frame(unique(.hacia)), coords = 1:2,
                        crs = crs_m)
  as.numeric(sf::st_distance(desde, hacia[sf::st_nearest_feature(desde, hacia), ],
                             by_element = TRUE))
}
xy_test <- sf::st_coordinates(sf::st_transform(
  sf::st_as_sf(test, coords = c("lon", "lat"), crs = 4326), crs_m))
distancias <- list("Chapinero (test real)" = fun_distancia_vecino(xy_test, xy))
filas_pseudo <- list()
for (z in pseudo$zona) {
  nombre <- paste0(z, ": ", pseudo$nombre[pseudo$zona == z])
  filas_zona <- which(zona == z)
  filas_pseudo[[nombre]] <- filas_zona
  set.seed(semilla)
  conservar <- sample(filas_zona, round(prop_conservar * length(filas_zona)))
  fuera <- setdiff(filas_zona, conservar)
  distancias[[nombre]] <- fun_distancia_vecino(xy[fuera, ], xy[-fuera, ])
}
message("Distancia a la propiedad de train más cercana (m):")
print(data.frame(
  zona = names(distancias),
  mediana = sapply(distancias, function(d) round(median(d))),
  p90 = sapply(distancias, function(d) round(quantile(d, 0.9))),
  pct_menos_500m = sapply(distancias, function(d) round(100 * mean(d < 500), 1))
), row.names = FALSE)

# Simulación.
detalle <- fun_evaluar_esquemas(x_simulacion, y_simulacion, xy, filas_pseudo,
                                .k = k, .prop_conservar = prop_conservar,
                                .semilla = semilla)
write.csv(detalle, "02_outputs/tables/seleccion_cv_detalle.csv",
          row.names = FALSE)

# Resumen por esquema y elección del ganador (solo esquemas espaciales).
seleccion_cv <- detalle |>
  group_by(esquema) |>
  summarise(mae_cv = mean(mae_cv),
            mae_verdadero = mean(mae_verdadero),
            brecha_media_pct = mean(brecha_pct),
            brecha_absoluta_pct = mean(abs(brecha_pct)),
            .groups = "drop") |>
  arrange(brecha_absoluta_pct)
esquema_elegido <- seleccion_cv$esquema[seleccion_cv$esquema != "Aleatoria"][1]
seleccion_cv$elegido <- seleccion_cv$esquema == esquema_elegido

write.csv(seleccion_cv, "02_outputs/tables/seleccion_cv.csv", row.names = FALSE)
writeLines(kable(seleccion_cv |>
                   transmute(Esquema = esquema,
                             `MAE CV` = round(mae_cv, 1),
                             `MAE verdadero` = round(mae_verdadero, 1),
                             `Brecha media (%)` = round(brecha_media_pct, 1),
                             `Brecha absoluta (%)` = round(brecha_absoluta_pct, 1)),
                 format = "latex", booktabs = TRUE, linesep = "",
                 align = c("l", "r", "r", "r", "r")),
           "02_outputs/tables/seleccion_cv.tex")

message("Comparación de esquemas (MAE en millones; brecha negativa = la CV ",
        "promete menos error del real):")
print(as.data.frame(seleccion_cv), row.names = FALSE, digits = 3)
message("Esquema de CV espacial elegido: ", esquema_elegido)

# 4. CV aleatoria: repartir equilibradamente por rangos de precio ---------
set.seed(semilla)
# Usar rangos de igual tamano evita problemas con precios repetidos.
grupo_precio <- cut(rank(train$price, ties.method = "random"),
                    breaks = k, labels = FALSE)
fold_aleatoria <- integer(nrow(train))

for (g in seq_len(k)) {
  filas <- which(grupo_precio == g)
  fold_aleatoria[filas] <- sample(rep(seq_len(k), length.out = length(filas)))
}

# 5. CV espacial: el esquema elegido y el de 5 regiones (comparación) -----
fold_espacial <- fun_esquemas_cv(xy, k, semilla)[[esquema_elegido]]
set.seed(semilla)
fold_espacial_k5 <- stats::kmeans(xy, centers = k, nstart = 30)$cluster

# 6. Guardar indices de validacion para reutilizarlos ---------------------
cv_folds <- list(
  aleatoria   = lapply(seq_len(k), function(i) which(fold_aleatoria == i)),
  espacial    = lapply(seq_len(k), function(i) which(fold_espacial == i)),
  espacial_k5 = lapply(seq_len(k), function(i) which(fold_espacial_k5 == i)),
  esquema_espacial = esquema_elegido
)

saveRDS(cv_folds, "03_temp/cv_folds.rds")

# Diagnostico: numero de propiedades en cada fold -------------------------
print(data.frame(
  fold = seq_len(k),
  n_aleatoria = lengths(cv_folds$aleatoria),
  n_espacial  = lengths(cv_folds$espacial),
  n_espacial_k5 = lengths(cv_folds$espacial_k5)
), row.names = FALSE)

message("Folds guardados en 03_temp/cv_folds.rds (espacial = ",
        esquema_elegido, ").")

# Limpiamos el entorno: borramos solo los objetos que creó este script.
rm(list = setdiff(ls(), objetos_previos))
