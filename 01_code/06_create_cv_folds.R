##########################################################
# 06_create_cv_folds.R
#
# Objetivo: crear las particiones de validación cruzada que comparten
# todos los modelos (common ground): una aleatoria y una espacial.
#
# Etapa del Problem Set: Etapa 2 (elegir hiperparámetros con CV espacial)
# y Etapa 3 (comparar CV espacial frente a CV aleatoria).
# Insumos:  03_temp/train_features.rds y 03_temp/test_features.rds.
# Salidas:  03_temp/cv_folds.rds: lista con los índices de VALIDACIÓN de
#           cada fold (aleatoria y espacial), en el mismo orden de filas de
#           train_features.rds. Sirve tal cual para lightgbm::lgb.cv() y,
#           con su complemento, para trainControl(index = ...) de caret.
#           03_temp/similitud_chapinero.rds: similitud de cada propiedad de
#           train con las de Chapinero (propensity score).
#
# Decisiones:
#   - Similitud con Chapinero: regresión logística que distingue test
#     (Chapinero) de train con las características de la vivienda y de su
#     entorno, sin coordenadas ni nombres de barrios ("adversarial
#     validation"). Se calcula fuera de muestra y no usa el precio.
#   - CV aleatoria: 5 folds equilibrados por rangos de precio, para que
#     cada fold tenga precios bajos, medios y altos.
#   - CV espacial: 5 regiones compactas creadas con k-means sobre las
#     coordenadas (en metros). Cada fold deja por fuera una región entera,
#     como le pasa al modelo con Chapinero. Es la misma idea del
#     "clustering espacial" del paquete spatialsample (Lecture 14).
#   - Las propiedades con las mismas coordenadas (el mismo edificio o el
#     mismo anuncio repetido) quedan siempre en el mismo fold espacial, lo
#     que evita que el modelo "vea" la misma propiedad al validar.
##########################################################

# Objetos que ya existían en el entorno antes de correr este script. Al
# final se borra solo lo que este script creó, sin tocar lo demás.
objetos_previos <- ls()

# Parametros ---------------------------------------------------------------
k <- 5
semilla <- 2026
crs_m <- 3116  # Coordenadas proyectadas en metros, como en el script 03.

# Datos --------------------------------------------------------------------
if (!require(pacman)) install.packages("pacman")
pacman::p_load(sf, dplyr)

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

# 2. CV aleatoria: repartir equilibradamente por rangos de precio ---------
set.seed(semilla)
# Usar rangos de igual tamano evita problemas con precios repetidos.
grupo_precio <- cut(rank(train$price, ties.method = "random"),
                    breaks = k, labels = FALSE)
fold_aleatoria <- integer(nrow(train))

for (g in seq_len(k)) {
  filas <- which(grupo_precio == g)
  fold_aleatoria[filas] <- sample(rep(seq_len(k), length.out = length(filas)))
}

# 3. CV espacial: cada zona geografica completa se deja fuera ------------
# k-means genera cinco regiones compactas a partir de latitud y longitud.
# Dos propiedades con las mismas coordenadas quedan en el mismo fold.
puntos <- sf::st_as_sf(train, coords = c("lon", "lat"), crs = 4326)
xy <- sf::st_coordinates(sf::st_transform(puntos, crs_m))
fold_espacial <- stats::kmeans(xy, centers = k, nstart = 30)$cluster

# 4. Guardar indices de validacion para reutilizarlos ---------------------
cv_folds <- list(
  aleatoria = lapply(seq_len(k), function(i) which(fold_aleatoria == i)),
  espacial  = lapply(seq_len(k), function(i) which(fold_espacial == i))
)

saveRDS(cv_folds, "03_temp/cv_folds.rds")

# Diagnostico: numero de propiedades en cada fold -------------------------
print(data.frame(
  fold = seq_len(k),
  n_aleatoria = lengths(cv_folds$aleatoria),
  n_espacial  = lengths(cv_folds$espacial)
), row.names = FALSE)

message("Folds guardados en 03_temp/cv_folds.rds")

# Limpiamos el entorno: borramos solo los objetos que creó este script.
rm(list = setdiff(ls(), objetos_previos))
