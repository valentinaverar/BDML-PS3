##########################################################
# 05_create_cv_folds.R
#
# Objetivo: crear las particiones de validación cruzada que comparten
# todos los modelos (common ground): una aleatoria y una espacial.
#
# Etapa del Problem Set: Etapa 2 (elegir hiperparámetros con CV espacial)
# y Etapa 3 (comparar CV espacial frente a CV aleatoria).
# Insumos:  03_temp/train_features.rds.
# Salidas:  03_temp/cv_folds.rds: lista con los índices de VALIDACIÓN de
#           cada fold (aleatoria y espacial), en el mismo orden de filas de
#           train_features.rds. Sirve tal cual para lightgbm::lgb.cv() y,
#           con su complemento, para trainControl(index = ...) de caret.
#
# Decisiones:
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
pacman::p_load(sf)

train <- readRDS("03_temp/train_features.rds")
set.seed(semilla)

# 1. CV aleatoria: repartir equilibradamente por rangos de precio ---------
# Usar rangos de igual tamano evita problemas con precios repetidos.
grupo_precio <- cut(rank(train$price, ties.method = "random"),
                    breaks = k, labels = FALSE)
fold_aleatoria <- integer(nrow(train))

for (g in seq_len(k)) {
  filas <- which(grupo_precio == g)
  fold_aleatoria[filas] <- sample(rep(seq_len(k), length.out = length(filas)))
}

# 2. CV espacial: cada zona geografica completa se deja fuera ------------
# k-means genera cinco regiones compactas a partir de latitud y longitud.
# Dos propiedades con las mismas coordenadas quedan en el mismo fold.
puntos <- sf::st_as_sf(train, coords = c("lon", "lat"), crs = 4326)
xy <- sf::st_coordinates(sf::st_transform(puntos, crs_m))
fold_espacial <- stats::kmeans(xy, centers = k, nstart = 30)$cluster

# 3. Guardar indices de validacion para reutilizarlos ---------------------
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
