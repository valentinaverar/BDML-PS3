##########################################################
# 06_create_cv_folds.R
#
# Objetivo: crear folds de validacion cruzada compartidos por
# todos los modelos: aleatorios (por rangos de precio) y
# espaciales (zonas geograficas completas).
#
# Entrada: 03_temp/train_features.rds
# Salida:  03_temp/cv_folds.rds
#
# Cada fold contiene los INDICES DE VALIDACION, en el mismo
# orden de filas que train_features.rds. Este formato se puede
# pasar directamente al argumento 'folds' de lightgbm::lgb.cv().
##########################################################

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

rm(k, semilla, crs_m, train, grupo_precio, fold_aleatoria, g, filas, puntos, xy, fold_espacial, cv_folds)