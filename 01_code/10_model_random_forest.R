##########################################################
# 10_model_random_forest.R
#
# Objetivo: estimar un Random Forest con el conjunto completo, elegir mtry
# y el tamaño mínimo de nodo con la CV espacial, medir el error con CV
# espacial y aleatoria, y generar dos envíos: el precio total y el híbrido
# de precio por m².
#
# Etapa del Problem Set: Etapa 2 (familia Random Forest) y Etapa 3.
# Insumos:  03_temp/train_model.rds, test_model.rds,
#           conjuntos_predictores.rds (script 05) y cv_folds.rds (script 06,
#           k-means de 20 regiones).
# Salidas:  02_outputs/predictions/RF_completo_....csv y
#           RF_completo_m2_....csv (envíos).
#           02_outputs/tables/RF_completo_busqueda.csv: MAE espacial de cada
#           mtry y min.node.size.
#           02_outputs/tables/RF_completo_importancia.csv.
#           Filas en 02_outputs/tables/registro_modelos.csv.
#           03_temp/oof_rf.rds (predicciones fuera de muestra).
#
# Decisiones:
#   - ranger, que es el Random Forest del cuaderno de validación espacial
#     (Lecture 14). num.threads = 6 y seed = 2026.
#   - Se predice el precio en pesos. La pérdida de ranger por defecto es el
#     error cuadrático; el hiperparámetro se elige igual con el MAE en
#     pesos, que es la métrica de Kaggle.
#   - Grilla en la CV espacial: mtry en {15; 30; 60} (con 174 predictores,
#     sqrt(p) es cerca de 13 y p/3 cerca de 58) y min.node.size en {5; 20}.
#   - La búsqueda usa 300 árboles, para que las 30 combinaciones (6 x 5
#     folds) terminen en un tiempo razonable. El modelo que se registra y
#     se envía usa 500 árboles, con el mtry y el min.node.size ya elegidos.
#   - Segunda variante, como en el script 13: donde la superficie es real
#     se predice el precio por m² y se multiplica por esa superficie; donde
#     fue imputada se usa el bosque de precio total. Mismos hiperparámetros.
#   - El modelo final guarda la importancia por impureza
#     (importance = "impurity").
#   - Las predicciones se acotan al rango de train (300 a 1.650 millones).
#   - Si RF_completo_busqueda.csv ya tiene las 6 combinaciones y
#     buscar_hiperparametros es FALSE, no se repite la grilla.
##########################################################

# Objetos que ya existían en el entorno antes de correr este script. Al
# final se borra solo lo que este script creó, sin tocar lo demás.
objetos_previos <- ls()

# Parámetros ---------------------------------------------------------------
buscar_hiperparametros <- FALSE
semilla <- 2026
hilos <- 6L
mtrys <- c(15L, 30L, 60L)
min_nodos <- c(5L, 20L)
arboles_busqueda <- 300L
arboles_final <- 500L

# Paquetes y funciones -----------------------------------------------------
if (!require(pacman)) install.packages("pacman")
pacman::p_load(dplyr, ranger)

source("01_code/functions/fun_save_submission.R")
source("01_code/functions/fun_metricas.R")

# Datos --------------------------------------------------------------------
train <- readRDS("03_temp/train_model.rds")
test <- readRDS("03_temp/test_model.rds")
conjuntos_predictores <- readRDS("03_temp/conjuntos_predictores.rds")
cv_folds <- readRDS("03_temp/cv_folds.rds")

if (!all(sort(unlist(cv_folds$espacial)) == seq_len(nrow(train)))) {
  stop("Los folds no coinciden con train_model.rds. Corra de nuevo los ",
       "scripts 05 y 06.")
}
if (!identical(cv_folds$esquema_espacial, "k-means 20 regiones")) {
  stop("cv_folds$espacial es '", cv_folds$esquema_espacial,
       "'. Con los datos fijos debe ser k-means de 20 regiones.")
}

precio <- train$price
rango_precio <- range(precio)
n <- nrow(train)

fun_matriz <- function(.datos, .vars) {
  x <- .datos[.vars]
  x$property_type <- as.integer(x$property_type == "Casa")
  as.data.frame(lapply(x, as.numeric))
}
x_train <- fun_matriz(train, conjuntos_predictores$completo)
x_test <- fun_matriz(test, conjuntos_predictores$completo)
message("Random Forest: conjunto completo, ", ncol(x_train), " predictores.")

fun_superficie <- function(.datos) {
  #' Superficie real de cada vivienda, o NA si fue imputada
  #'
  #' @description La misma regla del script 13.
  fun_conocida <- function(.v) {
    indicador <- paste0("falta_", .v)
    if (indicador %in% names(.datos)) .datos[[indicador]] == 0 else
      rep(TRUE, nrow(.datos))
  }
  dplyr::case_when(
    fun_conocida("surface_covered_completa") ~ .datos$surface_covered_completa,
    fun_conocida("surface_total_completa") ~ .datos$surface_total_completa,
    TRUE ~ NA_real_
  )
}
superficie_train <- fun_superficie(train)
superficie_test <- fun_superficie(test)
filas_m2 <- which(!is.na(superficie_train) & superficie_train > 0)

fun_acotar <- function(.pred) pmin(pmax(.pred, rango_precio[1]), rango_precio[2])

fun_rf <- function(.filas, .y, .x_nuevo, .mtry, .min_nodo, .arboles,
                   .importancia = "none") {
  #' Ajustar ranger y predecir la variable que se le pasó
  #'
  #' @param .y \code{numeric}. Precio en pesos o precio por m².
  #' @param .importancia \code{character}. "impurity" solo en el modelo final.
  #'
  #' @returns \code{list} con el modelo y las predicciones, sin acotar.
  modelo <- ranger::ranger(
    x = x_train[.filas, , drop = FALSE], y = .y[.filas],
    num.trees = .arboles, mtry = .mtry, min.node.size = .min_nodo,
    num.threads = hilos, seed = semilla, importance = .importancia
  )
  pred <- as.numeric(predict(modelo, data = .x_nuevo)$predictions)
  list(modelo = modelo, pred = pred)
}

# 1. Grilla con la CV espacial (precio total, 300 árboles) -----------------
ruta_busqueda <- "02_outputs/tables/RF_completo_busqueda.csv"
dir.create("02_outputs/tables", recursive = TRUE, showWarnings = FALSE)
busqueda <- if (file.exists(ruta_busqueda)) {
  read.csv(ruta_busqueda)
} else {
  data.frame()
}
grilla <- expand.grid(mtry = mtrys, min_node_size = min_nodos)

if (buscar_hiperparametros || nrow(busqueda) < nrow(grilla)) {
  if (buscar_hiperparametros) busqueda <- data.frame()
  for (i in seq_len(nrow(grilla))) {
    mtry <- grilla$mtry[i]
    min_nodo <- grilla$min_node_size[i]
    if (nrow(busqueda) > 0 &&
        any(busqueda$mtry == mtry & busqueda$min_node_size == min_nodo)) next
    inicio <- proc.time()[["elapsed"]]
    oof <- rep(NA_real_, n)
    for (k in seq_along(cv_folds$espacial)) {
      val <- cv_folds$espacial[[k]]
      oof[val] <- fun_acotar(fun_rf(
        setdiff(seq_len(n), val), precio, x_train[val, , drop = FALSE],
        mtry, min_nodo, arboles_busqueda
      )$pred)
      message("  mtry ", mtry, ", min nodo ", min_nodo, ": fold ", k, " listo.")
    }
    fila <- data.frame(
      mtry = mtry, min_node_size = min_nodo,
      arboles_busqueda = arboles_busqueda,
      mae_espacial = mean(abs(oof - precio))
    )
    busqueda <- bind_rows(busqueda, fila)
    write.csv(busqueda, ruta_busqueda, row.names = FALSE)
    message("mtry ", mtry, ", min nodo ", min_nodo, ": MAE ",
            round(fila$mae_espacial / 1e6, 1), " millones (",
            round(proc.time()[["elapsed"]] - inicio), " s).")
  }
}

mejor <- busqueda[which.min(busqueda$mae_espacial), ]
mtry <- as.integer(mejor$mtry)
min_nodo <- as.integer(mejor$min_node_size)
message("Random Forest elegido: mtry ", mtry, ", min nodo ", min_nodo,
        ", ", arboles_final, " árboles en el envío.")

nombre <- paste0("RF_completo_mtry_", mtry, "_minnodo_", min_nodo,
                 "_arboles_", arboles_final)
nombre_m2 <- paste0("RF_completo_m2_mtry_", mtry, "_minnodo_", min_nodo,
                    "_arboles_", arboles_final)
descripcion <- paste0(
  "ranger; mtry=", mtry, "; min.node.size=", min_nodo,
  "; arboles=", arboles_final,
  "; busqueda con ", arboles_busqueda, " arboles"
)
descripcion_m2 <- paste0(
  "Híbrido: precio por m2 x superficie (superficie conocida) y precio total ",
  "(superficie imputada); ", descripcion
)

# 2. Error fuera de muestra con cada tipo de CV (500 árboles) --------------
fun_oof <- function(.folds) {
  #' Precio total y predicción híbrida, fuera de muestra
  total <- rep(NA_real_, n)
  hibrida <- rep(NA_real_, n)
  for (k in seq_along(.folds)) {
    val <- .folds[[k]]
    message("  fold ", k, "/", length(.folds))
    total[val] <- fun_acotar(fun_rf(
      setdiff(seq_len(n), val), precio, x_train[val, , drop = FALSE],
      mtry, min_nodo, arboles_final
    )$pred)
    ajuste_m2 <- setdiff(filas_m2, val)
    val_m2 <- intersect(val, filas_m2)
    hibrida[val] <- total[val]
    if (length(ajuste_m2) > min_nodo && length(val_m2) > 0) {
      tasa <- fun_rf(
        ajuste_m2, precio / superficie_train,
        x_train[val_m2, , drop = FALSE], mtry, min_nodo, arboles_final
      )$pred
      hibrida[val_m2] <- fun_acotar(tasa * superficie_train[val_m2])
    }
  }
  list(total = total, hibrida = hibrida)
}

oof_rf <- data.frame(property_id = train$property_id)
for (tipo in c("espacial", "aleatoria")) {
  message("CV ", tipo, "...")
  oof <- fun_oof(cv_folds[[tipo]])
  fun_registrar_cv(
    .archivo = nombre, .algoritmo = "Random Forest", .conjunto = "completo",
    .n_predictores = ncol(x_train), .hiperparametros = descripcion,
    .tipo_cv = tipo, .obs = precio, .pred = oof$total
  )
  fun_registrar_cv(
    .archivo = nombre_m2, .algoritmo = "Random Forest precio m2",
    .conjunto = "completo", .n_predictores = ncol(x_train),
    .hiperparametros = descripcion_m2, .tipo_cv = tipo,
    .obs = precio, .pred = oof$hibrida
  )
  oof_rf[[paste0("RF_completo_", tipo)]] <- oof$total
  oof_rf[[paste0("RF_completo_m2_", tipo)]] <- oof$hibrida
  message("  precio total: MAE ", round(mean(abs(oof$total - precio)) / 1e6, 1),
          " millones; híbrido: ",
          round(mean(abs(oof$hibrida - precio)) / 1e6, 1), " millones.")
}
saveRDS(oof_rf, "03_temp/oof_rf.rds")

# 3. Modelo final, importancia y envíos -----------------------------------
final <- fun_rf(seq_len(n), precio, x_test, mtry, min_nodo, arboles_final,
                .importancia = "impurity")
fun_save_submission(.ids = test$property_id, .pred = fun_acotar(final$pred),
                    .nombre = nombre)

tasa_test <- fun_rf(
  filas_m2, precio / superficie_train, x_test, mtry, min_nodo, arboles_final
)$pred
pred_hibrida <- fun_acotar(final$pred)
conocida_test <- !is.na(superficie_test) & superficie_test > 0
pred_hibrida[conocida_test] <- fun_acotar(tasa_test[conocida_test] *
                                            superficie_test[conocida_test])
fun_save_submission(.ids = test$property_id, .pred = pred_hibrida,
                    .nombre = nombre_m2)

importancia <- final$modelo$variable.importance
importancia <- data.frame(
  variable = names(importancia),
  importancia = as.numeric(importancia)
)
importancia <- importancia[order(-importancia$importancia), ]
write.csv(importancia, "02_outputs/tables/RF_completo_importancia.csv",
          row.names = FALSE)

# Resumen ------------------------------------------------------------------
read.csv("02_outputs/tables/registro_modelos.csv") |>
  filter(grepl("^Random Forest", algoritmo)) |>
  transmute(archivo = substr(archivo, 1, 42), tipo_cv,
            mae_millones = round(mae / 1e6, 1),
            mae_similares_chapinero = round(mae_similares_chapinero / 1e6, 1),
            sesgo_similares_chapinero = round(sesgo_similares_chapinero / 1e6, 1)) |>
  print(row.names = FALSE)

rm(list = setdiff(ls(), objetos_previos))
