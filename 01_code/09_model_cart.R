##########################################################
# 09_model_cart.R
#
# Objetivo: estimar un árbol CART con el conjunto completo, elegir cp y
# minbucket con la CV espacial, medir el error con CV espacial y aleatoria,
# y generar dos envíos: el precio total y el híbrido de precio por m².
#
# Etapa del Problem Set: Etapa 2 (familia CARTs) y Etapa 3.
# Insumos:  03_temp/train_model.rds, test_model.rds,
#           conjuntos_predictores.rds (script 05) y cv_folds.rds (script 06,
#           k-means de 20 regiones).
# Salidas:  02_outputs/predictions/CART_completo_....csv y
#           CART_completo_m2_....csv (envíos).
#           02_outputs/tables/CART_completo_busqueda.csv: MAE espacial de
#           cada cp y minbucket.
#           02_outputs/tables/CART_completo_importancia.csv.
#           02_outputs/figures/CART_completo_arbol_4niveles.png: el mismo
#           árbol mostrado solo hasta 4 niveles, para las diapositivas.
#           Filas en 02_outputs/tables/registro_modelos.csv.
#           03_temp/oof_cart.rds (predicciones fuera de muestra).
#
# Decisiones:
#   - rpart con method = "anova" sobre el precio en pesos, que es la
#     métrica de Kaggle (error absoluto). No se modela el logaritmo.
#   - Grilla en la CV espacial: cp en {0,0001; 0,0005; 0,001; 0,005; 0,01}
#     y minbucket en {10; 30; 100}. maxdepth = 30, el límite de rpart.
#     xval = 0 porque la validación cruzada es la de nuestros folds, no la
#     interna de rpart.
#   - Los hiperparámetros se eligen minimizando el MAE en pesos. Con los
#     elegidos se miden las dos CV y se registran con fun_registrar_cv().
#   - Segunda variante, como en el script 13: donde la superficie es real
#     se predice el precio por m² y se multiplica por esa superficie; donde
#     fue imputada se usa el árbol de precio total. Usa el mismo cp y el
#     mismo minbucket. Los árboles no extrapolan, y esta variante es la que
#     en LightGBM bajó el error en Chapinero.
#   - Las predicciones se acotan al rango de train (300 a 1.650 millones).
#   - Si CART_completo_busqueda.csv ya tiene las 15 combinaciones y
#     buscar_hiperparametros es FALSE, no se repite la grilla.
#   - La figura de 4 niveles es solo para explicar. El envío usa maxdepth 30.
##########################################################

# Objetos que ya existían en el entorno antes de correr este script. Al
# final se borra solo lo que este script creó, sin tocar lo demás.
objetos_previos <- ls()

# Parámetros ---------------------------------------------------------------
buscar_hiperparametros <- FALSE
cps <- c(0.0001, 0.0005, 0.001, 0.005, 0.01)
minbuckets <- c(10L, 30L, 100L)
maxdepth <- 30L

# Paquetes y funciones -----------------------------------------------------
if (!require(pacman)) install.packages("pacman")
pacman::p_load(dplyr, rpart, rpart.plot)

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
message("CART: conjunto completo, ", ncol(x_train), " predictores.")

fun_superficie <- function(.datos) {
  #' Superficie real de cada vivienda, o NA si fue imputada
  #'
  #' @description La misma regla del script 13: cubierta o, si falta, total,
  #' incluidas las tomadas de la descripción. NA si las dos fueron imputadas.
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
fun_num <- function(.x) {
  gsub(".", "_", formatC(.x, format = "f", digits = 4, drop0trailing = TRUE),
       fixed = TRUE)
}

fun_cart <- function(.filas, .y, .x_nuevo, .cp, .minbucket, .profundidad = maxdepth) {
  #' Ajustar un árbol y predecir la variable que se le pasó
  #'
  #' @param .y \code{numeric}. Precio en pesos o precio por m².
  #' @param .profundidad \code{integer}. 30 en el envío; 4 solo en la figura.
  #'
  #' @returns \code{list} con el modelo y las predicciones, sin acotar.
  ajuste <- .filas
  datos <- x_train[ajuste, , drop = FALSE]
  datos$objetivo <- .y[ajuste]
  modelo <- rpart::rpart(
    objetivo ~ ., data = datos, method = "anova",
    control = rpart::rpart.control(
      cp = .cp, minbucket = .minbucket, maxdepth = .profundidad, xval = 0
    )
  )
  pred <- as.numeric(predict(modelo, newdata = .x_nuevo))
  list(modelo = modelo, pred = pred)
}

# 1. Grilla con la CV espacial (precio total) ------------------------------
ruta_busqueda <- "02_outputs/tables/CART_completo_busqueda.csv"
dir.create("02_outputs/tables", recursive = TRUE, showWarnings = FALSE)
busqueda <- if (file.exists(ruta_busqueda)) {
  read.csv(ruta_busqueda)
} else {
  data.frame()
}
grilla <- expand.grid(cp = cps, minbucket = minbuckets)

if (buscar_hiperparametros || nrow(busqueda) < nrow(grilla)) {
  if (buscar_hiperparametros) busqueda <- data.frame()
  for (i in seq_len(nrow(grilla))) {
    cp <- grilla$cp[i]
    minbucket <- grilla$minbucket[i]
    if (nrow(busqueda) > 0 &&
        any(busqueda$cp == cp & busqueda$minbucket == minbucket)) next
    inicio <- proc.time()[["elapsed"]]
    oof <- rep(NA_real_, n)
    for (val in cv_folds$espacial) {
      oof[val] <- fun_acotar(fun_cart(
        setdiff(seq_len(n), val), precio, x_train[val, , drop = FALSE],
        cp, minbucket
      )$pred)
    }
    fila <- data.frame(cp = cp, minbucket = minbucket,
                       mae_espacial = mean(abs(oof - precio)))
    busqueda <- bind_rows(busqueda, fila)
    write.csv(busqueda, ruta_busqueda, row.names = FALSE)
    message("cp ", cp, ", minbucket ", minbucket, ": MAE ",
            round(fila$mae_espacial / 1e6, 1), " millones (",
            round(proc.time()[["elapsed"]] - inicio), " s).")
  }
}

mejor <- busqueda[which.min(busqueda$mae_espacial), ]
cp <- mejor$cp
minbucket <- as.integer(mejor$minbucket)
message("CART elegido: cp ", cp, ", minbucket ", minbucket, ".")

nombre <- paste0("CART_completo_cp_", fun_num(cp), "_minbucket_", minbucket)
nombre_m2 <- paste0("CART_completo_m2_cp_", fun_num(cp), "_minbucket_", minbucket)
descripcion <- paste0("rpart anova; cp=", cp, "; minbucket=", minbucket,
                      "; maxdepth=30")
descripcion_m2 <- paste0(
  "Híbrido: precio por m2 x superficie (superficie conocida) y precio total ",
  "(superficie imputada); ", descripcion
)

# 2. Error fuera de muestra con cada tipo de CV ----------------------------
fun_oof <- function(.folds) {
  #' Precio total y predicción híbrida, fuera de muestra
  total <- rep(NA_real_, n)
  hibrida <- rep(NA_real_, n)
  for (val in .folds) {
    total[val] <- fun_acotar(fun_cart(
      setdiff(seq_len(n), val), precio, x_train[val, , drop = FALSE],
      cp, minbucket
    )$pred)
    ajuste_m2 <- setdiff(filas_m2, val)
    val_m2 <- intersect(val, filas_m2)
    hibrida[val] <- total[val]
    if (length(ajuste_m2) > minbucket && length(val_m2) > 0) {
      tasa <- fun_cart(
        ajuste_m2, precio / superficie_train,
        x_train[val_m2, , drop = FALSE], cp, minbucket
      )$pred
      hibrida[val_m2] <- fun_acotar(tasa * superficie_train[val_m2])
    }
  }
  list(total = total, hibrida = hibrida)
}

oof_cart <- data.frame(property_id = train$property_id)
for (tipo in c("espacial", "aleatoria")) {
  message("CV ", tipo, "...")
  oof <- fun_oof(cv_folds[[tipo]])
  fun_registrar_cv(
    .archivo = nombre, .algoritmo = "CART", .conjunto = "completo",
    .n_predictores = ncol(x_train), .hiperparametros = descripcion,
    .tipo_cv = tipo, .obs = precio, .pred = oof$total
  )
  fun_registrar_cv(
    .archivo = nombre_m2, .algoritmo = "CART precio m2", .conjunto = "completo",
    .n_predictores = ncol(x_train), .hiperparametros = descripcion_m2,
    .tipo_cv = tipo, .obs = precio, .pred = oof$hibrida
  )
  oof_cart[[paste0("CART_completo_", tipo)]] <- oof$total
  oof_cart[[paste0("CART_completo_m2_", tipo)]] <- oof$hibrida
  message("  precio total: MAE ", round(mean(abs(oof$total - precio)) / 1e6, 1),
          " millones; híbrido: ",
          round(mean(abs(oof$hibrida - precio)) / 1e6, 1), " millones.")
}
saveRDS(oof_cart, "03_temp/oof_cart.rds")

# 3. Modelo final, importancia, figura y envíos ---------------------------
final <- fun_cart(seq_len(n), precio, x_test, cp, minbucket)
fun_save_submission(.ids = test$property_id, .pred = fun_acotar(final$pred),
                    .nombre = nombre)

tasa_test <- fun_cart(filas_m2, precio / superficie_train, x_test, cp, minbucket)$pred
pred_m2 <- fun_acotar(tasa_test * superficie_test)
pred_hibrida <- fun_acotar(final$pred)
conocida_test <- !is.na(superficie_test) & superficie_test > 0
pred_hibrida[conocida_test] <- pred_m2[conocida_test]
fun_save_submission(.ids = test$property_id, .pred = pred_hibrida,
                    .nombre = nombre_m2)

importancia <- final$modelo$variable.importance
importancia <- data.frame(
  variable = names(importancia),
  importancia = as.numeric(importancia)
)
importancia <- importancia[order(-importancia$importancia), ]
write.csv(importancia, "02_outputs/tables/CART_completo_importancia.csv",
          row.names = FALSE)

# Árbol corto, en millones, solo para poder leerlo en las diapositivas.
datos_corto <- x_train
datos_corto$objetivo <- precio / 1e6
arbol_corto <- rpart::rpart(
  objetivo ~ ., data = datos_corto, method = "anova",
  control = rpart::rpart.control(cp = cp, minbucket = minbucket,
                                 maxdepth = 4, xval = 0)
)
dir.create("02_outputs/figures", recursive = TRUE, showWarnings = FALSE)
grDevices::png("02_outputs/figures/CART_completo_arbol_4niveles.png",
               width = 1400, height = 900, res = 140)
rpart.plot::rpart.plot(
  arbol_corto, type = 2, extra = 101, digits = 0, fallen.leaves = TRUE,
  main = "CART a 4 niveles. El numero del nodo es el precio medio, en millones."
)
grDevices::dev.off()

# Resumen ------------------------------------------------------------------
read.csv("02_outputs/tables/registro_modelos.csv") |>
  filter(grepl("^CART", algoritmo)) |>
  transmute(archivo = substr(archivo, 1, 42), tipo_cv,
            mae_millones = round(mae / 1e6, 1),
            mae_similares_chapinero = round(mae_similares_chapinero / 1e6, 1),
            sesgo_similares_chapinero = round(sesgo_similares_chapinero / 1e6, 1)) |>
  print(row.names = FALSE)

rm(list = setdiff(ls(), objetos_previos))
