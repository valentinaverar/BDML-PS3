##########################################################
# 17_model_blend_vecinos_red.R
#
# Objetivo: mezclar el LightGBM con vecinos (script 15) con la red
# neuronal (script 12) y generar el envío a Kaggle. Combina las dos mejoras
# que funcionaron en Chapinero: el precio de los vecinos y la mezcla con la
# red.
#
# Etapa del Problem Set: Etapa 2 (combinación de familias) y Etapa 4.
# Insumos:  03_temp/train_model.rds, test_model.rds (script 05);
#           03_temp/oof_lgbm_vecinos.rds y el envío
#           LGBM_completo_m2_vecinos_k..._lr_....csv (script 15);
#           03_temp/oof_nn.rds y el envío NN_completo_....csv (script 12);
#           02_outputs/tables/BLEND_m2_red_pesos.csv (script 16).
# Salidas:  02_outputs/predictions/BLEND_vecinos_red_w_....csv (envío).
#           Filas en 02_outputs/tables/registro_modelos.csv.
#           03_temp/oof_blend_vecinos_red.rds (predicciones fuera de muestra).
#
# Decisiones:
#   - Los vecinos le dicen al LightGBM cuánto vale el m² en la zona; la red
#     corrige la subestimación de los árboles. Son problemas distintos, así
#     que las dos mejoras pueden sumarse.
#   - Predicción = (1 - w) x LightGBM con vecinos + w x red, con el MISMO
#     peso w que eligió el script 16 para la mezcla sin vecinos (se lee de
#     su tabla de pesos). No se vuelve a elegir con la CV espacial porque
#     esa CV subestima lo que aportan los vecinos: al apartar una región
#     entera quedan a unos 660 m, frente a unos 220 m en Chapinero. En la
#     CV el 15 sale peor que el 13, pero en Kaggle sale mejor (166,7
#     frente a 169,2).
#   - No se reentrena nada: se usan las predicciones fuera de muestra y los
#     envíos de test de los dos scripts.
##########################################################

# Objetos que ya existían en el entorno antes de correr este script. Al
# final se borra solo lo que este script creó, sin tocar lo demás.
objetos_previos <- ls()

# Paquetes y funciones -----------------------------------------------------
if (!require(pacman)) install.packages("pacman")
pacman::p_load(dplyr)

source("01_code/functions/fun_save_submission.R")
source("01_code/functions/fun_metricas.R")

# Datos --------------------------------------------------------------------
train <- readRDS("03_temp/train_model.rds")
test <- readRDS("03_temp/test_model.rds")
oof_vecinos <- readRDS("03_temp/oof_lgbm_vecinos.rds")
oof_red <- readRDS("03_temp/oof_nn.rds")

for (oof in list(oof_vecinos, oof_red)) {
  if (!identical(oof$property_id, train$property_id)) {
    stop("Las predicciones fuera de muestra no están en el orden de ",
         "train_model.rds. Corra de nuevo los scripts 15 y 12.")
  }
}
if (!file.exists("02_outputs/tables/BLEND_m2_red_pesos.csv")) {
  stop("Falta la tabla de pesos de la mezcla. Corra primero el script 16.")
}
y <- train$price
rango_precio <- range(y)

fun_leer_envio <- function(.patron) {
  #' Leer un envío de Kaggle y ordenarlo como test_model.rds
  archivo <- list.files("02_outputs/predictions", pattern = .patron,
                        full.names = TRUE)
  if (length(archivo) != 1) {
    stop("Se esperaba un único envío con el patrón ", .patron, " y hay ",
         length(archivo), ".")
  }
  envio <- utils::read.csv(archivo)
  precio <- envio$price[match(test$property_id, envio$property_id)]
  if (anyNA(precio)) stop("El envío ", archivo, " no cubre todo el test.")
  message("Envío leído: ", basename(archivo))
  precio
}
test_vecinos <- fun_leer_envio("^LGBM_completo_m2_vecinos_k[0-9]+_lr_.*\\.csv$")
test_red <- fun_leer_envio("^NN_completo_.*\\.csv$")

fun_mezclar <- function(.lgbm, .red, .w) {
  #' Promedio ponderado, acotado al rango de precios de train
  pmin(pmax((1 - .w) * .lgbm + .w * .red, rango_precio[1]), rango_precio[2])
}

# Peso: el que eligió el 16 para la mezcla sin vecinos -------------------
pesos_11e <- utils::read.csv("02_outputs/tables/BLEND_m2_red_pesos.csv")
w <- pesos_11e$peso_red[which.min(pesos_11e$mae_similares)]
message("Peso de la red (elegido en el 16): ", w, ".")
if (w == 0) stop("En el 16 la red no aportaba (peso 0). No se mezcla.")

# Registro, predicciones fuera de muestra y envío -------------------------
nombre <- paste0("BLEND_vecinos_red_w_", fun_hiperparam_str(w, 2))
descripcion <- paste0("Mezcla (1-w) x LightGBM precio m2 con vecinos (15) ",
                      "+ w x red neuronal (12); w=", w,
                      " (el mismo elegido en el 16)")
oof_blend <- data.frame(property_id = train$property_id)
for (tipo in c("espacial", "aleatoria")) {
  pred <- fun_mezclar(oof_vecinos[[paste0("LGBM_vecinos_", tipo)]],
                      oof_red[[paste0("NN_completo_", tipo)]], w)
  fun_registrar_cv(
    .archivo = nombre, .algoritmo = "Mezcla LGBM vecinos-red",
    .conjunto = "completo + vecinos", .n_predictores = NA_integer_,
    .hiperparametros = descripcion, .tipo_cv = tipo, .obs = y, .pred = pred
  )
  oof_blend[[paste0("BLEND_vecinos_red_", tipo)]] <- pred
  message("CV ", tipo, ": MAE = ", round(mean(abs(pred - y)) / 1e6, 1),
          " millones.")
}
saveRDS(oof_blend, "03_temp/oof_blend_vecinos_red.rds")
fun_save_submission(.ids = test$property_id,
                    .pred = fun_mezclar(test_vecinos, test_red, w),
                    .nombre = nombre)

# Limpiamos el entorno: borramos solo los objetos que creó este script.
rm(list = setdiff(ls(), objetos_previos))
