##########################################################
# Script maestro - Problem Set 3: Making Money with ML?
#
# Correr este archivo reproduce todos los resultados del repositorio.
#
# Para reproducir todo, desde la carpeta raíz del repositorio:
# - En una sesión interactiva de R: source("01_code/00_rundirectory.R")
# - Desde la terminal:              R CMD BATCH 01_code/00_rundirectory.R
#
# Autores (Equipo 03):
# - Julián Herrera
# - Andrés Silva
# - Valentina Vera
#
# Big Data y Machine Learning para Economía Aplicada (MECA 4107)
# Universidad de los Andes - 2026-20
##########################################################
# NOTA: este script solo llama a los demás scripts, en orden.
# Los scripts se activan (se quita el #) a medida que se construyen, para
# que el repositorio corra completo sin errores en cada versión.
##########################################################

# Verificación del directorio de trabajo ----------------------------------
# Todas las rutas del repositorio son relativas a la carpeta raíz. Si R no
# está parado ahí, ningún script encontrará sus archivos. Detenemos la
# ejecución con un mensaje claro en lugar de un error confuso más adelante.
if (!file.exists("01_code/00_rundirectory.R")) {
  stop(
    "\n\nEl directorio de trabajo no es la carpeta raíz del repositorio.\n",
    "Directorio actual: ", getwd(), "\n",
    "Solución: use setwd(\"ruta/a/BDML-PS3\") o abra el proyecto desde esa ",
    "carpeta y vuelva a correr source(\"01_code/00_rundirectory.R\").\n",
    call. = FALSE
  )
}

# Carpeta de archivos intermedios (no versionada).
dir.create("03_temp", showWarnings = FALSE)

# Bloque A: datos (Etapa 1) ------------------------------------------------

# Obtiene los datos crudos de Kaggle (ver README, sección Datos).
source("01_code/01_get_data.R")

# Limpia la base de propiedades (train y test).
source("01_code/02_clean_data.R")

# Predictores externos: OSM, IDECA (TransMilenio), modelo de elevación y
# estrato por manzana (Datos Abiertos Bogotá).
source("01_code/03_build_geo_features.R")

# Predictores a partir del título y la descripción del anuncio.
source("01_code/04_build_text_features.R")

# Imputación de faltantes y conjuntos de predictores (base, completo y
# ampliado).
source("01_code/05_build_model_data.R")

# Similitud con Chapinero y particiones de validación cruzada (aleatoria y
# espacial) compartidas por todos los modelos.
source("01_code/06_create_cv_folds.R")

# Bloque B: las seis familias (Etapa 2; cada script genera sus envíos) ----
# Cada script puede correrse solo, una vez corridos los scripts 01 a 06.

source("01_code/07_model_linear_regression.R")
source("01_code/08_model_elastic_net.R")
source("01_code/09_model_cart.R")
source("01_code/10_model_random_forest.R")
source("01_code/11_model_boosting.R")
source("01_code/12_model_neural_network.R")

# Bloque C: mejoras sobre el algoritmo ganador (LightGBM) -------------------

# LightGBM que predice el precio por m² (donde la superficie es real).
source("01_code/13_model_boosting_precio_m2.R")
# Mezcla LightGBM + Elastic Net para extrapolar el precio en Chapinero.
source("01_code/14_model_blend_elastic_net.R")
# LightGBM de precio por m² con el precio de las viviendas vecinas.
source("01_code/15_model_boosting_vecinos.R")
# Mezcla del LightGBM de precio por m² (13) con la red neuronal (12); elige
# el peso de la red.
source("01_code/16_model_blend_red.R")
# Mejor modelo: LightGBM con vecinos (15) mezclado con la red (12).
source("01_code/17_model_blend_vecinos_red.R")

# Bloque D: análisis para las presentaciones -------------------------------

# Comparación de algoritmos (deck compare_equipo_03).
source("01_code/18_compare_algorithms.R")

# Mejor modelo, errores y decisión de compra (deck best_equipo_03).
source("01_code/19_best_model_analysis.R")
source("01_code/20_business_analysis.R")

# Bloque E: diapositivas ---------------------------------------------------
source("04_slides/compile_slides.R")
