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
# source("01_code/02_clean_data.R")

# Predictores externos de OpenStreetMap (mínimo 4).
# source("01_code/03_build_osm_features.R")

# Predictores a partir del título y la descripción del anuncio (mínimo 4).
# source("01_code/04_build_text_features.R")

# Distribución de precios, faltantes y diferencias de Chapinero.
# source("01_code/05_descriptive_analysis.R")

# Particiones de validación cruzada (aleatoria y espacial) compartidas por
# todos los modelos.
# source("01_code/06_create_cv_folds.R")

# Bloque B: modelos (Etapa 2; cada script genera sus envíos a Kaggle) -----
# Cada script puede correrse solo, una vez corridos los scripts 01 a 06.

# source("01_code/07_model_linear_regression.R")
# source("01_code/08_model_elastic_net.R")
# source("01_code/09_model_cart.R")
# source("01_code/10_model_random_forest.R")
# source("01_code/11_model_boosting.R")
# source("01_code/12_model_neural_network.R")

# Bloque C: análisis para las presentaciones -------------------------------

# Etapa 2: comparación de algoritmos (deck compare_equipo_03).
# source("01_code/13_compare_algorithms.R")

# Etapa 3: validación espacial frente a la aleatoria y frente a Kaggle.
# source("01_code/14_spatial_validation.R")

# Etapa 4: mejor modelo, errores y decisión de compra (deck best_equipo_03).
# source("01_code/15_best_model_analysis.R")
# source("01_code/16_business_decision.R")
