##########################################################
# 01_get_data.R
#
# Objetivo: dejar los datos crudos de la competencia de Kaggle en
# 00_data/raw/, listos para los siguientes scripts.
#
# Insumos:  ninguno (o el zip / los CSV de la competencia, si ya existen).
# Salidas:  00_data/raw/ con train.csv, train_personas.csv,
#           test_hogares.csv, test.csv, sample_submission.csv
#           
#
# Lógica ("esquema mixto"). Se usa la primera opción que funcione:
#   1. Si los CSV ya están en 00_data/raw/, se usan tal cual.
#   2. Si está el zip de la competencia en 00_data/raw/, se descomprime.
#   3. Si está instalada la CLI de Kaggle, se descarga el zip y se
#      descomprime.
#   4. Si nada de lo anterior es posible, el script se detiene con un
#      mensaje de error que explica cómo obtener los datos.
#
# Decisión: los datos NO se guardan en el repositorio. La plantilla del curso
# pide no versionar datos crudos y, además, train_personas.csv (~242 MB)
# supera el límite de 100 MB por archivo de GitHub. Los datos se obtienen
# siempre de la fuente original (Kaggle).
##########################################################


# Parámetros --------------------------------------------------------------
competencia <- "uniandes-bdml-202620-ps-3"
dir_raw     <- file.path("00_data", "raw")
archivo_zip <- file.path(dir_raw, paste0(competencia, ".zip"))

# Archivos que deben existir para que el resto del código funcione.
archivos_requeridos <- c("train.csv", "test.csv",
                         "submission_template.csv")

dir.create(dir_raw, recursive = TRUE, showWarnings = FALSE)

# Funciones auxiliares ----------------------------------------------------
fun_faltantes <- function() {
  # Devuelve los archivos requeridos que todavía no están en 00_data/raw/.
  archivos_requeridos[!file.exists(file.path(dir_raw, archivos_requeridos))]
}

fun_descomprimir <- function(.zip) {
  # Extrae el zip de la competencia en 00_data/raw/. El zip se conserva
  # (está en el .gitignore) para poder volver a extraer sin descargar.
  message("Descomprimiendo ", .zip, " ...")
  utils::unzip(zipfile = .zip, exdir = dir_raw)
}

# Paso 1: ¿ya están los datos? -------------------------------------------
if (length(fun_faltantes()) == 0) {
  message("Datos encontrados en ", dir_raw, ". No es necesario descargar.")
} else {
  
  # Paso 2: ¿está el zip de la competencia en la carpeta? ----------------
  if (file.exists(archivo_zip)) {
    fun_descomprimir(archivo_zip)
    
    # Paso 3: ¿está instalada la CLI de Kaggle? ----------------------------
  } else if (nzchar(Sys.which("kaggle"))) {
    message("Descargando los datos de Kaggle con la CLI ...")
    # Equivale a correr en la terminal:
    # kaggle competitions download -c uniandes-bdml-202620-ps-3 -p 00_data/raw
    estado <- system2(command = "kaggle",
                      args = c("competitions", "download",
                               "-c", competencia,
                               "-p", shQuote(dir_raw)))
    
    # La CLI puede estar instalada pero fallar (token ausente o inválido,
    # reglas de la competencia no aceptadas, sin internet).
    if (estado != 0 || !file.exists(archivo_zip)) {
      stop(
        "\n\nLa CLI de Kaggle está instalada, pero la descarga falló ",
        "(código de salida: ", estado, ").\n",
        "Causas probables:\n",
        "  - No hay un token de API configurado en ~/.kaggle/kaggle.json.\n",
        "  - No ha aceptado las reglas de la competencia en ",
        "https://www.kaggle.com/competitions/", competencia, "/rules\n",
        "  - No hay conexión a internet.\n",
        "Revise el mensaje de la CLI arriba y vuelva a correr el código.\n",
        call. = FALSE
      )
    }
    fun_descomprimir(archivo_zip)
    
    # Paso 4: no hay forma de obtener los datos ----------------------------
  } else {
    stop(
      "\n\nNo se encontraron los datos de la competencia y la CLI de Kaggle ",
      "no está disponible.\n\n",
      "Opción A (automática): configure la CLI de Kaggle, una sola vez:\n",
      "  1. En la terminal: pip install kaggle\n",
      "  2. En kaggle.com: Settings > API > Create New Token. Guarde el\n",
      "     archivo descargado en ~/.kaggle/kaggle.json\n",
      "  3. Acepte las reglas de la competencia en\n",
      "     https://www.kaggle.com/competitions/", competencia, "/rules\n",
      "  4. Vuelva a correr source(\"01_code/00_rundirectory.R\").\n",
      "  Si ya instaló la CLI y ve este mensaje, R no la encuentra en el PATH\n",
      "  (frecuente en RStudio). Corra el código desde la terminal con\n",
      "  R CMD BATCH 01_code/00_rundirectory.R\n\n",
      "Opción B (manual): descargue el zip desde\n",
      "  https://www.kaggle.com/competitions/", competencia, "/data\n",
      "  (botón Download All) y guárdelo, sin descomprimir, en\n",
      "  ", archivo_zip, "\n",
      call. = FALSE
    )
  }
}

# Verificación final ------------------------------------------------------
# Confirmamos que el zip contenía lo esperado antes de seguir.
faltantes <- fun_faltantes()
if (length(faltantes) > 0) {
  stop(
    "\n\nFaltan archivos en ", dir_raw, ": ",
    paste(faltantes, collapse = ", "), ".\n",
    "Borre el zip de esa carpeta y vuelva a correr el código para ",
    "descargarlo de nuevo.\n",
    call. = FALSE
  )
}

message("Listo: los ", length(archivos_requeridos),
        " archivos requeridos están en ", dir_raw, ".")

# Limpiamos los objetos de este script para no contaminar los siguientes.
rm(competencia, dir_raw, archivo_zip, archivos_requeridos,
   fun_faltantes, fun_descomprimir, faltantes)
if (exists("estado")) rm(estado)
