##########################################################
# fun_osm.R
#
# Función para descargar capas de OpenStreetMap (OSM) con una copia local
# (caché). La primera vez descarga la capa y la guarda en disco; las
# siguientes la lee del disco. Así el script de variables geográficas se
# puede volver a correr sin repetir descargas lentas y sin depender de que
# el servidor de OSM esté disponible.
##########################################################

fun_osm <- function(.nombre, .clave, .valores, .bbox, .dir_cache) {
  #' Descargar una capa de OpenStreetMap, con caché local
  #'
  #' @description \code{fun_osm()} consulta OSM a través de la API Overpass
  #' (paquete osmdata) y devuelve todos los elementos que tienen la etiqueta
  #' \code{.clave} con alguno de los \code{.valores}, dentro del rectángulo
  #' \code{.bbox}. El resultado se guarda en \code{.dir_cache} como
  #' osm_<.nombre>.rds; si ese archivo ya existe, se lee en lugar de
  #' descargar. Como el servidor de OSM a veces falla, se intenta hasta tres
  #' veces, esperando un poco más entre cada intento.
  #'
  #' @param .nombre \code{character}. Nombre corto de la capa; se usa para el
  #' nombre del archivo en caché (por ejemplo, "parques").
  #' @param .clave \code{character}. Etiqueta (key) de OSM, por ejemplo
  #' "leisure" o "amenity".
  #' @param .valores \code{character}. Valores aceptados de la etiqueta, por
  #' ejemplo c("park", "garden").
  #' @param .bbox \code{numeric}. Rectángulo de búsqueda en grados, en el
  #' orden c(xmin, ymin, xmax, ymax).
  #' @param .dir_cache \code{character}. Carpeta donde se guardan las capas
  #' descargadas.
  #'
  #' @returns Objeto \code{osmdata} con puntos, líneas y polígonos
  #' (\code{osm_points}, \code{osm_lines}, \code{osm_polygons}, etc.).
  archivo <- file.path(.dir_cache, paste0("osm_", .nombre, ".rds"))
  if (file.exists(archivo)) return(readRDS(archivo))
  
  message("Descargando OSM: ", .nombre)
  for (intento in 1:3) {
    resultado <- tryCatch({
      q <- osmdata::opq(bbox = .bbox, timeout = 180)
      q <- osmdata::add_osm_feature(q, key = .clave, value = .valores)
      osmdata::osmdata_sf(q, quiet = TRUE)
    }, error = function(e) e)
    if (!inherits(resultado, "error")) {
      saveRDS(resultado, archivo)
      return(resultado)
    }
    message("Intento ", intento, " fallido: ", conditionMessage(resultado))
    if (intento < 3) Sys.sleep(10 * intento)
  }
  stop("No se pudo descargar OSM: ", .nombre)
}
