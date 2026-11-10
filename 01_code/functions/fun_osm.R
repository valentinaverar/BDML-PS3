fun_osm <- function(nombre, clave, valores) {
  archivo <- file.path(dir_cache, paste0("osm_", nombre, ".rds"))
  if (file.exists(archivo)) return(readRDS(archivo))
  
  message("Descargando OSM: ", nombre)
  for (intento in 1:3) {
    resultado <- tryCatch({
      q <- osmdata::opq(bbox = bbox_osm, timeout = 180)
      q <- osmdata::add_osm_feature(q, key = clave, value = valores)
      osmdata::osmdata_sf(q, quiet = TRUE)
    }, error = function(e) e)
    if (!inherits(resultado, "error")) {
      saveRDS(resultado, archivo)
      return(resultado)
    }
    message("Intento ", intento, " fallido: ", conditionMessage(resultado))
    if (intento < 3) Sys.sleep(10 * intento)
  }
  stop("No se pudo descargar OSM: ", nombre)
}
