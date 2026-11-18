##########################################################
# fun_geometrias.R
#
# Función para extraer, de una capa descargada de OpenStreetMap, solo las
# geometrías que interesan, en el sistema de coordenadas de trabajo. Se usa
# en 03_build_geo_features.R para preparar los destinos (vías, parques,
# universidades, etc.) antes de medir distancias.
##########################################################

fun_geometrias <- function(.osm, .clave, .valores, .crs,
                           .tipos = c("osm_points", "osm_polygons",
                                      "osm_multipolygons"),
                           .a_puntos = FALSE) {
  #' Extraer geometrías de una capa de OSM
  #'
  #' @description \code{fun_geometrias()} recorre los tipos de geometría
  #' pedidos (puntos, líneas o polígonos) y, en cada uno, conserva solo los
  #' elementos cuya etiqueta \code{.clave} toma alguno de los
  #' \code{.valores}. Esto es necesario porque osmdata también devuelve
  #' elementos auxiliares (por ejemplo, los vértices de un polígono como
  #' puntos sueltos). Luego proyecta al CRS de trabajo, corrige geometrías
  #' inválidas y une todo en un solo objeto.
  #'
  #' @param .osm Objeto \code{osmdata}, por ejemplo el resultado de
  #' \code{fun_osm()}.
  #' @param .clave \code{character}. Etiqueta de OSM por la que se filtra.
  #' @param .valores \code{character}. Valores aceptados de la etiqueta.
  #' @param .crs CRS de destino (código EPSG); debe ser proyectado, en
  #' metros, para que las distancias posteriores queden en metros.
  #' @param .tipos \code{character}. Tipos de geometría de osmdata que se
  #' incluyen, por ejemplo c("osm_lines", "osm_multilines") para vías.
  #' @param .a_puntos \code{logical}. Si es TRUE, convierte los polígonos en
  #' un punto dentro de cada uno. Sirve para contar establecimientos: un
  #' restaurante mapeado como edificio cuenta igual que uno mapeado como
  #' punto.
  #'
  #' @returns Objeto \code{sfc} con las geometrías en \code{.crs}. Si no se
  #' encontró ningún elemento, un \code{sfc} vacío.
  geom <- list()
  for (tipo in .tipos) {
    x <- .osm[[tipo]]
    if (is.null(x) || nrow(x) == 0 || !.clave %in% names(x)) next
    x <- x[!is.na(x[[.clave]]) & x[[.clave]] %in% .valores, , drop = FALSE]
    if (nrow(x) == 0) next
    x <- sf::st_make_valid(sf::st_transform(x, .crs))
    if (.a_puntos && tipo != "osm_points") {
      x <- suppressWarnings(sf::st_point_on_surface(x))
    }
    geom[[length(geom) + 1]] <- sf::st_geometry(x)
  }
  if (!length(geom)) return(sf::st_sfc(crs = .crs))
  do.call(c, geom)
}
