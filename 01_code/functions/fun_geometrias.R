fun_geometrias <- function(osm, clave, valores,
                           tipos = c("osm_points", "osm_polygons",
                                     "osm_multipolygons"), a_puntos = FALSE) {
  geom <- list()
  for (tipo in tipos) {
    x <- osm[[tipo]]
    if (is.null(x) || nrow(x) == 0 || !clave %in% names(x)) next
    x <- x[!is.na(x[[clave]]) & x[[clave]] %in% valores, , drop = FALSE]
    if (nrow(x) == 0) next
    x <- sf::st_make_valid(sf::st_transform(x, crs_m))
    if (a_puntos && tipo != "osm_points") {
      x <- suppressWarnings(sf::st_point_on_surface(x))
    }
    geom[[length(geom) + 1]] <- sf::st_geometry(x)
  }
  if (!length(geom)) return(sf::st_sfc(crs = crs_m))
  do.call(c, geom)
}