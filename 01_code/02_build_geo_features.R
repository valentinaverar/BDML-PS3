# PS3 - Features geograficas -----------------------------------------------
# Ejecutar desde la raiz del repositorio:
# source("01_code/03_build_geo_features.R")
# Entrada: 00_data/raw/{train,test}.csv
# Salida:  03_temp/{train_geo,test_geo}.rds
# Fuentes externas y raster: 03_temp/geo_cache/
# Distancias en linea recta, no por red vial.

# Parametros --------------------------------------------------------------
dir_raw <- file.path("00_data", "raw")
dir_temp <- "03_temp"
dir_cache <- file.path(dir_temp, "geo_cache")
radio_m <- 500
crs_m <- 3116
z_dem <- 12
forzar_recalculo <- FALSE

paquetes <- c("readr", "dplyr", "sf", "osmdata", "terra", "elevatr")
for (paquete in paquetes) {
  if (!require(paquete, character.only = TRUE)) {
    install.packages(paquete)
    library(paquete, character.only = TRUE)
  }
}

dir.create(dir_cache, recursive = TRUE, showWarnings = FALSE)
archivo_train <- file.path(dir_raw, "train.csv")
archivo_test <- file.path(dir_raw, "test.csv")
salida_train <- file.path(dir_temp, "train_geo.rds")
salida_test <- file.path(dir_temp, "test_geo.rds")

if (!all(file.exists(c(archivo_train, archivo_test)))) {
  stop("Faltan train.csv o test.csv en 00_data/raw/.")
}

if (!forzar_recalculo && file.exists(salida_train) && file.exists(salida_test)) {
  message("Las bases geograficas ya existen. Se omite el calculo.")
} else {
  
  # 1. Leer coordenadas y calcular una vez por ubicacion unica -----------
  train <- readr::read_csv(archivo_train, show_col_types = FALSE)
  test <- readr::read_csv(archivo_test, show_col_types = FALSE)
  
  coords <- rbind(train[c("lon", "lat")], test[c("lon", "lat")])
  if (anyNA(coords) || !all(is.finite(as.matrix(coords)))) {
    stop("Hay coordenadas faltantes o no numericas en los CSV.")
  }
  
  coords_unicas <- unique(coords)
  clave <- function(x) paste(x$lon, x$lat, sep = "|")
  indice <- match(clave(coords), clave(coords_unicas))
  puntos <- sf::st_as_sf(coords_unicas, coords = c("lon", "lat"), crs = 4326)
  puntos <- sf::st_transform(puntos, crs_m)
  message("Calculando para ", nrow(puntos), " ubicaciones unicas.")
  
  # Buscar fuentes tambien alrededor de los limites de la muestra.
  area <- sf::st_buffer(sf::st_as_sfc(sf::st_bbox(puntos)), 4000)
  bb <- sf::st_bbox(sf::st_transform(area, 4326))
  bbox_osm <- unname(c(bb["xmin"], bb["ymin"], bb["xmax"], bb["ymax"]))
  
  # 2. Descargar y reutilizar datos de OpenStreetMap --------------------
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
  
  # Extraer solo los elementos que tienen la etiqueta solicitada.
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
  
  # Estaciones troncales: capa oficial IDECA (MapServer, layer 9).
  archivo_tm <- file.path(dir_cache, "estaciones_troncales.geojson")
  if (!file.exists(archivo_tm)) {
    url <- paste0(
      "https://serviciosgis.catastrobogota.gov.co/arcgis/rest/services/",
      "Mapa_Referencia/Mapa_Referencia/MapServer/9/query",
      "?where=1%3D1&outFields=*&outSR=4326&f=geojson"
    )
    message("Descargando estaciones de TransMilenio...")
    archivo_tmp <- paste0(archivo_tm, ".part")
    utils::download.file(url, archivo_tmp, mode = "wb", quiet = TRUE)
    if (!file.rename(archivo_tmp, archivo_tm)) {
      stop("No se pudo guardar la capa de TransMilenio.")
    }
  }
  tm <- sf::st_read(archivo_tm, quiet = TRUE)
  if (!nrow(tm)) stop("La capa de estaciones esta vacia.")
  tm <- sf::st_geometry(sf::st_transform(tm, crs_m))
  
  osm_vias <- fun_osm("vias", "highway",
                      c("motorway", "trunk", "primary", "secondary"))
  vias <- fun_geometrias(osm_vias, "highway",
                         c("motorway", "trunk", "primary", "secondary"),
                         tipos = c("osm_lines", "osm_multilines"))
  
  osm_cc <- fun_osm("cc", "shop", "mall")
  cc <- fun_geometrias(osm_cc, "shop", "mall")
  
  osm_parques <- fun_osm("parques", "leisure", c("park", "garden"))
  parques <- fun_geometrias(osm_parques, "leisure", "park")
  
  osm_universidades <- fun_osm("universidades", "amenity", "university")
  universidades <- fun_geometrias(osm_universidades, "amenity", "university")
  
  osm_gastro <- fun_osm("gastro", "amenity", c("restaurant", "cafe"))
  gastronomia <- fun_geometrias(osm_gastro, "amenity",
                                c("restaurant", "cafe"), a_puntos = TRUE)
  
  osm_bosques <- fun_osm("bosques", "natural", "wood")
  osm_cesped <- fun_osm("cesped", "landuse", c("grass", "forest"))
  verdes <- c(
    fun_geometrias(osm_parques, "leisure", c("park", "garden"),
                   tipos = c("osm_polygons", "osm_multipolygons")),
    fun_geometrias(osm_bosques, "natural", "wood",
                   tipos = c("osm_polygons", "osm_multipolygons")),
    fun_geometrias(osm_cesped, "landuse", c("grass", "forest"),
                   tipos = c("osm_polygons", "osm_multipolygons"))
  )
  if (!length(verdes)) stop("OSM no devolvio poligonos de zonas verdes.")
  verdes <- sf::st_collection_extract(sf::st_make_valid(verdes),
                                      "POLYGON", warn = FALSE)
  verdes <- verdes[!sf::st_is_empty(verdes)]
  if (!length(verdes)) stop("No quedaron zonas verdes validas.")
  verde_unido <- sf::st_sf(geometry = sf::st_union(verdes))
  
  # 3. Distancias, conteo de restaurantes y porcentaje verde ------------
  fun_distancia <- function(destinos) {
    if (!length(destinos)) stop("Falta una capa geografica para distancias.")
    cercanos <- sf::st_nearest_feature(sf::st_geometry(puntos), destinos)
    as.numeric(sf::st_distance(sf::st_geometry(puntos),
                               destinos[cercanos], by_element = TRUE))
  }
  
  fun_conteo <- function(destinos, bloque = 250) {
    if (!length(destinos)) stop("No hay restaurantes o cafes en OSM.")
    resultado <- integer(nrow(puntos))
    for (inicio in seq(1, nrow(puntos), by = bloque)) {
      ids <- inicio:min(inicio + bloque - 1, nrow(puntos))
      vecinos <- sf::st_is_within_distance(sf::st_geometry(puntos)[ids],
                                           destinos, dist = radio_m)
      resultado[ids] <- lengths(vecinos)
    }
    resultado
  }
  
  fun_pct_verde <- function(bloque = 250) {
    resultado <- numeric(nrow(puntos))
    for (inicio in seq(1, nrow(puntos), by = bloque)) {
      ids <- inicio:min(inicio + bloque - 1, nrow(puntos))
      circulos <- sf::st_buffer(sf::st_geometry(puntos)[ids], radio_m)
      inter <- suppressWarnings(sf::st_intersection(
        sf::st_sf(pos = seq_along(ids), geometry = circulos), verde_unido
      ))
      if (!nrow(inter)) next
      areas <- tapply(as.numeric(sf::st_area(inter)), inter$pos, sum)
      pos <- as.integer(names(areas))
      resultado[ids[pos]] <- pmin(100, 100 * as.numeric(areas) /
                                    as.numeric(sf::st_area(circulos))[pos])
    }
    resultado
  }
  
  geo <- data.frame(
    dist_transmilenio_m = fun_distancia(tm),
    dist_via_principal_m = fun_distancia(vias),
    dist_centro_comercial_m = fun_distancia(cc),
    dist_parque_m = fun_distancia(parques),
    dist_universidad_m = fun_distancia(universidades),
    num_restaurantes_cafes_500m = fun_conteo(gastronomia)
  )
  geo$pct_zona_verde_500m <- fun_pct_verde()
  
  # 4. Pendiente a partir de un modelo digital de elevacion --------------
  archivo_pendiente <- file.path(dir_cache, "pendiente_pct.tif")
  if (!file.exists(archivo_pendiente)) {
    message("Descargando elevaciones y calculando pendiente...")
    marco <- sf::st_sf(geometry = sf::st_transform(
      sf::st_as_sfc(sf::st_bbox(puntos)), 4326))
    dem <- elevatr::get_elev_raster(locations = marco, z = z_dem,
                                    src = "aws", clip = "tile", verbose = FALSE)
    pendiente_grados <- terra::terrain(terra::rast(dem), v = "slope",
                                       unit = "degrees", neighbors = 8)
    pendiente_pct <- 100 * tan(pendiente_grados * pi / 180)
    terra::writeRaster(pendiente_pct, archivo_pendiente, overwrite = TRUE)
  }
  pendiente_pct <- terra::rast(archivo_pendiente)
  puntos_dem <- sf::st_transform(puntos, sf::st_crs(terra::crs(pendiente_pct)))
  geo$pendiente_terreno_pct <- terra::extract(
    pendiente_pct, terra::vect(puntos_dem), method = "bilinear")[, 2]
  
  # 5. Recuperar las filas originales y guardar -------------------------
  nombres <- c("dist_transmilenio_m", "dist_via_principal_m",
               "dist_centro_comercial_m", "dist_parque_m",
               "dist_universidad_m", "num_restaurantes_cafes_500m",
               "pendiente_terreno_pct", "pct_zona_verde_500m")
  adicionales <- geo[indice, nombres, drop = FALSE]
  rownames(adicionales) <- NULL
  n_train <- nrow(train)
  train_geo <- dplyr::bind_cols(train, adicionales[seq_len(n_train), , drop = FALSE])
  test_geo <- dplyr::bind_cols(test, adicionales[n_train + seq_len(nrow(test)), , drop = FALSE])
  stopifnot(nrow(train_geo) == nrow(train), nrow(test_geo) == nrow(test))
  saveRDS(train_geo, salida_train)
  saveRDS(test_geo, salida_test)
  message("Guardados: ", salida_train, " y ", salida_test)
  print(data.frame(
    variable = nombres,
    NA_train = colSums(is.na(train_geo[nombres])),
    NA_test = colSums(is.na(test_geo[nombres]))
  ), row.names = FALSE)
}
