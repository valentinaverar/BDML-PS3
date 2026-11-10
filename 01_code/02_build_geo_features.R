# PS3 - Construcción de características geográficas -----------------------
# Ejecutar desde la raíz del proyecto:
# source("01_code/03_build_geo_features.R")
# Instalar una vez: install.packages(c("readr", "dplyr", "sf", "osmdata",
#                                   "terra", "elevatr"))
#
# Entradas : 00_data/raw/train.csv, 00_data/raw/test.csv
# Salidas  : 03_temp/train_geo.rds, 03_temp/test_geo.rds
# Cache    : 03_temp/geo_cache/... (no subir a GitHub)
# AVISO    : las distancias son euclidianas, no distancias caminando.

# set-up

paquetes <- c("readr", "dplyr", "sf", "osmdata", "terra", "elevatr")

for (paquete in paquetes) {
  if (!require(paquete, character.only = TRUE)) {
    install.packages(paquete)
    library(paquete, character.only = TRUE)
  }
}

dir_raw <- file.path("00_data", "raw")
dir_temp <- "03_temp"
radio_m <- 500
crs_m <- 3116                   # MAGNA-SIRGAS / Colombia Bogotá (metros)
z_dem <- 12                     # resolución DEM; reducir a 11 si pesa mucho
forzar_recalculo <- FALSE       # TRUE recalcula features; reutiliza capas cacheadas
version_features <- "v1"



archivo_train <- file.path(dir_raw, "train.csv")
archivo_test  <- file.path(dir_raw, "test.csv")
if (!all(file.exists(c(archivo_train, archivo_test)))) {
  stop("Faltan train.csv o test.csv en 00_data/raw/.", call. = FALSE)
}
dir.create(dir_temp, recursive = TRUE, showWarnings = FALSE)

# Identifica una versión de cache compatible con ambos archivos de entrada.
md5_train <- unname(tools::md5sum(archivo_train))
md5_test  <- unname(tools::md5sum(archivo_test))
dir_cache <- file.path(dir_temp, "geo_cache",
                       paste0(version_features, "_", substr(md5_train, 1, 10),
                              "_", substr(md5_test, 1, 10)))
dir.create(dir_cache, recursive = TRUE, showWarnings = FALSE)
archivo_resultado_train <- file.path(dir_temp, "train_geo.rds")
archivo_resultado_test  <- file.path(dir_temp, "test_geo.rds")
archivo_firma <- file.path(dir_temp, "geo_features_firma.rds")
firma_actual <- basename(dir_cache)

if (!forzar_recalculo && all(file.exists(c(archivo_resultado_train,
                                           archivo_resultado_test,
                                           archivo_firma))) &&
    identical(readRDS(archivo_firma), firma_actual)) {
  message("Features geográficas existentes y actualizadas. Se omite el cálculo.")
} else {
  
  # Paso 1. Datos originales
  train <- readr::read_csv(archivo_train, show_col_types = FALSE)
  test  <- readr::read_csv(archivo_test, show_col_types = FALSE)
  
  fun_columna <- function(df, candidatos, etiqueta) {
    nombres <- tolower(names(df))
    indice <- match(tolower(candidatos), nombres)
    indice <- indice[!is.na(indice)]
    if (length(indice) == 0) {
      stop("No encuentro columna de ", etiqueta, ". Columnas disponibles: ",
           paste(names(df), collapse = ", "),
           ". Ajuste los candidatos en fun_coords().", call. = FALSE)
    }
    names(df)[indice[1]]
  }
  
  fun_coords <- function(df, etiqueta) {
    col_lon <- fun_columna(df, c("lon", "longitude", "longitud", "lng",
                                 "long", "x"), paste0("longitud en ", etiqueta))
    col_lat <- fun_columna(df, c("lat", "latitude", "latitud", "y"),
                           paste0("latitud en ", etiqueta))
    # La conversión explícita acepta columnas numéricas o texto con punto decimal.
    lon <- suppressWarnings(as.numeric(df[[col_lon]]))
    lat <- suppressWarnings(as.numeric(df[[col_lat]]))
    message(etiqueta, ": usando ", col_lon, " y ", col_lat, ".")
    data.frame(lon = lon, lat = lat)
  }
  
  coords_train <- fun_coords(train, "train")
  coords_test <- fun_coords(test, "test")
  coords <- rbind(coords_train, coords_test)
  validas <- is.finite(coords$lon) & is.finite(coords$lat) &
    coords$lon >= -75.5 & coords$lon <= -73.0 &
    coords$lat >= 3.5 & coords$lat <= 5.3
  
  if (!any(validas)) {
    stop("No se encontraron coordenadas WGS84 válidas de Bogotá. ",
         "Revise nombres, orden y sistema de coordenadas.", call. = FALSE)
  }
  if (any(!validas)) {
    warning(sum(!validas), " inmuebles sin coordenadas WGS84 válidas; ",
            "sus variables geográficas quedarán como NA.")
  }
  
  # Calculamos una sola vez para ubicaciones repetidas, sin modificar el
  # orden de las filas de train/test.
  coords_validas <- coords[validas, , drop = FALSE]
  claves <- paste(coords_validas$lon, coords_validas$lat, sep = "|")
  idx_unicos <- !duplicated(claves)
  idx_repetidas <- match(claves, claves[idx_unicos])
  puntos <- sf::st_as_sf(coords_validas[idx_unicos, , drop = FALSE],
                         coords = c("lon", "lat"), crs = 4326)
  puntos <- sf::st_transform(puntos, crs_m)
  message("Ubicaciones únicas: ", nrow(puntos), ".")
  
  # Extendemos el bbox 4 km para reducir problemas en los bordes.
  area <- sf::st_buffer(sf::st_as_sfc(sf::st_bbox(puntos)), 4000)
  bb <- sf::st_bbox(sf::st_transform(area, 4326))
  bbox_osm <- unname(c(bb["xmin"], bb["ymin"], bb["xmax"], bb["ymax"]))
  
  # Paso 2. Descargar y cachear capas
  fun_osm <- function(nombre, clave, valores) {
    archivo <- file.path(dir_cache, paste0("osm_", nombre, ".rds"))
    if (file.exists(archivo)) return(readRDS(archivo))
    message("Consultando OSM: ", nombre, " ...")
    for (intento in seq_len(3)) {
      resultado <- tryCatch({
        consulta <- osmdata::opq(bbox = bbox_osm, timeout = 180)
        consulta <- osmdata::add_osm_feature(consulta, key = clave,
                                             value = valores)
        osmdata::osmdata_sf(consulta, quiet = TRUE)
      }, error = function(e) e)
      if (!inherits(resultado, "error")) {
        saveRDS(resultado, archivo)
        return(resultado)
      }
      message("Intento ", intento, ": ", conditionMessage(resultado))
      if (intento < 3) Sys.sleep(10 * intento)
    }
    stop("No se pudo descargar la capa OSM '", nombre,
         "'. Intente más tarde; lo ya descargado queda cacheado.",
         call. = FALSE)
  }
  
  # Extrae SOLO objetos con el tag pedido. Los nodos auxiliares de las
  # geometrías OSM no se deben contar como restaurantes o centros comerciales.
  fun_geometrias <- function(osm, clave, valores,
                             tipos = c("osm_points", "osm_polygons",
                                       "osm_multipolygons"),
                             convertir_a_puntos = FALSE) {
    geometrías <- list()
    for (tipo in tipos) {
      x <- osm[[tipo]]
      if (is.null(x) || nrow(x) == 0 || !clave %in% names(x)) next
      filtro <- !is.na(x[[clave]]) & x[[clave]] %in% valores
      x <- x[filtro, , drop = FALSE]
      if (nrow(x) == 0) next
      x <- sf::st_transform(x, crs_m)
      x <- sf::st_make_valid(x)
      if (convertir_a_puntos && tipo != "osm_points") {
        x <- suppressWarnings(sf::st_point_on_surface(x))
      }
      geometrías[[length(geometrías) + 1]] <- sf::st_geometry(x)
    }
    if (length(geometrías) == 0) return(sf::st_sfc(crs = crs_m))
    do.call(c, geometrías)
  }
  
  # Estaciones reales del sistema troncal, fuente IDECA / TransMilenio.
  archivo_tm <- file.path(dir_cache, "estaciones_troncales.geojson")
  if (!file.exists(archivo_tm)) {
    message("Descargando estaciones troncales desde IDECA ...")
    url_tm <- paste0(
      "https://serviciosgis.catastrobogota.gov.co/arcgis/rest/services/",
      "Mapa_Referencia/Mapa_Referencia/MapServer/9/query",
      "?where=1%3D1&outFields=*&outSR=4326&f=geojson"
    )
    temporal <- paste0(archivo_tm, ".part")
    estado <- tryCatch(utils::download.file(url_tm, temporal,
                                            mode = "wb", quiet = TRUE),
                       error = function(e) e)
    if (inherits(estado, "error") || !is.numeric(estado) || estado != 0) {
      stop("Falló la descarga oficial de estaciones de TransMilenio. ",
           "Revise conexión/servicio. URL: ", url_tm, call. = FALSE)
    }
    file.rename(temporal, archivo_tm)
  }
  tm <- tryCatch(sf::st_read(archivo_tm, quiet = TRUE),
                 error = function(e) stop("GeoJSON de estaciones inválido: ",
                                          conditionMessage(e), call. = FALSE))
  if (nrow(tm) == 0) stop("La capa oficial de TransMilenio está vacía.")
  tm <- sf::st_geometry(sf::st_transform(tm, crs_m))
  
  osm_vias <- fun_osm("vias", "highway",
                      c("motorway", "trunk", "primary", "secondary"))
  vias <- fun_geometrias(osm_vias, "highway",
                         c("motorway", "trunk", "primary", "secondary"),
                         tipos = c("osm_lines", "osm_multilines"))
  
  osm_cc <- fun_osm("centros_comerciales", "shop", "mall")
  centros_comerciales <- fun_geometrias(osm_cc, "shop", "mall")
  
  osm_parques <- fun_osm("parques_jardines", "leisure", c("park", "garden"))
  parques <- fun_geometrias(osm_parques, "leisure", "park")
  
  osm_universidades <- fun_osm("universidades", "amenity", "university")
  universidades <- fun_geometrias(osm_universidades, "amenity", "university")
  
  osm_gastro <- fun_osm("gastronomia", "amenity", c("restaurant", "cafe"))
  gastronomia <- fun_geometrias(osm_gastro, "amenity",
                                c("restaurant", "cafe"),
                                convertir_a_puntos = TRUE)
  
  osm_vegetacion <- fun_osm("bosques", "natural", "wood")
  osm_cesped <- fun_osm("cesped_bosque", "landuse", c("grass", "forest"))
  
  # Unión sin doble conteo de áreas verdes superpuestas.
  verdes <- c(
    fun_geometrias(osm_parques, "leisure", c("park", "garden"),
                   tipos = c("osm_polygons", "osm_multipolygons")),
    fun_geometrias(osm_vegetacion, "natural", "wood",
                   tipos = c("osm_polygons", "osm_multipolygons")),
    fun_geometrias(osm_cesped, "landuse", c("grass", "forest"),
                   tipos = c("osm_polygons", "osm_multipolygons"))
  )
  if (length(verdes) == 0) stop("OSM no devolvió polígonos verdes.")
  verdes <- sf::st_make_valid(verdes)
  verdes <- sf::st_collection_extract(verdes, "POLYGON", warn = FALSE)
  verdes <- verdes[!sf::st_is_empty(verdes)]
  if (length(verdes) == 0) stop("No quedaron polígonos verdes válidos.")
  verde_unido <- sf::st_sf(geometry = sf::st_union(verdes))
  
  # Paso 3. Construcción de las ocho features
  fun_distancia <- function(p, destinos, nombre) {
    if (length(destinos) == 0) stop("No hay elementos para: ", nombre)
    cercanos <- sf::st_nearest_feature(sf::st_geometry(p), destinos)
    as.numeric(sf::st_distance(sf::st_geometry(p), destinos[cercanos],
                               by_element = TRUE))
  }
  
  fun_contar_500m <- function(p, destinos, radio, bloque = 250) {
    if (length(destinos) == 0) stop("No hay puntos gastronómicos en OSM.")
    resultado <- integer(nrow(p))
    for (inicio in seq(1L, nrow(p), by = bloque)) {
      ids <- seq.int(inicio, min(inicio + bloque - 1L, nrow(p)))
      vecinos <- sf::st_is_within_distance(sf::st_geometry(p)[ids],
                                           destinos, dist = radio)
      resultado[ids] <- lengths(vecinos)
    }
    resultado
  }
  
  fun_porcentaje_verde <- function(p, poligono_verde, radio, bloque = 250) {
    resultado <- numeric(nrow(p))
    for (inicio in seq(1L, nrow(p), by = bloque)) {
      ids <- seq.int(inicio, min(inicio + bloque - 1L, nrow(p)))
      buffers <- sf::st_buffer(sf::st_geometry(p)[ids], radio)
      capa_buffers <- sf::st_sf(pos = seq_along(ids), geometry = buffers)
      inter <- suppressWarnings(sf::st_intersection(capa_buffers,
                                                    poligono_verde))
      if (nrow(inter) == 0) next
      area_por_pos <- tapply(as.numeric(sf::st_area(inter)), inter$pos, sum)
      pos <- as.integer(names(area_por_pos))
      resultado[ids[pos]] <- pmin(100, 100 * as.numeric(area_por_pos) /
                                    as.numeric(sf::st_area(buffers))[pos])
    }
    resultado
  }
  
  message("Calculando distancias y densidades...")
  geo_unicas <- data.frame(
    dist_transmilenio_m = fun_distancia(puntos, tm, "TransMilenio"),
    dist_via_principal_m = fun_distancia(puntos, vias, "vías principales"),
    dist_centro_comercial_m = fun_distancia(puntos, centros_comerciales,
                                            "centros comerciales"),
    dist_parque_m = fun_distancia(puntos, parques, "parques"),
    dist_universidad_m = fun_distancia(puntos, universidades,
                                       "universidades"),
    num_restaurantes_cafes_500m = fun_contar_500m(puntos, gastronomia, radio_m)
  )
  
  message("Calculando porcentaje de zonas verdes...")
  geo_unicas$pct_zona_verde_500m <-
    fun_porcentaje_verde(puntos, verde_unido, radio_m)
  
  # Pendiente en porcentaje a partir de un DEM (no confundir con grados).
  archivo_pendiente <- file.path(dir_cache, "pendiente_pct.tif")
  if (!file.exists(archivo_pendiente)) {
    message("Descargando modelo digital de elevación para pendiente...")
    marco_dem <- sf::st_sf(geometry = sf::st_transform(
      sf::st_as_sfc(sf::st_bbox(puntos)), 4326))
    dem <- elevatr::get_elev_raster(locations = marco_dem, z = z_dem,
                                    src = "aws", clip = "tile", verbose = FALSE)
    elevacion <- terra::rast(dem)
    pendiente_grados <- terra::terrain(elevacion, v = "slope",
                                       unit = "degrees", neighbors = 8)
    pendiente_pct <- 100 * tan(pendiente_grados * pi / 180)
    terra::writeRaster(pendiente_pct, archivo_pendiente, overwrite = TRUE)
  }
  pendiente_pct <- terra::rast(archivo_pendiente)
  # Transformar coordenadas al CRS del raster antes de extraer valores.
  puntos_dem <- sf::st_transform(puntos, sf::st_crs(terra::crs(pendiente_pct)))
  geo_unicas$pendiente_terreno_pct <-
    terra::extract(pendiente_pct, terra::vect(puntos_dem),
                   method = "bilinear")[, 2]
  
  # Reordenar variables según la selección del equipo.
  nombres_features <- c(
    "dist_transmilenio_m", "dist_via_principal_m",
    "dist_centro_comercial_m", "dist_parque_m",
    "dist_universidad_m", "num_restaurantes_cafes_500m",
    "pendiente_terreno_pct", "pct_zona_verde_500m"
  )
  geo_unicas <- geo_unicas[, nombres_features, drop = FALSE]
  
  # Paso 4. Restaurar filas originales y guardar
  nuevas <- as.data.frame(matrix(NA_real_, nrow = nrow(coords),
                                 ncol = length(nombres_features)))
  names(nuevas) <- nombres_features
  nuevas[validas, ] <- geo_unicas[idx_repetidas, , drop = FALSE]
  
  n_train <- nrow(train)
  train_geo <- dplyr::bind_cols(train,
                                nuevas[seq_len(n_train), , drop = FALSE])
  test_geo <- dplyr::bind_cols(test,
                               nuevas[n_train + seq_len(nrow(test)), , drop = FALSE])
  
  stopifnot(nrow(train_geo) == nrow(train), nrow(test_geo) == nrow(test))
  saveRDS(train_geo, archivo_resultado_train)
  saveRDS(test_geo, archivo_resultado_test)
  saveRDS(firma_actual, archivo_firma)
  
  message("Listo. Train: ", nrow(train_geo), " filas. Test: ",
          nrow(test_geo), " filas.")
  message("Guardados: ", archivo_resultado_train, " y ", archivo_resultado_test)
  print(data.frame(variable = nombres_features,
                   na_train = colSums(is.na(train_geo[nombres_features])),
                   na_test = colSums(is.na(test_geo[nombres_features]))),
        row.names = FALSE)
}

### Limpiamos

rm()