##########################################################
# 03_build_geo_features.R
#
# Objetivo: construir predictores geograficos de cada inmueble.
# Insumos: 03_temp/train_clean.rds y 03_temp/test_clean.rds.
# Salidas: 03_temp/train_geo.rds y 03_temp/test_geo.rds.
# Descargas: 03_temp/geo_cache/.
#
# Variables:
#   - dist_transmilenio_m
#   - dist_via_principal_m
#   - dist_centro_comercial_m
#   - dist_parque_m
#   - dist_universidad_m
#   - num_restaurantes_cafes_500m
#   - pct_zona_verde_500m
#   - pendiente_terreno_pct
#   - estrato_manzana (nuevo, fuente oficial SDP / Datos Abiertos Bogota)
#
# Si ya existen train_geo.rds y test_geo.rds, solo se agrega
# estrato_manzana sin repetir los calculos de OSM y elevacion.
##########################################################
objetos_previos <- ls()

source("01_code/functions/fun_osm.R")
source("01_code/functions/fun_geometrias.R")

# Parametros --------------------------------------------------------------
dir_temp <- "03_temp"
dir_cache <- file.path(dir_temp, "geo_cache")
radio_m <- 500
crs_m <- 3116
z_dem <- 12
forzar_recalculo <- FALSE

# Paquetes ----------------------------------------------------------------
if (!require(pacman)) install.packages("pacman")
pacman::p_load(dplyr, sf, osmdata, terra, elevatr, jsonlite)

dir.create(dir_cache, recursive = TRUE, showWarnings = FALSE)
archivo_train <- file.path(dir_temp, "train_clean.rds")
archivo_test <- file.path(dir_temp, "test_clean.rds")
salida_train <- file.path(dir_temp, "train_geo.rds")
salida_test <- file.path(dir_temp, "test_geo.rds")

if (!all(file.exists(c(archivo_train, archivo_test)))) {
  stop("Faltan 03_temp/train_clean.rds o 03_temp/test_clean.rds. ",
       "Corra primero 01_code/02_clean_data.R.")
}

if (!forzar_recalculo && file.exists(salida_train) && file.exists(salida_test)) {
  message("Las ocho variables geograficas ya existen. No se recalculan.")
} else {
  
  # 1. Leer coordenadas y calcular una vez por ubicacion unica -----------
  train <- readRDS(archivo_train)
  test <- readRDS(archivo_test)
  
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
  
  # 2. Descargar y reutilizar datos externos ----------------------------
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
  
  # Capas OpenStreetMap ---------------------------------------------------
  osm_vias <- fun_osm("vias", "highway",
                      c("motorway", "trunk", "primary", "secondary"),
                      .bbox = bbox_osm, .dir_cache = dir_cache)
  vias <- fun_geometrias(osm_vias, "highway",
                         c("motorway", "trunk", "primary", "secondary"),
                         .crs = crs_m,
                         .tipos = c("osm_lines", "osm_multilines"))
  
  osm_cc <- fun_osm("cc", "shop", "mall",
                    .bbox = bbox_osm, .dir_cache = dir_cache)
  cc <- fun_geometrias(osm_cc, "shop", "mall", .crs = crs_m)
  
  osm_parques <- fun_osm("parques", "leisure", c("park", "garden"),
                         .bbox = bbox_osm, .dir_cache = dir_cache)
  parques <- fun_geometrias(osm_parques, "leisure", "park", .crs = crs_m)
  
  osm_universidades <- fun_osm("universidades", "amenity", "university",
                               .bbox = bbox_osm, .dir_cache = dir_cache)
  universidades <- fun_geometrias(osm_universidades, "amenity", "university",
                                  .crs = crs_m)
  
  osm_gastro <- fun_osm("gastro", "amenity", c("restaurant", "cafe"),
                        .bbox = bbox_osm, .dir_cache = dir_cache)
  gastronomia <- fun_geometrias(osm_gastro, "amenity",
                                c("restaurant", "cafe"), .crs = crs_m,
                                .a_puntos = TRUE)
  
  osm_bosques <- fun_osm("bosques", "natural", "wood",
                         .bbox = bbox_osm, .dir_cache = dir_cache)
  osm_cesped <- fun_osm("cesped", "landuse", c("grass", "forest"),
                        .bbox = bbox_osm, .dir_cache = dir_cache)
  verdes <- c(
    fun_geometrias(osm_parques, "leisure", c("park", "garden"), .crs = crs_m,
                   .tipos = c("osm_polygons", "osm_multipolygons")),
    fun_geometrias(osm_bosques, "natural", "wood", .crs = crs_m,
                   .tipos = c("osm_polygons", "osm_multipolygons")),
    fun_geometrias(osm_cesped, "landuse", c("grass", "forest"), .crs = crs_m,
                   .tipos = c("osm_polygons", "osm_multipolygons"))
  )
  if (!length(verdes)) stop("OSM no devolvio poligonos de zonas verdes.")
  verdes <- sf::st_collection_extract(sf::st_make_valid(verdes),
                                      "POLYGON", warn = FALSE)
  verdes <- verdes[!sf::st_is_empty(verdes)]
  if (!length(verdes)) stop("No quedaron zonas verdes validas.")
  verde_unido <- sf::st_sf(geometry = sf::st_union(verdes))
  
  # 3. Distancias, conteos y porcentaje de zonas verdes -----------------
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
  
  # 4. Pendiente a partir de un modelo digital de elevacion -------------
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
  
  # 5. Recuperar filas originales y guardar -----------------------------
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

# 6. NUEVO: estrato oficial de la manzana -------------------------------
# Se puede ejecutar sobre los RDS existentes, sin recalcular las 8
# variables anteriores. Fuente: Secretaria Distrital de Planeacion.
train_geo <- readRDS(salida_train)
test_geo <- readRDS(salida_test)

if (forzar_recalculo || !"estrato_manzana" %in% names(train_geo) ||
    !"estrato_manzana" %in% names(test_geo)) {
  
  archivo_estratos <- file.path(dir_cache, "estratificacion_manzanas.geojson")
  if (!file.exists(archivo_estratos)) {
    message("Descargando estratificacion oficial de Bogota...")
    # Recurso GeoJSON del portal Datos Abiertos Bogota (SDP).
    id_recurso <- "29f2d770-bd5d-4450-9e95-8737167ba12f"
    api <- paste0("https://datosabiertos.bogota.gov.co/api/3/action/resource_show?id=",
                  id_recurso)
    url <- jsonlite::fromJSON(api)$result$url
    archivo_tmp <- paste0(archivo_estratos, ".part")
    utils::download.file(url, archivo_tmp, mode = "wb", quiet = TRUE)
    file.rename(archivo_tmp, archivo_estratos)
  }
  estratos <- sf::st_read(archivo_estratos, quiet = TRUE)
  campo_estrato <- grep("^estrato$", names(estratos),
                        ignore.case = TRUE, value = TRUE)[1]
  estratos <- sf::st_transform(estratos[campo_estrato], crs_m)
  
  # Un unico cruce espacial para las coordenadas distintas de train y test.
  ubicaciones <- unique(rbind(train_geo[c("lon", "lat")],
                              test_geo[c("lon", "lat")]))
  puntos_estrato <- sf::st_as_sf(ubicaciones, coords = c("lon", "lat"), crs = 4326)
  puntos_estrato <- sf::st_transform(puntos_estrato, crs_m)
  coincidencias <- sf::st_intersects(puntos_estrato, estratos)
  id_manzana <- vapply(coincidencias,
                       function(i) if (length(i)) i[1] else NA_integer_,
                       integer(1))
  valor_estrato <- as.integer(estratos[[campo_estrato]][id_manzana])
  valor_estrato[!valor_estrato %in% 1:6] <- NA_integer_
  
  clave <- function(x) paste(x$lon, x$lat, sep = "|")
  train_geo$estrato_manzana <- valor_estrato[match(clave(train_geo), clave(ubicaciones))]
  test_geo$estrato_manzana <- valor_estrato[match(clave(test_geo), clave(ubicaciones))]
  
  saveRDS(train_geo, salida_train)
  saveRDS(test_geo, salida_test)
  message("Estrato agregado a train_geo.rds y test_geo.rds.")
  message("Estrato disponible: train = ",
          round(100 * mean(!is.na(train_geo$estrato_manzana)), 1),
          "%; test = ",
          round(100 * mean(!is.na(test_geo$estrato_manzana)), 1), "%.")
} else {
  message("estrato_manzana ya existe. No se recalcula.")
}

rm(list = setdiff(ls(), c(objetos_previos, "fun_osm", "fun_geometrias")))

