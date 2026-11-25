##########################################################
# 19_best_model_analysis.R
#
# Objetivo: producir las figuras, tablas y cifras del deck del mejor modelo
# (best_equipo_03): en qué se diferencia Chapinero (incluido un mapa del
# precio por m² en Bogotá), cómo se construyeron
# los pseudo-Chapineros para elegir la CV espacial, la cascada de envíos a
# Kaggle, la importancia de las variables, lo que promete la CV frente a
# Kaggle y el sesgo del modelo por rango de precio.
#
# Etapa del Problem Set: Etapa 1 (cómo difiere Chapinero), Etapa 3
# (validación espacial) y Etapa 4 (el mejor modelo).
# Insumos:  03_temp/train_features.rds, test_features.rds, train_model.rds,
#           similitud_chapinero.rds, cv_folds.rds,
#           oof_blend_vecinos_red.rds (scripts 05, 06 y 17);
#           02_outputs/tables/registro_modelos.csv,
#           zonas_similitud_chapinero.csv, seleccion_cv.csv (script 06),
#           LGBM_vecinos_importancia.csv (script 15), LGBM_completo_ensayos.csv
#           (script 11), BLEND_m2_red_pesos.csv (script 16);
#           03_temp/test_model.rds y los envíos del 15, el 12 y el 17 en
#           02_outputs/predictions/ (ejemplo de una vivienda de Chapinero);
#           00_data/kaggle_scores.csv;
#           00_data/external/geo_cache/estratificacion_manzanas.rds (fondo
#           del mapa).
# Salidas:  02_outputs/figures/best_*.pdf y
#           02_outputs/tables/best_datos.tex (cifras del deck como comandos
#           de LaTeX, por ejemplo \MAEKaggle).
#
# Decisiones:
#   - Este script no entrena modelos: solo lee resultados guardados.
#   - Las cifras se escriben como comandos de LaTeX para que el deck siempre
#     muestre los valores de la última corrida.
#   - Mejor modelo = el de menor puntaje público en kaggle_scores.csv.
#   - Las zonas de los pseudo-Chapineros se reconstruyen con el mismo
#     k-means del script 06 (misma semilla), y se verifica que coincidan
#     con la tabla que guardó ese script.
#   - Las figuras usan el mismo estilo del PS2 (colores, tipografía, títulos
#     centrados y fuente al pie).
##########################################################

# Objetos que ya existían en el entorno antes de correr este script. Al
# final se borra solo lo que este script creó, sin tocar lo demás.
objetos_previos <- ls()

# Paquetes -----------------------------------------------------------------
if (!require(pacman)) install.packages("pacman")
pacman::p_load(dplyr, tidyr, ggplot2, ggrepel, sf, FNN, patchwork, forcats)

source("01_code/functions/fun_vecinos.R")

# Parámetros ---------------------------------------------------------------
semilla <- 2026
crs_m <- 3116      # Coordenadas proyectadas en metros, como en el script 03.
n_zonas <- 20      # Igual que en el script 06.
n_pseudo <- 3      # Igual que en el script 06.
min_tamano_zona <- 1000

# Estilo de las figuras (el mismo del PS2) ------------------------------------
color_1 <- "#2a78d6"
color_2 <- "#0f3570"        # Azul oscuro.
color_3 <- "#7fb0ec"        # Azul claro.
color_4 <- "#b9d3f3"        # Azul muy claro.
gris <- "#a3a29b"
texto_1 <- "#0b0b0b"
texto_2 <- "#52514e"
tema <- theme_minimal(base_size = 11) +
  theme(panel.grid.minor = element_blank(),
        axis.text = element_text(color = texto_2),
        axis.title = element_text(color = texto_2),
        plot.title = element_text(color = texto_1, face = "bold", hjust = 0.5,
                                  margin = margin(b = 4)),
        plot.subtitle = element_text(color = texto_2, hjust = 0.5,
                                     margin = margin(b = 14)),
        plot.caption = element_text(color = texto_2, hjust = 0.5,
                                    margin = margin(t = 14)),
        plot.margin = margin(12, 14, 10, 14),
        plot.title.position = "plot",
        plot.caption.position = "plot",
        legend.position = "top",
        legend.text = element_text(color = texto_2))
fun_coma <- function(.x, .digits = 1) {
  formatC(.x, format = "f", digits = .digits, decimal.mark = ",",
          big.mark = ".")
}
fun_entero <- function(.x) {
  formatC(round(.x), format = "d", big.mark = ".", decimal.mark = ",")
}
fun_pct <- function(.x, .digits = 0) paste0(fun_coma(100 * .x, .digits), "\\%")
fun_guardar <- function(.grafico, .nombre, .ancho = 7.5, .alto = 4.6) {
  #' Guardar una figura en PDF
  #'
  #' Usa cairo_pdf si funciona en este computador. En algunas instalaciones
  #' de R (p. ej. macOS sin XQuartz) cairo no está disponible y ggsave() no
  #' guarda nada sin dar error; en ese caso se usa el dispositivo pdf
  #' estándar con codificación WinAnsi (tildes, eñes, ² y ×). Al final se
  #' verifica que el archivo exista.
  ruta <- paste0("02_outputs/figures/", .nombre, ".pdf")
  unlink(ruta)
  con_cairo <- tryCatch({
    ggsave(ruta, .grafico, width = .ancho, height = .alto, device = cairo_pdf)
    file.exists(ruta) && file.size(ruta) > 0
  }, warning = function(w) FALSE, error = function(e) FALSE)
  if (!con_cairo) {
    unlink(ruta)
    ggsave(ruta, .grafico, width = .ancho, height = .alto,
           device = "pdf", encoding = "WinAnsi.enc")
  }
  if (!file.exists(ruta)) stop("No se pudo guardar la figura ", ruta, ".")
}

# Datos --------------------------------------------------------------------
train <- readRDS("03_temp/train_features.rds")
test <- readRDS("03_temp/test_features.rds")
train_model <- readRDS("03_temp/train_model.rds")
similitud <- readRDS("03_temp/similitud_chapinero.rds")
cv_folds <- readRDS("03_temp/cv_folds.rds")
oof <- readRDS("03_temp/oof_blend_vecinos_red.rds")
registro <- read.csv("02_outputs/tables/registro_modelos.csv")
kaggle <- read.csv("00_data/kaggle_scores.csv")
zonas <- read.csv("02_outputs/tables/zonas_similitud_chapinero.csv")
seleccion_cv <- read.csv("02_outputs/tables/seleccion_cv.csv")
importancia <- read.csv("02_outputs/tables/LGBM_vecinos_importancia.csv")
ensayos <- read.csv("02_outputs/tables/LGBM_completo_ensayos.csv")
pesos_red <- read.csv("02_outputs/tables/BLEND_m2_red_pesos.csv")

for (.x in list(train_model, similitud, oof)) {
  if (!identical(.x$property_id, train$property_id)) {
    stop("Los insumos no están en el mismo orden de filas que train. ",
         "Corra de nuevo los scripts 05, 06, 15 y 17.")
  }
}

y <- train$price
similares <- similitud$similitud >= stats::quantile(similitud$similitud, 0.9)
mejor <- kaggle[which.min(kaggle$puntaje_publico), ]
fila_mejor <- function(.tipo) {
  registro[registro$archivo == mejor$archivo & registro$tipo_cv == .tipo, ]
}
if (nrow(fila_mejor("espacial")) == 0) {
  stop("El mejor envío de Kaggle (", mejor$archivo, ") no está en el ",
       "registro de modelos.")
}
pred_espacial <- oof[[grep("_espacial$", names(oof), value = TRUE)]]

# Coordenadas en metros.
fun_xy <- function(.datos) {
  sf::st_coordinates(sf::st_transform(
    sf::st_as_sf(.datos[c("lon", "lat")], coords = c("lon", "lat"),
                 crs = 4326), crs_m))
}
xy_train <- fun_xy(train)
xy_test <- fun_xy(test)

# 1. Chapinero es distinto ---------------------------------------------------
# Cuatro rasgos de la vivienda y su entorno: Chapinero (test), las
# propiedades de train más parecidas a Chapinero (10% con mayor similitud)
# y el resto de train.
fun_rasgos <- function(.datos, .grupo) {
  superficie <- dplyr::coalesce(.datos$surface_covered, .datos$surface_total,
                                .datos$txt_area_m2)
  tibble(
    grupo = .grupo,
    `Apartamentos (%)` = 100 * mean(.datos$property_type == "Apartamento"),
    `Estrato 5 o 6 (%)` = 100 * mean(.datos$estrato_manzana >= 5, na.rm = TRUE),
    `Superficie mediana (m²)` = stats::median(superficie, na.rm = TRUE),
    `Distancia al centro de empleo (km)` =
      stats::median(.datos$dist_centro_empleo_m, na.rm = TRUE) / 1000
  )
}
rasgos <- bind_rows(
  fun_rasgos(train[!similares, ], "Resto de Bogotá (train)"),
  fun_rasgos(train[similares, ], "Train parecido a Chapinero"),
  fun_rasgos(test, "Chapinero (test)")
) |>
  pivot_longer(-grupo, names_to = "rasgo", values_to = "valor") |>
  mutate(grupo = factor(grupo, levels = c("Resto de Bogotá (train)",
                                          "Train parecido a Chapinero",
                                          "Chapinero (test)")),
         rasgo = factor(rasgo, levels = unique(rasgo)))

grafico_rasgos <- rasgos |>
  ggplot(aes(x = valor, y = fct_rev(grupo), color = grupo)) +
  geom_segment(aes(x = 0, xend = valor, yend = fct_rev(grupo)),
               linewidth = 0.8, alpha = 0.5) +
  geom_point(size = 3) +
  geom_text(aes(label = ifelse(valor < 10, fun_coma(valor, 1), fun_coma(valor, 0))),
            hjust = -0.45, size = 3.2, color = texto_1) +
  facet_wrap(~rasgo, scales = "free_x", ncol = 2) +
  scale_color_manual(values = c("Resto de Bogotá (train)" = gris,
                                "Train parecido a Chapinero" = color_1,
                                "Chapinero (test)" = color_2),
                     guide = "none") +
  scale_x_continuous(expand = expansion(mult = c(0, 0.3)), limits = c(0, NA)) +
  labs(title = "Chapinero no es la Bogotá promedio",
       subtitle = "Rasgos de las viviendas de Chapinero frente a las de train",
       x = NULL, y = NULL,
       caption = paste0("Train parecido a Chapinero: 10% con mayor similitud ",
                        "(propensity score del script 06). Estrato: sin dato excluido.")) +
  tema +
  theme(panel.grid.major.y = element_blank(),
        axis.text.y = element_text(color = texto_1, size = 9),
        strip.text = element_text(color = texto_1, face = "bold", size = 9.5),
        panel.spacing = unit(1.2, "lines"))
fun_guardar(grafico_rasgos, "best_chapinero_distinto", 7.5, 4.6)

# 2. Pseudo-Chapineros ----------------------------------------------------------
# Mismas zonas del script 06: k-means con 20 centros sobre las coordenadas
# de train, con la misma semilla.
set.seed(semilla)
zona <- suppressWarnings(stats::kmeans(xy_train, centers = n_zonas,
                                       nstart = 10, iter.max = 50)$cluster)
tamanos <- as.data.frame(table(zona = zona), responseName = "n") |>
  mutate(zona = as.integer(as.character(zona))) |>
  inner_join(zonas, by = "zona")
if (any(tamanos$n != tamanos$propiedades)) {
  stop("Las zonas no coinciden con las del script 06. Corra de nuevo el 06.")
}
pseudo <- zonas |>
  filter(propiedades >= min_tamano_zona) |>
  slice_head(n = n_pseudo) |>
  mutate(etiqueta = paste0(tools::toTitleCase(sub(" /.*", "", nombre)),
                           "\n(similitud ", fun_coma(similitud_media, 2), ")"))

puntos_train <- tibble(x = xy_train[, 1], y = xy_train[, 2], zona = zona) |>
  left_join(pseudo |> select(zona, etiqueta), by = "zona") |>
  mutate(tipo = ifelse(is.na(etiqueta), "Resto de train", "Pseudo-Chapinero"))
puntos_test <- tibble(x = xy_test[, 1], y = xy_test[, 2])
centros <- puntos_train |>
  filter(!is.na(etiqueta)) |>
  group_by(etiqueta) |>
  summarise(x = stats::median(x), y = stats::median(y), .groups = "drop")

# Fondo de los mapas: manzanas de Bogotá (en gris muy claro).
manzanas_bogota <- readRDS("00_data/external/geo_cache/estratificacion_manzanas.rds") |>
  sf::st_transform(crs_m) |>
  sf::st_geometry() |>
  sf::st_simplify(dTolerance = 15)

# El mapa de los pseudo-Chapineros se acerca a Chapinero y a esas zonas.
xy_foco <- rbind(xy_test,
                 as.matrix(filter(puntos_train, !is.na(etiqueta))[c("x", "y")]))
caja <- sf::st_bbox(c(xmin = min(xy_foco[, 1]) - 4500,
                      xmax = max(xy_foco[, 1]) + 2500,
                      ymin = min(xy_foco[, 2]) - 2500,
                      ymax = max(xy_foco[, 2]) + 2500),
                    crs = sf::st_crs(crs_m))
manzanas <- sf::st_crop(manzanas_bogota, caja)

grafico_mapa <- ggplot() +
  geom_sf(data = manzanas, fill = "#e9e8e4", color = NA) +
  geom_point(data = filter(puntos_train, tipo == "Resto de train"),
             aes(x, y), color = gris, size = 0.3, alpha = 0.35) +
  geom_point(data = filter(puntos_train, tipo == "Pseudo-Chapinero"),
             aes(x, y), color = color_3, size = 0.45, alpha = 0.7) +
  geom_point(data = puntos_test, aes(x, y), color = color_2, size = 0.45,
             alpha = 0.6) +
  ggrepel::geom_label_repel(data = centros, aes(x, y, label = etiqueta),
                            size = 3.3, color = color_1, fill = "white",
                            label.size = 0, lineheight = 0.9, nudge_x = -3800,
                            direction = "y", segment.color = color_1,
                            min.segment.length = 0, seed = semilla) +
  annotate("label", x = stats::median(xy_test[, 1]) + 1300,
           y = stats::median(xy_test[, 2]) - 1500, label = "Chapinero\n(test)",
           size = 3.5, color = color_2, fill = "white", alpha = 0.85,
           label.size = 0, fontface = "bold", lineheight = 0.9) +
  coord_sf(datum = NA, expand = FALSE,
           xlim = c(caja["xmin"], caja["xmax"]),
           ylim = c(caja["ymin"], caja["ymax"])) +
  labs(title = "Los tres pseudo-Chapineros",
       subtitle = "Zonas de train más parecidas a Chapinero, apartadas para simular el reto",
       x = NULL, y = NULL,
       caption = "Azul claro: pseudo-Chapineros. Azul oscuro: Chapinero (test).") +
  tema +
  theme(panel.grid = element_blank(), axis.text = element_blank())
fun_guardar(grafico_mapa, "best_pseudo_chapinero_mapa", 5.4, 6.2)

# 2b. Mapa del precio por m² ------------------------------------------------
# Descriptivo: precio mediano por m² de los anuncios de train en hexágonos
# de 600 m. Solo anuncios con superficie observada en la base (cubierta o,
# si falta, total) entre 20 y 500 m², y hexágonos con al menos 5 anuncios.
# El contorno marca los hexágonos con al menos 5 anuncios de Chapinero
# (test), que no tienen precio. No se estima un efecto causal.
lado_hexagono <- 600
min_anuncios_hexagono <- 5
puntos_m2 <- tibble(x = xy_train[, 1], y = xy_train[, 2],
                    area = dplyr::coalesce(train$surface_covered,
                                           train$surface_total),
                    precio_m2 = train$price / area / 1e6) |>
  filter(between(area, 20, 500)) |>
  sf::st_as_sf(coords = c("x", "y"), crs = crs_m)
hexagonos <- sf::st_make_grid(sf::st_as_sfc(sf::st_bbox(puntos_m2)),
                              cellsize = lado_hexagono, square = FALSE) |>
  sf::st_sf() |>
  mutate(id = row_number())
precio_hexagono <- sf::st_join(puntos_m2, hexagonos) |>
  sf::st_drop_geometry() |>
  group_by(id) |>
  summarise(precio_m2 = stats::median(precio_m2), n = n(), .groups = "drop") |>
  filter(n >= min_anuncios_hexagono)
hexagonos_precio <- inner_join(hexagonos, precio_hexagono, by = "id")
anuncios_test_hexagono <- sf::st_join(
  sf::st_as_sf(tibble(x = xy_test[, 1], y = xy_test[, 2]),
               coords = c("x", "y"), crs = crs_m), hexagonos) |>
  sf::st_drop_geometry() |>
  count(id)
contorno_chapinero <- hexagonos |>
  filter(id %in% anuncios_test_hexagono$id[
    anuncios_test_hexagono$n >= min_anuncios_hexagono]) |>
  sf::st_union()
# Anuncios de train dentro del contorno de Chapinero (cifra del deck).
en_zona_chapinero <- lengths(sf::st_intersects(puntos_m2, contorno_chapinero)) > 0
tope_m2 <- 9
caja_m2 <- sf::st_bbox(hexagonos_precio)

grafico_mapa_m2 <- ggplot() +
  geom_sf(data = sf::st_crop(manzanas_bogota, caja_m2), fill = "#e9e8e4",
          color = NA) +
  geom_sf(data = hexagonos_precio, aes(fill = pmin(precio_m2, tope_m2)),
          color = "white", linewidth = 0.1) +
  geom_sf(data = contorno_chapinero, fill = NA, color = texto_1,
          linewidth = 0.9) +
  scale_fill_gradientn(colours = c("#c9f0e6", "#5cc6b4", "#2a78d6",
                                   "#3b3fa0", "#5b2a86"),
                       limits = c(NA, tope_m2), breaks = c(3, 5, 7, 9),
                       labels = c("3", "5", "7", "9 o más"),
                       name = "Millones\npor m²") +
  coord_sf(datum = NA, expand = FALSE,
           xlim = c(caja_m2["xmin"], caja_m2["xmax"]),
           ylim = c(caja_m2["ymin"], caja_m2["ymax"])) +
  labs(title = "Chapinero está entre las zonas más caras de Bogotá",
       subtitle = paste0("Precio mediano por m² de los anuncios de train, en ",
                         "hexágonos de ", lado_hexagono, " m"),
       x = NULL, y = NULL,
       caption = paste0("Contorno negro: hexágonos con anuncios de Chapinero ",
                        "(test, sin precio).\nHexágonos con al menos ",
                        min_anuncios_hexagono, " anuncios con superficie ",
                        "entre 20 y 500 m².")) +
  tema +
  theme(panel.grid = element_blank(), axis.text = element_blank(),
        legend.position = "right",
        legend.title = element_text(color = texto_2, size = 9))
fun_guardar(grafico_mapa_m2, "best_mapa_precio_m2", 5.4, 6.2)

# Las tres validaciones cruzadas que guardó el script 06, en el mapa. Cada
# color es un fold. Debajo de cada mapa: cuánto se alejó el error que
# promete ese esquema del error real en los pseudo-Chapineros.
fun_fold <- function(.folds) {
  fold <- integer(nrow(train))
  for (f in seq_along(.folds)) fold[.folds[[f]]] <- f
  fold
}
fun_brecha <- function(.esquema, .col) seleccion_cv[[.col]][seleccion_cv$esquema == .esquema]
esquemas_cv <- tibble(
  esquema = c("Aleatoria", "k-means 5 regiones", "k-means 20 regiones"),
  nombre = c("Aleatoria", "Espacial: 5 regiones", "Espacial: 20 regiones"),
  folds = list(cv_folds$aleatoria, cv_folds$espacial_k5, cv_folds$espacial)
) |>
  mutate(brecha = sapply(esquema, fun_brecha, .col = "brecha_media_pct"),
         titulo = factor(nombre, levels = nombre))
puntos_folds <- esquemas_cv |>
  rowwise() |>
  reframe(titulo = titulo, x = xy_train[, 1], y = xy_train[, 2],
          fold = factor(fun_fold(folds)))
caja_ciudad <- sf::st_bbox(c(xmin = min(xy_train[, 1]) - 500,
                             xmax = max(xy_train[, 1]) + 500,
                             ymin = min(xy_train[, 2]) - 500,
                             ymax = max(xy_train[, 2]) + 500),
                           crs = sf::st_crs(crs_m))
grafico_esquemas <- ggplot() +
  geom_sf(data = sf::st_crop(manzanas_bogota, caja_ciudad), fill = "#e9e8e4",
          color = NA) +
  geom_point(data = puntos_folds, aes(x, y, color = fold), size = 0.12,
             alpha = 0.6) +
  facet_wrap(~titulo, nrow = 1) +
  scale_color_manual(values = c("#2a78d6", "#eb6834", "#1baf7a", "#8c6bb1", "#f2b134"),
                     guide = "none") +
  coord_sf(datum = NA, expand = FALSE,
           xlim = c(caja_ciudad["xmin"], caja_ciudad["xmax"]),
           ylim = c(caja_ciudad["ymin"], caja_ciudad["ymax"])) +
  labs(x = NULL, y = NULL, caption = "Cada color es uno de los 5 folds.") +
  tema +
  theme(panel.grid = element_blank(), axis.text = element_blank(),
        strip.text = element_text(color = texto_1, face = "bold", size = 13,
                                  lineheight = 1.1),
        plot.caption = element_text(size = 12, color = texto_2, hjust = 0.5),
        panel.spacing = unit(1.5, "lines"))
fun_guardar(grafico_esquemas, "best_cv_esquemas", 10, 4.6)

# Los 8 esquemas que se compararon en el script 06: cuánto se aleja, en
# promedio y con signo, el error que promete cada uno del error real en los
# pseudo-Chapineros. Se eligió el de menor brecha absoluta.
nombres_esquema <- c(
  "Aleatoria" = "Aleatoria",
  "k-means 5 regiones" = "5 regiones (k-means)",
  "k-means 20 regiones" = "20 regiones (k-means)",
  "Bloques de 300 m" = "Bloques de 300 m", "Bloques de 500 m" = "Bloques de 500 m",
  "Bloques de 700 m" = "Bloques de 700 m", "Bloques de 1 km" = "Bloques de 1 km",
  "Bloques de 2 km" = "Bloques de 2 km")
seleccion <- seleccion_cv |>
  mutate(nombre = dplyr::coalesce(unname(nombres_esquema[esquema]), esquema),
         nombre = fct_reorder(nombre, brecha_media_pct),
         tipo = case_when(elegido ~ "Elegida",
                          esquema == "Aleatoria" ~ "Aleatoria",
                          TRUE ~ "Otras espaciales"))
grafico_seleccion <- seleccion |>
  ggplot(aes(x = brecha_media_pct, y = nombre, fill = tipo)) +
  geom_col(width = 0.65) +
  geom_vline(xintercept = 0, color = texto_2) +
  geom_text(aes(label = paste0(ifelse(brecha_media_pct > 0, "+", ""),
                               fun_coma(brecha_media_pct, 1), "%"),
                hjust = ifelse(brecha_media_pct < 0, 1.15, -0.15)),
            size = 3.6, color = texto_1) +
  scale_fill_manual(values = c("Elegida" = color_2, "Otras espaciales" = color_3,
                               "Aleatoria" = gris), name = NULL) +
  scale_x_continuous(labels = function(x) paste0(x, "%"),
                     expand = expansion(mult = 0.18)) +
  labs(title = "Probamos 8 formas de validar",
       subtitle = "Error que promete cada validación frente al error real en los pseudo-Chapineros",
       x = "Diferencia promedio (negativo = promete menos error del real)", y = NULL,
       caption = paste0("Promedio en 3 pseudo-Chapineros y 2 modelos (Random Forest y ",
                        "regresión lineal). Error real promedio: ",
                        fun_coma(unique(round(seleccion_cv$mae_verdadero, 6)), 0),
                        " millones.")) +
  tema +
  theme(panel.grid.major.y = element_blank(),
        axis.text.y = element_text(color = texto_1, size = 10.5))
fun_guardar(grafico_seleccion, "best_seleccion_cv", 7.5, 4.6)

# 3. Cascada de envíos a Kaggle ---------------------------------------------------
# Los pasos que llevaron al mejor modelo: cada uno agrega una idea.
pasos <- tibble(
  archivo = c("LM_base", "LM_completo",
              "LGBM_completo_lr_0_017_hojas_66_prof_7_minhoja_23_rondas_4123",
              "LGBM_completo_m2_lr_0_017_hojas_66_rondas_2689",
              "LGBM_completo_m2_vecinos_k10_lr_0_017_hojas_66_rondas_2689",
              mejor$archivo),
  paso = c("Regresión lineal\n17 variables", "Regresión lineal\n174 variables",
           "LightGBM\n174 variables", "Predecir el\nprecio por m²",
           "Agregar el precio\nde los vecinos", "Mezclar con la\nred neuronal")
) |>
  left_join(kaggle |> select(archivo, puntaje_publico), by = "archivo") |>
  mutate(mae = puntaje_publico / 1e6,
         paso = factor(paso, levels = paso),
         mejora = c(NA, -diff(mae)),
         grupo = c("Referencia", "Referencia", "Modelo", "Modelo", "Modelo",
                   "Final"))
if (anyNA(pasos$mae)) stop("Falta algún paso de la cascada en kaggle_scores.csv.")

altura <- max(pasos$mae) * 1.16
grafico_cascada <- pasos |>
  ggplot(aes(x = paso, y = mae, fill = grupo)) +
  geom_col(width = 0.62) +
  geom_vline(xintercept = 2.5, linetype = "dashed", color = texto_2) +
  geom_text(aes(label = fun_coma(mae, 1)), vjust = -0.5, size = 3.4,
            color = texto_1, fontface = "bold") +
  geom_text(data = filter(pasos, !is.na(mejora)),
            aes(y = mae / 2, label = paste0("-", fun_coma(mejora, 1))),
            size = 3, color = "white") +
  annotate("text", x = 1.5, y = altura, label = "Referencia: regresión lineal",
           size = 3.4, color = texto_2, fontface = "bold") +
  annotate("text", x = 4.5, y = altura,
           label = "Del primer LightGBM al mejor modelo", size = 3.4,
           color = color_1, fontface = "bold") +
  scale_fill_manual(values = c("Referencia" = "#d3d1cb", "Modelo" = "#8fb4e6",
                               "Final" = color_1), guide = "none") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.04)),
                     limits = c(0, altura * 1.03)) +
  labs(title = "MAE en Kaggle de cada paso hacia el mejor modelo",
       subtitle = "Millones de pesos; dentro de cada barra, la mejora frente al paso anterior",
       x = NULL, y = "MAE (millones)",
       caption = "Envíos del equipo a Kaggle (Chapinero, tabla pública).") +
  tema +
  theme(panel.grid.major.x = element_blank(),
        axis.text.x = element_text(color = texto_1, size = 8.8, lineheight = 0.9))
fun_guardar(grafico_cascada, "best_cascada_kaggle", 7.5, 4.6)

# 4. Importancia de las variables --------------------------------------------------
# Ganancia (reducción del error) que aporta cada variable en los árboles de
# los dos modelos finales del híbrido (script 15). No son efectos causales.
etiquetas <- c(
  vec_precio_m2 = "Precio por m² de los vecinos",
  vec_precio = "Precio de los vecinos",
  vec_dist_m = "Distancia a los vecinos",
  surface_covered_completa = "Superficie cubierta",
  surface_total_completa = "Superficie total",
  bathrooms_completa = "Baños",
  bedrooms = "Habitaciones",
  rooms = "Ambientes",
  dist_centro_empleo_m = "Distancia al centro de empleo",
  dist_centro_comercial_m = "Distancia a centro comercial",
  dist_parque_m = "Distancia a parque",
  dist_via_principal_m = "Distancia a vía principal",
  dist_transmilenio_m = "Distancia a TransMilenio",
  dist_universidad_m = "Distancia a universidad",
  pct_zona_verde_500m = "Zona verde a 500 m (%)",
  pendiente_terreno_pct = "Pendiente del terreno",
  num_restaurantes_cafes_500m = "Restaurantes y cafés a 500 m",
  estrato_manzana = "Estrato de la manzana",
  lat = "Latitud (norte-sur)", lon = "Longitud (oriente-occidente)",
  txt_parqueaderos = "Parqueaderos (texto)", txt_piso = "Piso (texto)",
  property_type = "Casa o apartamento", month = "Mes", year = "Año"
)
fun_bloque <- function(.v) {
  dplyr::case_when(
    startsWith(.v, "vec_") ~ "Precio de los vecinos",
    .v %in% c("surface_covered_completa", "surface_total_completa",
              "bathrooms_completa", "bedrooms", "rooms", "property_type",
              "month", "year") ~ "Vivienda",
    startsWith(.v, "raiz_") | startsWith(.v, "txt_") |
      .v == "superficie_desde_texto" ~ "Texto del anuncio",
    startsWith(.v, "falta_") | startsWith(.v, "flag_") ~ "Calidad del dato",
    TRUE ~ "Ubicación y entorno"
  )
}
colores_bloque <- c("Precio de los vecinos" = color_2, "Vivienda" = color_1,
                    "Ubicación y entorno" = color_3, "Texto del anuncio" = color_4,
                    "Calidad del dato" = gris)
importancia <- importancia |>
  mutate(bloque = factor(fun_bloque(Feature), levels = names(colores_bloque)),
         nombre = dplyr::coalesce(unname(etiquetas[Feature]), Feature),
         modelo = ifelse(modelo == "m2", "Precio por m²", "Precio total"))

top_m2 <- importancia |>
  filter(modelo == "Precio por m²") |>
  slice_max(Gain, n = 12) |>
  mutate(nombre = fct_reorder(nombre, Gain))
panel_top <- top_m2 |>
  ggplot(aes(x = 100 * Gain, y = nombre, fill = bloque)) +
  geom_col(width = 0.7) +
  geom_text(aes(label = fun_coma(100 * Gain, 1)), hjust = -0.2, size = 2.9,
            color = texto_1) +
  scale_fill_manual(values = colores_bloque, guide = "none") +
  scale_x_continuous(expand = expansion(mult = c(0, 0.15))) +
  labs(subtitle = "12 principales, modelo de precio por m²",
       x = "Ganancia (%)", y = NULL) +
  tema +
  theme(panel.grid.major.y = element_blank(),
        axis.text.y = element_text(color = texto_1, size = 8.5),
        plot.subtitle = element_text(margin = margin(b = 6)))
por_bloque <- importancia |>
  group_by(modelo, bloque) |>
  summarise(gain = 100 * sum(Gain), .groups = "drop")
panel_bloque <- por_bloque |>
  ggplot(aes(x = gain, y = fct_rev(bloque), fill = bloque, alpha = modelo)) +
  geom_col(position = position_dodge(width = 0.75), width = 0.7) +
  geom_text(aes(label = fun_coma(gain, 0), group = modelo),
            position = position_dodge(width = 0.75), hjust = -0.25, size = 2.8,
            color = texto_1, alpha = 1) +
  scale_fill_manual(values = colores_bloque, guide = "none") +
  scale_alpha_manual(values = c("Precio por m²" = 1, "Precio total" = 0.45),
                     name = NULL) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.18))) +
  labs(subtitle = "Por bloque de variables, los dos modelos",
       x = "Ganancia (%)", y = NULL) +
  tema +
  theme(panel.grid.major.y = element_blank(),
        axis.text.y = element_text(color = texto_1, size = 8.5),
        plot.subtitle = element_text(margin = margin(b = 6)),
        legend.key.size = unit(0.35, "cm"))
grafico_importancia <- (panel_top | panel_bloque) +
  plot_layout(widths = c(1.15, 1)) +
  plot_annotation(
    title = "El precio de los vecinos es lo que más pesa",
    caption = paste0("Ganancia: reducción del error que aporta cada variable en los ",
                     "árboles del modelo final (suma 100%). No son efectos causales."),
    theme = tema)
fun_guardar(grafico_importancia, "best_importancia", 8.6, 4.6)

# 5. Variables de texto ------------------------------------------------------------
# Las 12 variables del anuncio que más aportan, en promedio entre los dos
# modelos del híbrido. Las raíces entran al modelo si aparecen en al menos
# el 5% de los anuncios (script 04).
vocabulario <- readRDS("03_temp/vocabulario_texto.rds")
palabras <- c(
  balcon = "Balcón", terraz = "Terraza", abiert = "Abierto",
  gimnasi = "Gimnasio", bbq = "BBQ", remodel = "Remodelado", cas = "Casa",
  servici = "Servicio", buen = "Buen, buena",
  deposit = "Depósito", estudi = "Estudio", comedor = "Comedor",
  cocin = "Cocina", vist = "Vista", chimene = "Chimenea",
  integral = "Integral (cocina)", exclus = "Exclusivo", cubiert = "Cubierto",
  garaj = "Garaje", cerc = "Cerca", piscin = "Piscina", ascensor = "Ascensor",
  conjunt = "Conjunto", porter = "Portería", vigil = "Vigilancia"
)
texto <- importancia |>
  filter(bloque == "Texto del anuncio") |>
  group_by(Feature) |>
  summarise(gain = 100 * sum(Gain) / 2, .groups = "drop") |>
  mutate(raiz = sub("^raiz_", "", Feature),
         pct_anuncios = vocabulario$frecuencia$pct_anuncios[
           match(raiz, vocabulario$frecuencia$raiz)],
         nombre = case_when(
           Feature == "txt_parqueaderos" ~ "N.º de parqueaderos",
           Feature == "txt_piso" ~ "Piso",
           TRUE ~ dplyr::coalesce(unname(palabras[raiz]), raiz)),
         tipo = ifelse(startsWith(Feature, "txt_"), "Cantidad extraída",
                       "Palabra (raíz)"),
         etiqueta = ifelse(is.na(pct_anuncios), fun_coma(gain, 2),
                           paste0(fun_coma(gain, 2), "  (en ",
                                  fun_coma(pct_anuncios, 0), "% de anuncios)")))
gain_texto_total <- sum(texto$gain)
grafico_texto <- texto |>
  slice_max(gain, n = 12) |>
  mutate(nombre = fct_reorder(nombre, gain)) |>
  ggplot(aes(x = gain, y = nombre, fill = tipo)) +
  geom_col(width = 0.7) +
  geom_text(aes(label = etiqueta), hjust = -0.08, size = 3.5, color = texto_1) +
  scale_fill_manual(values = c("Palabra (raíz)" = color_1,
                               "Cantidad extraída" = color_3), name = NULL) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.45))) +
  labs(title = "Las 12 variables de texto más importantes",
       subtitle = "Ganancia de cada variable (% del total del modelo)",
       x = "Ganancia (%)", y = NULL,
       caption = paste0("Promedio de los dos modelos finales. Entre paréntesis: anuncios ",
                        "que usan la palabra. En total, el texto aporta ",
                        fun_coma(gain_texto_total, 0), "% de la ganancia.")) +
  tema +
  theme(panel.grid.major.y = element_blank(),
        axis.text.y = element_text(color = texto_1, size = 11))
fun_guardar(grafico_texto, "best_texto_importancia", 7.5, 4.6)

# 6. Lo que promete la CV frente a Kaggle --------------------------------------
# Error del mejor modelo con cada CV, en las propiedades parecidas a
# Chapinero y en Kaggle.
promesas <- tibble(
  medida = c("CV aleatoria", "CV espacial", "CV espacial,\nparecidas a Chapinero",
             "Kaggle (Chapinero)"),
  mae = c(fila_mejor("aleatoria")$mae, fila_mejor("espacial")$mae,
          fila_mejor("espacial")$mae_similares_chapinero,
          mejor$puntaje_publico) / 1e6,
  tipo = c("CV", "CV", "CV", "Kaggle")
) |>
  mutate(medida = factor(medida, levels = rev(medida)))
grafico_promesa <- promesas |>
  ggplot(aes(x = mae, y = medida, fill = tipo)) +
  geom_col(width = 0.62) +
  geom_text(aes(label = fun_coma(mae, 1)), hjust = -0.2, size = 3.6,
            color = texto_1, fontface = "bold") +
  scale_fill_manual(values = c("CV" = color_3, "Kaggle" = color_2), guide = "none") +
  scale_x_continuous(expand = expansion(mult = c(0, 0.15))) +
  labs(title = "Ninguna validación cruzada anticipa el error en Chapinero",
       subtitle = "MAE del mejor modelo (millones de pesos)", x = NULL, y = NULL,
       caption = paste0("Predicciones fuera de muestra en train. Parecidas a ",
                        "Chapinero: 10% de train con mayor similitud.")) +
  tema +
  theme(panel.grid.major.y = element_blank(),
        axis.text.y = element_text(color = texto_1, size = 10))
fun_guardar(grafico_promesa, "best_cv_frente_kaggle", 7, 4.2)

# 7. Sesgo por rango de precio ----------------------------------------------------
# Error medio (predicho - real) por decil del precio real, con las
# predicciones fuera de muestra de la CV espacial: el modelo "jala" los
# precios hacia el centro (regresión a la media).
cortes_decil <- stats::quantile(y, seq(0, 1, 0.1))
deciles <- tibble(precio = y, pred = pred_espacial, similar = similares) |>
  mutate(decil = cut(precio, cortes_decil, include.lowest = TRUE, labels = FALSE))
fun_sesgo_decil <- function(.datos, .grupo) {
  .datos |>
    group_by(decil) |>
    summarise(sesgo = mean(pred - precio) / 1e6, n = n(), .groups = "drop") |>
    mutate(grupo = .grupo)
}
rangos <- paste0(fun_coma(head(cortes_decil, -1) / 1e6, 0), "–",
                 fun_coma(cortes_decil[-1] / 1e6, 0))
sesgo_deciles <- bind_rows(
  fun_sesgo_decil(deciles, "Todo train"),
  fun_sesgo_decil(filter(deciles, similar), "Parecidas a Chapinero")
) |>
  mutate(rango = factor(rangos[decil], levels = rangos),
         grupo = factor(grupo, levels = c("Todo train", "Parecidas a Chapinero")))
grafico_deciles <- sesgo_deciles |>
  ggplot(aes(x = rango, y = sesgo, fill = grupo)) +
  geom_col(width = 0.75, position = position_dodge(width = 0.8)) +
  geom_hline(yintercept = 0, color = texto_2) +
  geom_text(aes(label = fun_coma(sesgo, 0), vjust = ifelse(sesgo > 0, -0.5, 1.4)),
            position = position_dodge(width = 0.8), size = 2.6, color = texto_1) +
  scale_fill_manual(values = c("Todo train" = color_3,
                               "Parecidas a Chapinero" = color_2), name = NULL) +
  scale_y_continuous(expand = expansion(mult = 0.12)) +
  labs(title = "Sobrestima las viviendas baratas y subestima las caras",
       subtitle = "Error medio (predicho - real) por decil del precio real, millones de pesos",
       x = "Precio real (millones de pesos)", y = "Error medio (millones)",
       caption = paste0("Mejor modelo, predicciones fuera de muestra con la CV espacial. ",
                        "Parecidas a Chapinero: 10% de train con mayor similitud.")) +
  tema +
  theme(panel.grid.major.x = element_blank(),
        axis.text.x = element_text(size = 8, angle = 30, hjust = 1))
fun_guardar(grafico_deciles, "best_sesgo_deciles", 7.5, 4.6)

# 8. Ejemplo: cómo valora el modelo una vivienda de Chapinero -----------------
# Un apartamento típico de test con superficie real (70 a 90 m²): el de
# predicción final más cercana a la mediana de ese grupo. Se muestran las
# piezas del modelo con los envíos guardados (15, 12 y 17).
fun_leer_envio <- function(.archivo) {
  #' Leer un envío de Kaggle en el orden de test_features.rds
  envio <- read.csv(paste0("02_outputs/predictions/", .archivo, ".csv"))
  envio$price[match(test$property_id, envio$property_id)]
}
archivo_red <- kaggle$archivo[kaggle$algoritmo == "Red neuronal"]
pred_lgbm_test <- fun_leer_envio(pasos$archivo[5])
pred_red_test <- fun_leer_envio(archivo_red)
pred_final_test <- fun_leer_envio(mejor$archivo)
test_model <- readRDS("03_temp/test_model.rds")
if (!identical(test_model$property_id, test$property_id)) {
  stop("test_model.rds y test_features.rds no tienen el mismo orden de filas.")
}
# Superficie real (no imputada), con la misma regla del 13 y el 15.
fun_superficie_real <- function(.datos) {
  dplyr::case_when(
    .datos$falta_surface_covered_completa == 0 ~ .datos$surface_covered_completa,
    .datos$falta_surface_total_completa == 0 ~ .datos$surface_total_completa,
    TRUE ~ NA_real_
  )
}
superficie_test <- fun_superficie_real(test_model)
superficie_train <- fun_superficie_real(train_model)
superficie_real_test <- !is.na(superficie_test)

candidatas <- which(superficie_real_test & test$property_type == "Apartamento" &
                      superficie_test >= 70 & superficie_test <= 90)
ej <- candidatas[which.min(abs(pred_final_test[candidatas] -
                                 stats::median(pred_final_test[candidatas])))]
grupo_anuncio <- fun_grupo_anuncio(train)
# Vecinos del ejemplo, como en el 15: precio por m² de las 10 viviendas de
# train más cercanas con superficie real.
filas_m2 <- which(!is.na(superficie_train))
vecinos_ej <- fun_vecinos(xy_train[filas_m2, ], (y / superficie_train)[filas_m2],
                          grupo_anuncio[filas_m2], xy_test[ej, , drop = FALSE],
                          fun_grupo_anuncio(test[ej, ]))
fun_millones <- function(.x, .d = 0) fun_coma(.x / 1e6, .d)

# 9. Cifras del deck ------------------------------------------------------------
mejor_ensayo <- ensayos[which.min(ensayos$mae), ]
fun_mae_registro <- function(.archivo, .tipo) {
  registro$mae[registro$archivo == .archivo & registro$tipo_cv == .tipo] / 1e6
}
peso_red <- pesos_red$peso_red[which.min(pesos_red$mae_similares)]
precio_medio_similares <- mean(y[similares])
cifras <- c(
  MAEKaggle = fun_coma(mejor$puntaje_publico / 1e6, 1),
  MAEBase = fun_coma(pasos$mae[1], 1),
  MAELgbm = fun_coma(pasos$mae[3], 1),
  MAECVEspacial = fun_coma(fila_mejor("espacial")$mae / 1e6, 1),
  MAECVAleatoria = fun_coma(fila_mejor("aleatoria")$mae / 1e6, 1),
  MAESimilares = fun_coma(fila_mejor("espacial")$mae_similares_chapinero / 1e6, 1),
  SesgoSimilares = fun_coma(fila_mejor("espacial")$sesgo_similares_chapinero / 1e6, 1),
  ErrorPct = fun_pct(mejor$puntaje_publico / precio_medio_similares),
  PrecioMedioSimilares = fun_entero(precio_medio_similares / 1e6),
  PrecioMedioResto = fun_entero(mean(y[!similares]) / 1e6),
  # Mapa del precio por m² (sección 2b).
  MdosBogota = fun_coma(stats::median(puntos_m2$precio_m2), 1),
  MdosZonaChapinero = fun_coma(stats::median(puntos_m2$precio_m2[en_zona_chapinero]), 1),
  NZonaChapinero = fun_entero(sum(en_zona_chapinero)),
  LadoHexagono = lado_hexagono,
  PctAptoTrain = fun_pct(mean(train$property_type == "Apartamento")),
  PctAptoTest = fun_pct(mean(test$property_type == "Apartamento")),
  # Anuncios sin ninguna de las dos superficies en las columnas originales.
  PctSinSupTrain = fun_pct(mean(is.na(train$surface_covered) & is.na(train$surface_total))),
  PctSinSupTest = fun_pct(mean(is.na(test$surface_covered) & is.na(test$surface_total))),
  NTrain = fun_entero(nrow(train)),
  NTest = fun_entero(nrow(test)),
  PctSuperficieReal = fun_pct(mean(superficie_real_test)),
  PctSuperficieImputada = fun_pct(1 - mean(superficie_real_test)),
  NEnsayos = nrow(ensayos),
  TasaAprendizaje = fun_coma(exp(mejor_ensayo$log_lr), 3),
  Hojas = mejor_ensayo$num_leaves,
  Profundidad = mejor_ensayo$max_depth,
  MinHoja = mejor_ensayo$min_data_in_leaf,
  Rondas = fun_entero(mejor_ensayo$rondas),
  PesoRed = fun_pct(peso_red),
  PesoLgbm = fun_pct(1 - peso_red),
  BrechaElegido = fun_coma(fun_brecha("k-means 20 regiones", "brecha_absoluta_pct"), 1),
  BrechaCinco = fun_coma(fun_brecha("k-means 5 regiones", "brecha_absoluta_pct"), 1),
  BrechaAleatoria = fun_coma(fun_brecha("Aleatoria", "brecha_absoluta_pct"), 1),
  NEsquemas = nrow(seleccion_cv),
  BloquesMin = fun_coma(min(abs(seleccion_cv$brecha_media_pct[startsWith(seleccion_cv$esquema, "Bloques")])), 0),
  BloquesMax = fun_coma(max(abs(seleccion_cv$brecha_media_pct[startsWith(seleccion_cv$esquema, "Bloques")])), 0),
  BrechaBloqueMejor = fun_coma(min(seleccion_cv$brecha_absoluta_pct[startsWith(seleccion_cv$esquema, "Bloques")]), 1),
  # Cuánto se alejó, en promedio y con signo, el error prometido del real.
  PrometeAleatoria = fun_coma(abs(fun_brecha("Aleatoria", "brecha_media_pct")), 0),
  PrometeCinco = fun_coma(abs(fun_brecha("k-means 5 regiones", "brecha_media_pct")), 0),
  PrometeVeinte = fun_coma(abs(fun_brecha("k-means 20 regiones", "brecha_media_pct")), 0),
  GainUbicacion = fun_coma(sum(por_bloque$gain[por_bloque$modelo == "Precio por m²" &
                                                 por_bloque$bloque %in% c("Ubicación y entorno", "Precio de los vecinos")]), 0),
  GainVecinos = fun_coma(sum(por_bloque$gain[por_bloque$modelo == "Precio por m²" &
                                               por_bloque$bloque == "Precio de los vecinos"]), 0),
  GainTexto = fun_coma(gain_texto_total, 0),
  GainVivienda = fun_coma(sum(por_bloque$gain[por_bloque$modelo == "Precio por m²" &
                                                por_bloque$bloque == "Vivienda"]), 0),
  MejoraPctBase = fun_pct(1 - mejor$puntaje_publico / 1e6 / pasos$mae[1]),
  NRaices = length(vocabulario$completo),
  MejoraRed = fun_coma(pasos$mae[5] - pasos$mae[6], 1),
  # Más texto no fue mejor: LightGBM con todo el TF-IDF (conjunto ampliado).
  KaggleAmpliado = fun_coma(kaggle$puntaje_publico[
    kaggle$archivo == "LGBM_ampliado_lr_0_017_hojas_66_prof_7_minhoja_23_rondas_4994"] / 1e6, 1),
  MejoraLMTexto = fun_coma(pasos$mae[1] - pasos$mae[2], 0),
  # Qué dijo cada CV sobre agregar los vecinos (13 frente a 15) y qué
  # dijo Kaggle.
  CVEspSinVec = fun_coma(fun_mae_registro(pasos$archivo[4], "espacial"), 1),
  CVEspConVec = fun_coma(fun_mae_registro(pasos$archivo[5], "espacial"), 1),
  CVAleSinVec = fun_coma(fun_mae_registro(pasos$archivo[4], "aleatoria"), 1),
  CVAleConVec = fun_coma(fun_mae_registro(pasos$archivo[5], "aleatoria"), 1),
  KaggleSinVec = fun_coma(pasos$mae[4], 1),
  KaggleConVec = fun_coma(pasos$mae[5], 1),
  SesgoCaras = fun_coma(-sesgo_deciles$sesgo[sesgo_deciles$grupo == "Todo train"][10], 0),
  EjSuperficie = fun_entero(superficie_test[ej]),
  EjHabitaciones = test$bedrooms[ej],
  EjBanos = test_model$bathrooms_completa[ej],
  EjEstrato = test$estrato_manzana[ej],
  EjVecinosMdos = fun_millones(vecinos_ej$mediana, 1),
  EjVecinosDist = fun_entero(vecinos_ej$distancia),
  EjLgbmMdos = fun_millones(pred_lgbm_test[ej] / superficie_test[ej], 1),
  EjLgbm = fun_millones(pred_lgbm_test[ej]),
  EjRed = fun_millones(pred_red_test[ej]),
  EjFinal = fun_millones(pred_final_test[ej]),
  SesgoBaratas = fun_coma(sesgo_deciles$sesgo[sesgo_deciles$grupo == "Todo train"][1], 0)
)
writeLines(paste0("\\newcommand{\\", names(cifras), "}{", cifras, "}"),
           "02_outputs/tables/best_datos.tex")
print(data.frame(cifra = names(cifras), valor = cifras), row.names = FALSE)

# Limpiamos el entorno: borramos solo los objetos que creó este script.
rm(list = setdiff(ls(), objetos_previos))
