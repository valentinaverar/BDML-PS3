##########################################################
# 18_compare_algorithms.R
#
# Objetivo: producir las figuras y cifras del deck de comparación de
# algoritmos (compare_equipo_03): ranking de las seis familias con la
# misma información, cuánto engaña la validación aleatoria a cada una,
# qué bajó más el error (información, algoritmo o hiperparámetros), si más
# variables de texto ayudan, y por qué mezclar árboles con la red.
#
# Etapa del Problem Set: Etapa 2 (entrenar y comparar algoritmos) y
# Etapa 3 (validación espacial frente a la aleatoria).
# Versión inicial del deck y del script: Andrés Silva.
# Insumos:  02_outputs/tables/registro_modelos.csv (scripts 07 a 12),
#           LGBM_completo_ensayos.csv (script 11), BLEND_m2_red_pesos.csv
#           (script 16) y *_busqueda.csv (scripts 08 a 12);
#           00_data/kaggle_scores.csv.
# Salidas:  02_outputs/figures/compare_*.pdf y
#           02_outputs/tables/compare_datos.tex (cifras del deck como
#           comandos de LaTeX, por ejemplo \CompEspLGBM).
#
# Decisiones:
#   - No entrena modelos: solo lee resultados guardados.
#   - Comparación justa entre familias: las mismas 174 variables (conjunto
#     completo), los mismos folds y el mismo objetivo. El ranking principal
#     usa la CV espacial de 20 regiones con el precio total (la única
#     variable a predecir que tienen las seis familias); el precio por m²
#     se muestra aparte para las tres familias de árboles.
#   - Kaggle no sirve para el ranking completo: no todas las variantes se
#     subieron y las que se subieron no siempre tienen el mismo objetivo.
#     Se reporta como referencia.
#   - Las mejoras posteriores del mejor modelo (vecinos, 177 variables, y
#     la mezcla con la red) se agregaron después de elegir el algoritmo;
#     se muestran en el deck best_equipo_03.
#   - Mismo estilo de figuras que los scripts 19 y 20.
##########################################################

# Objetos que ya existían en el entorno antes de correr este script. Al
# final se borra solo lo que este script creó, sin tocar lo demás.
objetos_previos <- ls()

# Paquetes -----------------------------------------------------------------
if (!require(pacman)) install.packages("pacman")
pacman::p_load(dplyr, tidyr, ggplot2, patchwork, forcats)

# Estilo de las figuras (el mismo de los scripts 19 y 20) --------------------
color_1 <- "#2a78d6"        # Azul.
color_2 <- "#0f3570"        # Azul oscuro.
color_3 <- "#7fb0ec"        # Azul claro.
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
registro <- read.csv("02_outputs/tables/registro_modelos.csv")
kaggle <- read.csv("00_data/kaggle_scores.csv")
ensayos <- read.csv("02_outputs/tables/LGBM_completo_ensayos.csv")
pesos_red <- read.csv("02_outputs/tables/BLEND_m2_red_pesos.csv")
if (!"archivo" %in% names(kaggle)) {
  stop("kaggle_scores.csv no tiene la columna 'archivo'. Revise que se haya ",
       "guardado como CSV UTF-8 sin BOM (no con Excel).")
}

fun_variante <- function(.algoritmo, .conjunto = "completo") {
  #' Mejor variante de un algoritmo y conjunto en el registro
  #'
  #' Si hay varias variantes (p. ej. dos búsquedas de hiperparámetros), se
  #' toma la de menor MAE en la CV espacial, la misma regla con la que se
  #' eligieron los hiperparámetros.
  #'
  #' @returns \code{list} con archivo, MAE espacial y aleatoria, MAE y sesgo
  #'   en las parecidas a Chapinero (millones) y puntaje de Kaggle (o NA).
  filas <- registro[registro$algoritmo == .algoritmo &
                      registro$conjunto == .conjunto, ]
  if (nrow(filas) == 0) {
    stop("No hay resultados de '", .algoritmo, "' (", .conjunto, ") en el ",
         "registro. Corra primero el script del modelo.")
  }
  espacial <- filas[filas$tipo_cv == "espacial", ]
  archivo <- espacial$archivo[which.min(espacial$mae)]
  fila_esp <- espacial[espacial$archivo == archivo, ]
  fila_ale <- filas[filas$archivo == archivo & filas$tipo_cv == "aleatoria", ]
  puntaje <- kaggle$puntaje_publico[kaggle$archivo == archivo]
  list(archivo = archivo,
       espacial = fila_esp$mae / 1e6,
       aleatoria = fila_ale$mae / 1e6,
       similares = fila_esp$mae_similares_chapinero / 1e6,
       sesgo = fila_esp$sesgo_similares_chapinero / 1e6,
       kaggle = if (length(puntaje) == 1) puntaje / 1e6 else NA_real_)
}

familias <- tibble(
  familia = c("Regresión lineal", "Elastic Net", "CART", "Random Forest",
              "LightGBM", "Red neuronal"),
  algoritmo_total = c("Regresión lineal", "Elastic Net", "CART",
                      "Random Forest", "LightGBM", "Red neuronal"),
  algoritmo_m2 = c(NA, NA, "CART precio m2", "Random Forest precio m2",
                   "LightGBM precio m2", NA),
  arbol = c(FALSE, FALSE, TRUE, TRUE, TRUE, FALSE)
)
total <- lapply(familias$algoritmo_total, fun_variante)
m2 <- lapply(familias$algoritmo_m2, function(.a) {
  if (is.na(.a)) NULL else fun_variante(.a)
})
names(total) <- names(m2) <- familias$familia
fun_extraer <- function(.lista, .campo) {
  sapply(.lista, function(.x) if (is.null(.x)) NA_real_ else .x[[.campo]])
}
familias <- familias |>
  mutate(esp_total = fun_extraer(total, "espacial"),
         ale_total = fun_extraer(total, "aleatoria"),
         sesgo_total = fun_extraer(total, "sesgo"),
         esp_m2 = fun_extraer(m2, "espacial"),
         kaggle_total = fun_extraer(total, "kaggle"),
         kaggle_m2 = fun_extraer(m2, "kaggle"))

# 1. Ranking con la misma información ---------------------------------------
ranking <- familias |>
  select(familia, `Precio total` = esp_total, `Precio por m²` = esp_m2) |>
  pivot_longer(-familia, names_to = "objetivo", values_to = "mae") |>
  filter(!is.na(mae)) |>
  mutate(familia = factor(familia, levels = familias$familia[
    order(familias$esp_total, decreasing = TRUE)]),
    objetivo = factor(objetivo, levels = c("Precio total", "Precio por m²")))
grafico_ranking <- ranking |>
  ggplot(aes(x = mae, y = familia, fill = objetivo)) +
  geom_col(position = position_dodge2(width = 0.8, preserve = "single",
                                      reverse = TRUE), width = 0.75) +
  geom_text(aes(label = fun_coma(mae, 1)),
            position = position_dodge2(width = 0.8, preserve = "single",
                                       reverse = TRUE),
            hjust = -0.15, size = 3.1, color = texto_1) +
  scale_fill_manual(values = c("Precio total" = color_1,
                               "Precio por m²" = color_2), name = NULL) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.12))) +
  labs(title = "Con la misma información, LightGBM tiene el menor error",
       subtitle = "MAE con la validación cruzada espacial (millones de pesos), 174 variables",
       x = "MAE (millones)", y = NULL,
       caption = paste0("Mismas 174 variables y mismos folds (20 regiones). Precio por m²: ",
                        "solo donde se conoce el área; donde no, precio total.")) +
  tema +
  theme(panel.grid.major.y = element_blank(),
        axis.text.y = element_text(color = texto_1, size = 10))
fun_guardar(grafico_ranking, "compare_ranking", 7.5, 4.6)

# 2. La validación aleatoria engaña más a los modelos flexibles ------------
optimismo <- familias |>
  mutate(familia = fct_reorder(familia, ale_total / esp_total, .desc = TRUE)) |>
  select(familia, Aleatoria = ale_total, Espacial = esp_total) |>
  pivot_longer(-familia, names_to = "cv", values_to = "mae")
etiquetas_opt <- familias |>
  mutate(pct = 100 * (1 - ale_total / esp_total),
         familia = factor(familia, levels = levels(optimismo$familia)))
grafico_optimismo <- optimismo |>
  ggplot(aes(x = mae, y = familia)) +
  geom_line(aes(group = familia), color = gris, linewidth = 1.2) +
  geom_point(aes(color = cv), size = 3.4) +
  geom_text(data = etiquetas_opt,
            aes(x = esp_total, label = paste0("-", fun_coma(pct, 0), "%")),
            hjust = -0.45, size = 3.2, color = texto_1, fontface = "bold") +
  scale_color_manual(values = c("Aleatoria" = color_3, "Espacial" = color_2),
                     name = "Validación cruzada") +
  scale_x_continuous(expand = expansion(mult = c(0.05, 0.15))) +
  labs(title = "La validación aleatoria promete mucho menos error a los modelos flexibles",
       subtitle = "MAE con cada validación (millones de pesos), precio total, 174 variables",
       x = "MAE (millones)", y = NULL,
       caption = "En negrita: cuánto menos error promete la aleatoria frente a la espacial.") +
  tema +
  theme(panel.grid.major.y = element_blank(),
        axis.text.y = element_text(color = texto_1, size = 10))
fun_guardar(grafico_optimismo, "compare_cv_optimismo", 7.5, 4.6)

# 3. Qué bajó más el error ----------------------------------------------------
# Todo con la CV espacial: más información (regresión lineal con 17 frente
# a 174 variables), mejor algoritmo (regresión lineal frente a LightGBM,
# las dos con 174) y ajuste de hiperparámetros (peor frente a mejor de los
# 30 ensayos de LightGBM).
lm_base <- fun_variante("Regresión lineal", "base")
reduccion <- tibble(
  fuente = c("Más información\n(regresión lineal: de 17 a 174 variables)",
             "Otro algoritmo\n(de regresión lineal a LightGBM, 174 variables)",
             "Ajustar hiperparámetros\n(de la peor a la mejor de 30 combinaciones)"),
  millones = c(lm_base$espacial - total[["Regresión lineal"]]$espacial,
               total[["Regresión lineal"]]$espacial - total[["LightGBM"]]$espacial,
               (max(ensayos$mae) - min(ensayos$mae)) / 1e6)
) |>
  mutate(fuente = factor(fuente, levels = rev(fuente)))
grafico_reduccion <- reduccion |>
  ggplot(aes(x = millones, y = fuente)) +
  geom_col(width = 0.6, fill = c(color_2, color_1, color_3)) +
  geom_text(aes(label = paste0("-", fun_coma(millones, 1))), hjust = -0.2,
            size = 3.6, color = texto_1, fontface = "bold") +
  scale_x_continuous(expand = expansion(mult = c(0, 0.15))) +
  labs(title = "La información y el algoritmo pesan más que los hiperparámetros",
       subtitle = "Reducción del MAE con la validación cruzada espacial (millones de pesos)",
       x = "Reducción del MAE (millones)", y = NULL,
       caption = "Validación cruzada espacial de 20 regiones, precio total.") +
  tema +
  theme(panel.grid.major.y = element_blank(),
        axis.text.y = element_text(color = texto_1, size = 9.5, lineheight = 0.95))
fun_guardar(grafico_reduccion, "compare_que_baja_error", 7.5, 4.2)

# 4. Más variables de texto: ¿ayudan? -----------------------------------------
# Conjunto completo (174: 141 raíces de palabras) frente a ampliado (más
# 12.000 términos TF-IDF), en Elastic Net y en LightGBM.
texto <- tibble(
  modelo = rep(c("Elastic Net", "LightGBM"), each = 2),
  conjunto = rep(c("Completo (174)", "Ampliado (12.174)"), 2),
  mae = c(total[["Elastic Net"]]$espacial,
          fun_variante("Elastic Net", "ampliado")$espacial,
          total[["LightGBM"]]$espacial,
          fun_variante("LightGBM", "ampliado")$espacial)
) |>
  mutate(conjunto = factor(conjunto, levels = c("Completo (174)",
                                                "Ampliado (12.174)")))
grafico_texto <- texto |>
  ggplot(aes(x = modelo, y = mae, fill = conjunto)) +
  geom_col(position = position_dodge(width = 0.75), width = 0.7) +
  geom_text(aes(label = fun_coma(mae, 1)), position = position_dodge(width = 0.75),
            vjust = -0.5, size = 3.4, color = texto_1) +
  scale_fill_manual(values = c("Completo (174)" = color_3,
                               "Ampliado (12.174)" = color_2), name = NULL) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.1))) +
  labs(title = "12.000 términos de texto ayudan al Elastic Net, no a LightGBM",
       subtitle = "MAE con la validación cruzada espacial (millones de pesos)",
       x = NULL, y = "MAE (millones)",
       caption = "Ampliado: el conjunto completo más la matriz TF-IDF de la descripción.") +
  tema +
  theme(panel.grid.major.x = element_blank(),
        axis.text.x = element_text(color = texto_1, size = 11))
fun_guardar(grafico_texto, "compare_texto", 6.5, 4.4)

# 5. Árboles frente a la red: el sesgo en viviendas como las de Chapinero --
# Sesgo = promedio de (predicción - precio real) en el 10% de train más
# parecido a Chapinero, con las predicciones fuera de muestra de la CV
# espacial. Negativo: el modelo se queda corto.
grafico_sesgo <- familias |>
  mutate(tipo = ifelse(arbol, "Árboles", "Lineales y red"),
         familia = fct_reorder(familia, sesgo_total)) |>
  ggplot(aes(x = sesgo_total, y = familia, fill = tipo)) +
  geom_col(width = 0.65) +
  geom_vline(xintercept = 0, color = texto_2) +
  geom_text(aes(label = fun_coma(sesgo_total, 0),
                hjust = ifelse(sesgo_total < 0, 1.2, -0.2)),
            size = 3.4, color = texto_1) +
  scale_fill_manual(values = c("Árboles" = color_2, "Lineales y red" = color_3),
                    name = NULL) +
  scale_x_continuous(expand = expansion(mult = 0.15)) +
  labs(title = "Los árboles se quedan cortos en viviendas como las de Chapinero",
       subtitle = "Sesgo medio: promedio de (predicción - precio real), millones de pesos",
       x = "Sesgo (millones); negativo = el modelo se queda corto", y = NULL,
       caption = paste0("10% de train más parecido a Chapinero, predicciones fuera de ",
                        "muestra con la CV espacial. Precio total, 174 variables.")) +
  tema +
  theme(panel.grid.major.y = element_blank(),
        axis.text.y = element_text(color = texto_1, size = 10))
fun_guardar(grafico_sesgo, "compare_sesgo", 7.5, 4.6)

# 5b. Cuánto de la red mezclar con LightGBM -----------------------------------
# Peso de la red elegido en el script 16: el que minimiza el MAE en las
# parecidas a Chapinero. También se muestra el MAE en toda la CV.
peso_elegido <- pesos_red$peso_red[which.min(pesos_red$mae_similares)]
curva_pesos <- pesos_red |>
  select(peso_red, `Parecidas a Chapinero` = mae_similares,
         `Toda la CV espacial` = mae_toda_cv) |>
  pivot_longer(-peso_red, names_to = "grupo", values_to = "mae")
grafico_pesos <- curva_pesos |>
  ggplot(aes(x = 100 * peso_red, y = mae, color = grupo)) +
  geom_vline(xintercept = 100 * peso_elegido, linetype = "dotted",
             color = texto_2) +
  geom_line(linewidth = 1.1) +
  geom_point(data = filter(curva_pesos, peso_red == peso_elegido), size = 3) +
  annotate("text", x = 100 * peso_elegido, y = max(curva_pesos$mae),
           label = paste0("Elegido: ", round(100 * peso_elegido), "% red"),
           hjust = -0.1, vjust = 1, size = 3.4, color = color_2,
           fontface = "bold") +
  scale_color_manual(values = c("Parecidas a Chapinero" = color_2,
                                "Toda la CV espacial" = color_3), name = NULL) +
  scale_x_continuous(labels = function(x) paste0(x, "%"), breaks = seq(0, 100, 10)) +
  labs(title = paste0("Mezclar un ", round(100 * peso_elegido),
                      "% de red neuronal reduce el error"),
       subtitle = "MAE de la mezcla LightGBM por m² + red neuronal (millones de pesos)",
       x = "Peso de la red neuronal en la mezcla", y = "MAE (millones)",
       caption = "Predicciones fuera de muestra con la CV espacial (script 16).") +
  tema
fun_guardar(grafico_pesos, "compare_peso_red", 7.5, 4.6)

# 6. Cifras del deck ------------------------------------------------------------
fun_cifra <- function(.x, .d = 1) ifelse(is.na(.x), "--", fun_coma(.x, .d))
fam_corto <- c(`Regresión lineal` = "LM", `Elastic Net` = "EN", CART = "CART",
               `Random Forest` = "RF", LightGBM = "LGBM", `Red neuronal` = "Red")
cifras <- c()
for (i in seq_len(nrow(familias))) {
  corto <- fam_corto[[familias$familia[i]]]
  cifras[paste0("CompEsp", corto)] <- fun_cifra(familias$esp_total[i])
  cifras[paste0("CompAle", corto)] <- fun_cifra(familias$ale_total[i])
  cifras[paste0("CompMdos", corto)] <- fun_cifra(familias$esp_m2[i])
  cifras[paste0("CompSesgo", corto)] <- fun_cifra(familias$sesgo_total[i], 0)
  cifras[paste0("CompOpt", corto)] <- fun_cifra(
    100 * (1 - familias$ale_total[i] / familias$esp_total[i]), 0)
  cifras[paste0("CompKag", corto)] <- fun_cifra(familias$kaggle_total[i])
  cifras[paste0("CompKagMdos", corto)] <- fun_cifra(familias$kaggle_m2[i])
}
fun_n_busqueda <- function(.archivo) nrow(read.csv(.archivo))
cifras <- c(
  cifras,
  RedInformacion = fun_coma(reduccion$millones[1], 1),
  RedAlgoritmo = fun_coma(reduccion$millones[2], 1),
  RedHiper = fun_coma(reduccion$millones[3], 1),
  TextoENCompleto = fun_coma(texto$mae[1], 1),
  TextoENAmpliado = fun_coma(texto$mae[2], 1),
  TextoLGBMCompleto = fun_coma(texto$mae[3], 1),
  TextoLGBMAmpliado = fun_coma(texto$mae[4], 1),
  KagLGBMAmpliado = fun_coma(fun_variante("LightGBM", "ampliado")$kaggle, 1),
  KagENAmpliado = fun_coma(fun_variante("Elastic Net", "ampliado")$kaggle, 1),
  PesoRedComp = paste0(round(100 * peso_elegido), "\\%"),
  MezclaSinRed = fun_coma(pesos_red$mae_similares[pesos_red$peso_red == 0], 1),
  MezclaSoloRed = fun_coma(pesos_red$mae_similares[pesos_red$peso_red == 1], 1),
  MezclaCVSinRed = fun_coma(pesos_red$mae_toda_cv[pesos_red$peso_red == 0], 1),
  MezclaCVConRed = fun_coma(pesos_red$mae_toda_cv[pesos_red$peso_red == peso_elegido], 1),
  MezclaConRed = fun_coma(min(pesos_red$mae_similares), 1),
  SesgoSinRed = fun_coma(pesos_red$sesgo_similares[pesos_red$peso_red == 0], 0),
  SesgoConRed = fun_coma(pesos_red$sesgo_similares[pesos_red$peso_red == peso_elegido], 0),
  NBusEN = fun_n_busqueda("02_outputs/tables/EN_completo_busqueda.csv"),
  NBusCART = fun_n_busqueda("02_outputs/tables/CART_completo_busqueda.csv"),
  NBusRF = fun_n_busqueda("02_outputs/tables/RF_completo_busqueda.csv"),
  NBusLGBM = nrow(ensayos),
  NBusRed = fun_n_busqueda("02_outputs/tables/NN_completo_busqueda.csv")
)
writeLines(paste0("\\newcommand{\\", names(cifras), "}{", cifras, "}"),
           "02_outputs/tables/compare_datos.tex")
print(data.frame(cifra = names(cifras), valor = cifras), row.names = FALSE)

# Limpiamos el entorno: borramos solo los objetos que creó este script.
rm(list = setdiff(ls(), objetos_previos))
