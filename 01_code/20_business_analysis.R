##########################################################
# 20_business_analysis.R
#
# Objetivo: traducir los errores del mejor modelo a decisiones de compra en
# Chapinero: qué pasa si la empresa ofrece exactamente lo que predice el
# modelo, qué descuento sobre la predicción hace falta para que pocas
# compras se paguen de más, y en qué viviendas no conviene confiar en él.
#
# Etapa del Problem Set: Etapa 4 (del mejor modelo a decisiones de negocio).
# Insumos:  03_temp/train_model.rds, similitud_chapinero.rds,
#           oof_blend_vecinos_red.rds (scripts 05, 06 y 17);
#           02_outputs/tables/registro_modelos.csv; 00_data/kaggle_scores.csv.
# Salidas:  02_outputs/figures/negocio_*.pdf y
#           02_outputs/tables/negocio_datos.tex (cifras del deck como
#           comandos de LaTeX, por ejemplo \MargenRegla).
#
# Decisiones:
#   - Supuesto: el precio del anuncio es el valor de mercado de la vivienda
#     (es la mejor aproximación disponible).
#   - Si el modelo SOBRESTIMA (predicción > precio) y la empresa ofrece la
#     predicción, paga de más: pierde la diferencia. Si SUBESTIMA, la
#     oferta queda por debajo del precio: el vendedor no acepta y, como
#     mucho, se pierde el negocio. Por eso solo las sobrestimaciones
#     cuestan dinero.
#   - Regla de compra: ofrecer la predicción con un descuento (margen) m,
#     oferta = predicción x (1 - m). Riesgo = porcentaje de viviendas en
#     las que, aun con el descuento, la oferta supera el precio.
#   - Viviendas: el 10% de train más parecido a Chapinero, con las
#     predicciones fuera de muestra de la CV espacial (lo más cercano a
#     comprar en una zona que el modelo no vio).
#   - Ese error es menor que el que tuvo el modelo en Chapinero (MAE de la
#     tabla pública de Kaggle, que se calcula con viviendas de Chapinero).
#     Para no confiarse, los errores también se amplían por el cociente
#     entre ese MAE y el de la CV en las parecidas ("error en Chapinero");
#     el margen recomendado sale de ese escenario.
#   - Riesgo aceptable: 1 de cada 10 compras pagada de más. Es un supuesto
#     del negocio; la figura muestra el riesgo para cualquier margen.
#   - Para decidir dónde confiar, las viviendas se agrupan con lo que se
#     sabe ANTES de comprar (tipo, estrato y si el anuncio trae el área).
##########################################################

# Objetos que ya existían en el entorno antes de correr este script. Al
# final se borra solo lo que este script creó, sin tocar lo demás.
objetos_previos <- ls()

# Paquetes -----------------------------------------------------------------
if (!require(pacman)) install.packages("pacman")
pacman::p_load(dplyr, tidyr, ggplot2, forcats)

# Parámetros ---------------------------------------------------------------
riesgo_aceptable <- 0.10         # Máximo de compras pagadas de más.
margenes <- seq(0, 0.45, 0.01)   # Descuentos sobre la predicción.

# Estilo de las figuras (el mismo del script 19) ------------------------------
color_1 <- "#2a78d6"
color_2 <- "#0f3570"        # Azul oscuro.
color_3 <- "#9cc3ee"        # Azul claro.
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
train <- readRDS("03_temp/train_model.rds")
similitud <- readRDS("03_temp/similitud_chapinero.rds")
oof <- readRDS("03_temp/oof_blend_vecinos_red.rds")
registro <- read.csv("02_outputs/tables/registro_modelos.csv")
kaggle <- read.csv("00_data/kaggle_scores.csv")

for (.x in list(similitud, oof)) {
  if (!identical(.x$property_id, train$property_id)) {
    stop("Los insumos no están en el mismo orden de filas que train. ",
         "Corra de nuevo los scripts 05, 06 y 17.")
  }
}
mejor <- kaggle[which.min(kaggle$puntaje_publico), ]
fila <- registro[registro$archivo == mejor$archivo &
                   registro$tipo_cv == "espacial", ]
similares <- similitud$similitud >= stats::quantile(similitud$similitud, 0.9)

# Error relativo de cada vivienda: predicción / precio - 1 (positivo =
# sobrestima). Escenario "error en Chapinero": el mismo error, ampliado en
# la proporción en que el MAE de Kaggle supera al de la CV en las parecidas.
factor_kaggle <- mejor$puntaje_publico / fila$mae_similares_chapinero
parecidas <- tibble(
  precio = train$price,
  pred = oof[[grep("_espacial$", names(oof), value = TRUE)]],
  superficie_real = train$falta_surface_covered_completa == 0 |
    train$falta_surface_total_completa == 0,
  tipo = as.character(train$property_type),
  estrato = train$estrato_manzana
) |>
  filter(similares) |>
  mutate(error_rel = pred / precio - 1,
         error_rel_kaggle = factor_kaggle * error_rel)

# Riesgo de pagar de más con un descuento .m: predicción x (1 - m) > precio,
# es decir, (1 + error) x (1 - m) > 1.
fun_paga_de_mas <- function(.error, .m) (1 + .error) * (1 - .m) > 1
fun_riesgo <- function(.error, .m) mean(fun_paga_de_mas(.error, .m))

# 1. Sobrestimar y subestimar no cuestan lo mismo -------------------------------
pct_sobre <- mean(parecidas$error_rel > 0)
sobrepago_medio <- mean((parecidas$pred - parecidas$precio)[parecidas$error_rel > 0])
subestima_medio <- mean((parecidas$precio - parecidas$pred)[parecidas$error_rel < 0])
datos_errores <- parecidas |>
  mutate(error_pct = pmin(pmax(100 * error_rel, -60), 80),
         lado = ifelse(error_rel > 0, "Sobrestima", "Subestima"))
alto_hist <- max(table(cut(datos_errores$error_pct, seq(-60, 80, 4))))
grafico_errores <- datos_errores |>
  ggplot(aes(x = error_pct, fill = lado)) +
  geom_histogram(binwidth = 4, boundary = 0, color = "white", linewidth = 0.2) +
  geom_vline(xintercept = 0, color = texto_1, linewidth = 0.5) +
  annotate("label", x = -34, y = 0.78 * alto_hist, hjust = 0.5, size = 3.6,
           color = color_1, fill = "white", label.size = 0, lineheight = 1,
           label = paste0("Subestima: ", round(100 * (1 - pct_sobre)), "%\n",
                          "Ofrece menos de lo que vale:\n",
                          "el vendedor no acepta\n(se pierde el negocio)")) +
  annotate("label", x = 47, y = 0.78 * alto_hist, hjust = 0.5, size = 3.6,
           color = color_2, fill = "white", label.size = 0, lineheight = 1,
           label = paste0("Sobrestima: ", round(100 * pct_sobre), "%\n",
                          "Ofrece más de lo que vale:\n",
                          "paga de más, en promedio\n",
                          fun_coma(sobrepago_medio / 1e6, 0), " millones")) +
  scale_fill_manual(values = c("Sobrestima" = color_2, "Subestima" = color_3),
                    guide = "none") +
  scale_x_continuous(labels = function(x) paste0(ifelse(x > 0, "+", ""), x, "%"),
                     breaks = seq(-60, 80, 20)) +
  labs(title = paste0("Si se ofrece lo que predice el modelo, se paga de más en ",
                      round(10 * pct_sobre), " de cada 10 viviendas"),
       subtitle = "Diferencia entre la predicción y el precio, viviendas de train parecidas a Chapinero",
       x = "Predicción frente al precio", y = "Número de viviendas",
       caption = paste0("Cada barra agrupa viviendas con un error parecido; a la derecha de ",
                        "la línea, el modelo predice más que el precio.\nPredicciones ",
                        "fuera de muestra (CV espacial), 10% de train más parecido a ",
                        "Chapinero. Errores fuera de [-60%, +80%] en los extremos.")) +
  tema +
  theme(panel.grid.major.x = element_blank())
fun_guardar(grafico_errores, "negocio_errores", 7.5, 4.6)

# 2. Regla de compra con margen --------------------------------------------------
curva <- bind_rows(
  tibble(margen = margenes, escenario = "Con el error de la CV",
         riesgo = sapply(margenes, fun_riesgo, .error = parecidas$error_rel)),
  tibble(margen = margenes, escenario = "Con el error en Chapinero (recomendado)",
         riesgo = sapply(margenes, fun_riesgo, .error = parecidas$error_rel_kaggle))
) |>
  mutate(escenario = factor(escenario, levels = unique(escenario)))
fun_margen_minimo <- function(.escenario) {
  min(curva$margen[curva$escenario == .escenario & curva$riesgo <= riesgo_aceptable])
}
margen_cv <- fun_margen_minimo("Con el error de la CV")
margen_regla <- fun_margen_minimo("Con el error en Chapinero (recomendado)")
puntos_regla <- tibble(escenario = factor(levels(curva$escenario),
                                          levels = levels(curva$escenario)),
                       margen = c(margen_cv, margen_regla)) |>
  left_join(curva, by = c("escenario", "margen"))
riesgo_sin_margen <- fun_riesgo(parecidas$error_rel_kaggle, 0)

grafico_margen <- curva |>
  ggplot(aes(x = 100 * margen, y = 100 * riesgo, color = escenario,
             linetype = escenario)) +
  geom_hline(yintercept = 100 * riesgo_aceptable, color = texto_2,
             linetype = "dotted") +
  geom_line(linewidth = 1.1) +
  geom_point(data = puntos_regla, size = 3.2) +
  geom_text(data = puntos_regla,
            aes(label = paste0(round(100 * margen), "%"),
                hjust = ifelse(escenario == "Con el error de la CV", 1.4, -0.4)),
            vjust = ifelse(puntos_regla$escenario == "Con el error de la CV", 1.6, -1),
            size = 3.6, fontface = "bold", show.legend = FALSE) +
  annotate("text", x = 100 * max(margenes), y = 100 * riesgo_aceptable,
           label = "Meta: 1 de cada 10", vjust = -0.6, hjust = 1, size = 3.2,
           color = texto_2) +
  annotate("text", x = 1, y = 100 * riesgo_sin_margen, hjust = 0, vjust = -0.8,
           size = 3.2, color = texto_2,
           label = paste0("Sin descuento: ", round(100 * riesgo_sin_margen),
                          " de cada 100")) +
  scale_color_manual(values = c(gris, color_2), name = NULL) +
  scale_linetype_manual(values = c("dashed", "solid"), name = NULL) +
  scale_x_continuous(labels = function(x) paste0(x, "%"), breaks = seq(0, 45, 5)) +
  scale_y_continuous(limits = c(0, 55), breaks = seq(0, 50, 10)) +
  labs(title = "Cuanto mayor el descuento, menos compras se pagan de más",
       subtitle = "De cada 100 compras, cuántas quedarían pagadas por encima del precio",
       x = "Descuento sobre la predicción (oferta = predicción × (1 - descuento))",
       y = "Compras pagadas de más (de 100)",
       caption = paste0("Viviendas de train parecidas a Chapinero. Con el error en Chapinero: ",
                        "los errores de la CV multiplicados por ", fun_coma(factor_kaggle, 2),
                        "\n(MAE en Kaggle / MAE de la CV en las parecidas).")) +
  tema +
  theme(legend.key.width = unit(1.2, "cm"))
fun_guardar(grafico_margen, "negocio_margen", 7.5, 4.6)

# Costo del descuento: cuánto queda la oferta típica por debajo del precio.
oferta_tipica <- stats::median((1 + parecidas$error_rel_kaggle) * (1 - margen_regla)) - 1

# 3. Dónde no confiar en el modelo ------------------------------------------------
# Riesgo de pagar de más con el descuento recomendado, por grupos que se
# conocen antes de comprar.
grupos <- parecidas |>
  mutate(
    `Tipo` = tipo,
    `Estrato` = ifelse(estrato <= 3, "1 a 3", as.character(estrato)),
    `Área del anuncio` = ifelse(superficie_real, "La trae", "No la trae (imputada)"),
    paga_de_mas = fun_paga_de_mas(error_rel_kaggle, margen_regla)
  ) |>
  select(`Tipo`, `Estrato`, `Área del anuncio`, paga_de_mas) |>
  pivot_longer(-paga_de_mas, names_to = "variable", values_to = "grupo") |>
  group_by(variable, grupo) |>
  summarise(riesgo = mean(paga_de_mas), n = n(), .groups = "drop") |>
  mutate(variable = factor(variable, levels = c("Tipo", "Estrato", "Área del anuncio")),
         grupo = fct_rev(factor(grupo)),
         alto = riesgo > 1.5 * riesgo_aceptable,
         etiqueta = paste0(round(100 * riesgo), " de 100  (n = ",
                           formatC(n, big.mark = ".", decimal.mark = ",",
                                   format = "d"), ")"))
riesgo_total <- fun_riesgo(parecidas$error_rel_kaggle, margen_regla)

grafico_confiar <- grupos |>
  ggplot(aes(x = 100 * riesgo, y = grupo, fill = alto)) +
  geom_col(width = 0.65) +
  geom_vline(xintercept = 100 * riesgo_aceptable, linetype = "dotted",
             color = texto_2) +
  geom_text(aes(label = etiqueta), hjust = -0.08, size = 3, color = texto_1) +
  facet_grid(variable ~ ., scales = "free_y", space = "free_y", switch = "y") +
  scale_fill_manual(values = c("TRUE" = color_2, "FALSE" = color_3), guide = "none") +
  scale_x_continuous(expand = expansion(mult = c(0, 0.55))) +
  labs(title = paste0("Con ", round(100 * margen_regla),
                      "% de descuento, el riesgo se concentra en las casas"),
       subtitle = "De cada 100 compras de cada grupo, cuántas quedarían pagadas de más",
       x = "Compras pagadas de más (de 100)", y = NULL,
       caption = paste0("Viviendas de train parecidas a Chapinero, error en Chapinero. Línea ",
                        "punteada: meta de 10. Azul oscuro: más de 15 de cada 100.")) +
  tema +
  theme(panel.grid.major.y = element_blank(),
        strip.placement = "outside",
        strip.text.y.left = element_text(angle = 0, hjust = 1, face = "bold",
                                         color = texto_1, size = 9),
        axis.text.y = element_text(color = texto_1, size = 9),
        panel.spacing = unit(0.7, "lines"))
fun_guardar(grafico_confiar, "negocio_donde_confiar", 7.5, 4.6)

# 4. Cifras del deck ------------------------------------------------------------
riesgo_grupo <- function(.variable, .grupo) {
  fun_pct(grupos$riesgo[grupos$variable == .variable & grupos$grupo == .grupo])
}
cifras <- c(
  PctSobre = fun_pct(pct_sobre),
  PctSub = fun_pct(1 - pct_sobre),
  SobrepagoMedio = fun_coma(sobrepago_medio / 1e6, 0),
  SubestimaMedio = fun_coma(subestima_medio / 1e6, 0),
  FactorKaggle = fun_coma(factor_kaggle, 2),
  MargenCV = paste0(round(100 * margen_cv), "\\%"),
  MargenRegla = paste0(round(100 * margen_regla), "\\%"),
  RiesgoSinMargen = fun_pct(riesgo_sin_margen),
  RiesgoRegla = fun_pct(riesgo_total),
  OfertaTipica = fun_pct(-oferta_tipica),
  NParecidas = formatC(nrow(parecidas), big.mark = ".", decimal.mark = ",",
                       format = "d"),
  RiesgoCasa = riesgo_grupo("Tipo", "Casa"),
  RiesgoApto = riesgo_grupo("Tipo", "Apartamento"),
  RiesgoEstBajo = riesgo_grupo("Estrato", "1 a 3"),
  RiesgoEstSeis = riesgo_grupo("Estrato", "6"),
  RiesgoImputada = riesgo_grupo("Área del anuncio", "No la trae (imputada)"),
  RiesgoReal = riesgo_grupo("Área del anuncio", "La trae")
)
writeLines(paste0("\\newcommand{\\", names(cifras), "}{", cifras, "}"),
           "02_outputs/tables/negocio_datos.tex")
print(data.frame(cifra = names(cifras), valor = cifras), row.names = FALSE)

# Limpiamos el entorno: borramos solo los objetos que creó este script.
rm(list = setdiff(ls(), objetos_previos))
