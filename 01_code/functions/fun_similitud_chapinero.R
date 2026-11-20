##########################################################
# fun_similitud_chapinero.R
#
# Función para medir qué tanto se parece cada propiedad a las de Chapinero
# (test), con un propensity score ("adversarial validation"). Se usa en
# 05_create_cv_folds.R para elegir zonas de train parecidas a Chapinero
# y, en la Etapa 4, para medir el error del modelo en propiedades
# parecidas a las que la empresa quiere comprar.
##########################################################

fun_similitud_chapinero <- function(.train, .test, .vars, .k = 5,
                                    .semilla = 2026) {
  #' Probabilidad de que una propiedad "parezca" de Chapinero
  #'
  #' @description \code{fun_similitud_chapinero()} junta train y test, crea
  #' la variable es_test (1 = Chapinero) y estima una regresión logística
  #' que intenta distinguir unas de otras a partir de sus características.
  #' La probabilidad predicha es la similitud con Chapinero: una propiedad
  #' de train que el modelo confunde con una de test es muy parecida a
  #' ellas. No usa el precio (desconocido en test), así que no filtra
  #' información de la variable a predecir; y no debe recibir coordenadas
  #' ni nombres de lugares, porque entonces mediría "estar en Chapinero"
  #' (algo que ninguna propiedad de train cumple) y no "parecerse".
  #'
  #' La probabilidad se calcula fuera de muestra: los datos se parten en
  #' .k grupos y la de cada grupo sale de un modelo estimado con los demás,
  #' para que ninguna propiedad influya en su propio puntaje.
  #'
  #' @param .train,.test \code{data.frame}. Bases con las variables de
  #' .vars (pueden tener NA).
  #' @param .vars \code{character}. Predictores numéricos o 0/1. Los NA se
  #' imputan con la mediana conjunta y se agrega un indicador de faltante;
  #' las variables se estandarizan para que los coeficientes sean
  #' comparables entre sí.
  #' @param .k \code{integer}. Número de grupos para el cálculo fuera de
  #' muestra.
  #' @param .semilla \code{integer}. Semilla para reproducir los grupos.
  #'
  #' @returns \code{list} con: \code{train} y \code{test} (vectores de
  #' similitud, entre 0 y 1, en el orden de filas de cada base),
  #' \code{auc} (capacidad del modelo para distinguir test de train, fuera
  #' de muestra) y \code{coeficientes} (data.frame ordenado por la
  #' magnitud del coeficiente estandarizado).
  datos <- rbind(.train[.vars], .test[.vars])
  es_test <- c(rep(0L, nrow(.train)), rep(1L, nrow(.test)))
  
  # Faltantes: mediana conjunta + indicador; luego estandarizar.
  for (v in .vars) {
    if (anyNA(datos[[v]])) {
      datos[[paste0("falta_", v)]] <- as.integer(is.na(datos[[v]]))
      datos[[v]][is.na(datos[[v]])] <- stats::median(datos[[v]], na.rm = TRUE)
    }
  }
  constantes <- vapply(datos, function(x) stats::sd(x) == 0, logical(1))
  datos <- as.data.frame(scale(datos[!constantes]))
  
  # Probabilidad fuera de muestra, con grupos equilibrados en es_test.
  set.seed(.semilla)
  grupo <- integer(length(es_test))
  for (clase in 0:1) {
    filas <- which(es_test == clase)
    grupo[filas] <- sample(rep(seq_len(.k), length.out = length(filas)))
  }
  similitud <- numeric(length(es_test))
  for (g in seq_len(.k)) {
    ajuste <- suppressWarnings(stats::glm(
      es_test[grupo != g] ~ ., data = datos[grupo != g, , drop = FALSE],
      family = stats::binomial()))
    similitud[grupo == g] <- suppressWarnings(stats::predict(
      ajuste, newdata = datos[grupo == g, , drop = FALSE], type = "response"))
  }
  
  # AUC: probabilidad de que una propiedad de test reciba mayor similitud
  # que una de train elegida al azar (0,5 = no las distingue; 1 = perfecto).
  rango <- rank(similitud)
  n1 <- sum(es_test == 1)
  n0 <- sum(es_test == 0)
  auc <- (sum(rango[es_test == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
  
  # Coeficientes del modelo con todos los datos, para interpretar.
  completo <- suppressWarnings(stats::glm(es_test ~ ., data = datos,
                                          family = stats::binomial()))
  coeficientes <- data.frame(variable = names(stats::coef(completo))[-1],
                             coeficiente = unname(stats::coef(completo))[-1])
  coeficientes <- coeficientes[order(-abs(coeficientes$coeficiente)), ]
  
  list(train = similitud[es_test == 0], test = similitud[es_test == 1],
       auc = auc, coeficientes = coeficientes)
}
