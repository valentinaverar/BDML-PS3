##########################################################
# fun_simular_cv.R
#
# Funciones para elegir el esquema de validación cruzada espacial que mejor
# anticipa el error en Chapinero. Las usa 06_create_cv_folds.R.
#
# Idea: dentro de train se aparta una zona parecida a Chapinero
# ("pseudo-Chapinero"), se mide el error verdadero de un modelo en esa zona
# y se compara con el error que promete cada esquema de CV usando solo el
# resto de train. El mejor esquema es el que más se acerca al error
# verdadero.
##########################################################

fun_esquemas_cv <- function(.xy, .k = 5, .semilla = 2026) {
  #' Asignar observaciones a folds con distintos esquemas de CV
  #'
  #' @description \code{fun_esquemas_cv()} devuelve, para cada esquema, el
  #' fold (1 a .k) de cada observación. Esquemas: aleatoria; k-means con
  #' .k regiones (cada región es un fold); k-means con 20 regiones
  #' repartidas al azar en .k folds; y bloques cuadrados de 2 km, 1 km,
  #' 700 m, 500 m y 300 m repartidos al azar en .k folds (spatial block CV,
  #' Lecture 14). Las observaciones con las mismas coordenadas siempre
  #' quedan en el mismo fold en los esquemas espaciales.
  #'
  #' @param .xy \code{matrix}. Coordenadas proyectadas en metros (dos
  #' columnas: x, y), una fila por observación.
  #' @param .k \code{integer}. Número de folds.
  #' @param .semilla \code{integer}. Semilla para reproducir los folds.
  #'
  #' @returns \code{list} con nombre: un vector de folds por esquema.
  n <- nrow(.xy)
  
  fun_kmeans <- function(.regiones) {
    set.seed(.semilla)
    region <- suppressWarnings(stats::kmeans(
      .xy, centers = .regiones, nstart = 10, iter.max = 50)$cluster)
    if (.regiones == .k) return(region)
    set.seed(.semilla)
    fold_region <- sample(rep(seq_len(.k), length.out = .regiones))
    fold_region[region]
  }
  fun_bloques <- function(.lado) {
    bloque <- paste(floor(.xy[, 1] / .lado), floor(.xy[, 2] / .lado))
    unicos <- unique(bloque)
    set.seed(.semilla)
    fold_bloque <- sample(rep(seq_len(.k), length.out = length(unicos)))
    fold_bloque[match(bloque, unicos)]
  }
  
  set.seed(.semilla)
  list(
    "Aleatoria" = sample(rep(seq_len(.k), length.out = n)),
    "k-means 5 regiones" = fun_kmeans(.k),
    "k-means 20 regiones" = fun_kmeans(20),
    "Bloques de 2 km" = fun_bloques(2000),
    "Bloques de 1 km" = fun_bloques(1000),
    "Bloques de 700 m" = fun_bloques(700),
    "Bloques de 500 m" = fun_bloques(500),
    "Bloques de 300 m" = fun_bloques(300)
  )
}

fun_ajustar_predecir <- function(.modelo, .x_ajuste, .y_ajuste, .x_nuevo,
                                 .semilla = 2026) {
  #' Ajustar un modelo de prueba y predecir
  #'
  #' @description Modelos de prueba para comparar esquemas de CV: uno
  #' flexible ("Random Forest", ranger con 100 árboles) y uno simple
  #' ("Regresión lineal", MCO). No se afinan: solo sirven para medir qué
  #' tan bien anticipa cada esquema el error verdadero.
  #'
  #' @param .modelo \code{character}. "Random Forest" o "Regresión lineal".
  #' @param .x_ajuste,.x_nuevo \code{data.frame} numéricos sin NA.
  #' @param .y_ajuste \code{numeric}. Variable a predecir (log del precio).
  #' @param .semilla \code{integer}. Semilla del Random Forest.
  #'
  #' @returns \code{numeric}: predicciones para .x_nuevo (en log).
  if (.modelo == "Random Forest") {
    ajuste <- ranger::ranger(x = .x_ajuste, y = .y_ajuste, num.trees = 100,
                             min.node.size = 5, seed = .semilla,
                             verbose = FALSE)
    return(stats::predict(ajuste, .x_nuevo, verbose = FALSE)$predictions)
  }
  ajuste <- stats::lm.fit(cbind(1, as.matrix(.x_ajuste)), .y_ajuste)
  coeficientes <- ajuste$coefficients
  coeficientes[is.na(coeficientes)] <- 0  # columnas colineales
  drop(cbind(1, as.matrix(.x_nuevo)) %*% coeficientes)
}

fun_evaluar_esquemas <- function(.x, .y, .xy, .zonas, .k = 5,
                                 .prop_conservar = 0.12, .semilla = 2026,
                                 .modelos = c("Random Forest",
                                              "Regresión lineal")) {
  #' Comparar esquemas de CV contra el error verdadero en pseudo-Chapineros
  #'
  #' @description Para cada zona de .zonas: (1) aparta como pseudo-test la
  #' zona, salvo una fracción .prop_conservar de sus propiedades que se
  #' deja en train (así la zona queda tan cerca de train como Chapinero);
  #' (2) entrena cada modelo con el resto y mide el MAE verdadero en la
  #' zona; (3) con el resto de train, estima el MAE con cada esquema de
  #' fun_esquemas_cv(). El MAE se mide en millones de pesos.
  #'
  #' @param .x \code{data.frame} numérico sin NA (predictores de train).
  #' @param .y \code{numeric}. log(precio) de train.
  #' @param .xy \code{matrix}. Coordenadas en metros de train.
  #' @param .zonas \code{list} con nombre: índices de filas de cada zona.
  #' @param .k,.semilla Folds y semilla.
  #' @param .prop_conservar \code{numeric}. Fracción de la zona que se deja
  #' en train.
  #' @param .modelos \code{character}. Modelos de prueba.
  #'
  #' @returns \code{data.frame} con una fila por zona, modelo y esquema:
  #' mae_cv, mae_verdadero y brecha_pct (100 * (mae_cv - mae_verdadero) /
  #' mae_verdadero; negativa = la CV promete menos error del real).
  fun_mae <- function(.real, .pred) mean(abs(exp(.real) - exp(.pred))) / 1e6
  resultados <- list()
  
  for (zona in names(.zonas)) {
    filas_zona <- .zonas[[zona]]
    set.seed(.semilla)
    conservar <- sample(filas_zona, round(.prop_conservar * length(filas_zona)))
    fuera <- setdiff(filas_zona, conservar)
    dentro <- setdiff(seq_len(nrow(.x)), fuera)
    esquemas <- fun_esquemas_cv(.xy[dentro, , drop = FALSE], .k, .semilla)
    
    for (modelo in .modelos) {
      message("Zona ", zona, " | ", modelo, ": error verdadero...")
      pred_verdad <- fun_ajustar_predecir(modelo, .x[dentro, ], .y[dentro],
                                          .x[fuera, ], .semilla)
      mae_verdadero <- fun_mae(.y[fuera], pred_verdad)
      
      for (esquema in names(esquemas)) {
        fold <- esquemas[[esquema]]
        pred_cv <- numeric(length(dentro))
        for (f in seq_len(.k)) {
          validar <- which(fold == f)
          pred_cv[validar] <- fun_ajustar_predecir(
            modelo, .x[dentro[-validar], ], .y[dentro[-validar]],
            .x[dentro[validar], ], .semilla)
        }
        resultados[[length(resultados) + 1]] <- data.frame(
          zona = zona, modelo = modelo, esquema = esquema,
          mae_cv = fun_mae(.y[dentro], pred_cv),
          mae_verdadero = mae_verdadero
        )
      }
      message("  listo: MAE verdadero = ", round(mae_verdadero, 1),
              " millones.")
    }
  }
  
  resultados <- do.call(rbind, resultados)
  resultados$brecha_pct <- 100 * (resultados$mae_cv - resultados$mae_verdadero) /
    resultados$mae_verdadero
  resultados
}
