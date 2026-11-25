# Making Money with ML? Predicción de precios de vivienda en Chapinero

**Problem Set 3: predicción de precios de vivienda con modelos de regresión**

**Equipo 03:** Julián Herrera, Andrés Silva, Valentina Vera  
Big Data y Machine Learning para Economía Aplicada (MECA 4107), Universidad de los Andes, 2026-20

---

Este repositorio contiene el código para predecir el precio de las viviendas de Chapinero (Bogotá) con datos de Properati del resto de la ciudad. Las predicciones se evalúan con el error absoluto medio (MAE) en la competencia de Kaggle del curso.

La pregunta que guía el trabajo es: **¿cuál es el mejor modelo para predecir precios en Chapinero, y es lo bastante confiable para decidir qué comprar?** Como en el caso de Zillow, no todos los errores cuestan lo mismo: sobreestimar un precio hace que la empresa pague de más, mientras que subestimarlo, en el peor caso, significa perder un negocio.

## Resultados principales

**Mejor modelo: MAE de 164,5 millones en Kaggle** (tabla pública), 45% menos que la regresión lineal base (296,6). Es un LightGBM que predice el precio por m² y usa el precio de las viviendas vecinas, mezclado con una red neuronal:

1. **Vecinos:** para cada vivienda, la mediana del precio y del precio por m² de las 10 viviendas de train más cercanas (rezago espacial del precio), calculada sin fuga de información en la validación cruzada.
2. **LightGBM:** predice el precio por m² y lo multiplica por el área donde el área es real; donde se imputó, predice el precio total.
3. **Mezcla:** 75% LightGBM + 25% red neuronal. Los árboles subestiman donde los precios son altos y la red corrige parte de ese sesgo.

Cómo se llegó ahí (MAE en Kaggle, millones de pesos):

| Paso | Script | MAE |
|---|---|---|
| Regresión lineal, 17 variables de la base | 07 | 296,6 |
| Regresión lineal, 174 variables (entorno y texto) | 07 | 215,6 |
| LightGBM, 174 variables | 11 | 178,6 |
| LightGBM que predice el precio por m² | 13 | 169,2 |
| + precio de los vecinos | 15 | 166,7 |
| + mezcla con la red neuronal (**mejor modelo**) | 17 | **164,5** |

Otros hallazgos:

- **Con las mismas reglas gana LightGBM.** Con las mismas 174 variables, los mismos folds y el mismo objetivo, tiene el menor error de las seis familias en la validación cruzada espacial.
- **La validación aleatoria engaña a los modelos flexibles.** Con ella habría ganado Random Forest; con la espacial ganó LightGBM, y Kaggle le dio la razón a la espacial. La validación espacial de 20 regiones se eligió entre 8 esquemas, simulando "pseudo-Chapineros" dentro de train.
- **La información pesó más que los hiperparámetros.** Pasar de 17 a 174 variables y cambiar de algoritmo bajaron el error mucho más que ajustar los hiperparámetros.
- **Para comprar, el modelo sirve con margen.** En viviendas parecidas a Chapinero sobrestima el precio en casi la mitad de los casos. Ofreciendo 30% menos de lo que predice, solo 1 de cada 10 compras quedaría pagada por encima del precio. El riesgo es mayor en casas y estratos bajos.

## Presentaciones

- **El mejor modelo: ¿podemos confiar en él para comprar vivienda en Chapinero?** (`04_slides/best_equipo_03.pdf`). Cuenta la historia del mejor modelo, de los datos a la decisión de compra: cómo es Chapinero, qué variables usamos, cómo elegimos la validación cruzada, cómo funciona el modelo, en qué se equivoca y con qué regla comprar.
- **Comparación de algoritmos: ¿por qué ganó el mejor modelo?** (`04_slides/compare_equipo_03.pdf`). Compara las seis familias con las mismas reglas y explica las diferencias por el algoritmo, las variables, el ajuste de hiperparámetros y las características de los datos.

Las figuras, tablas y cifras de las presentaciones no se escriben a mano: las generan los scripts 18 a 20, y `04_slides/compile_slides.R` compila los dos decks.

## Instrucciones de replicación

Desde la carpeta raíz del repositorio, en R:

```r
source("01_code/00_rundirectory.R")
```

El script maestro corre todo en el orden de la numeración. Cada script también puede correrse solo, una vez corridos los anteriores.

**Requisitos:**

- **Datos de Kaggle.** No están en el repositorio. `01_get_data.R` los deja en `00_data/raw/`: usa los CSV si ya están, descomprime el zip de la competencia si está en esa carpeta, o lo descarga con la CLI de Kaggle si está instalada y configurada.
- **Datos externos.** OpenStreetMap, estaciones de TransMilenio, estratificación por manzana y pendiente del terreno están guardados en `00_data/external/geo_cache/` (copia fija descargada entre el 16 y el 20 de noviembre de 2026), para que las variables de entorno no cambien si OpenStreetMap se actualiza.
- **Paquetes de R.** Se instalan y cargan con `pacman::p_load()` al inicio de cada script.
- **Red neuronal (script 12).** Usa `keras3`, que necesita Python con TensorFlow. Si no está instalado, puede instalarse una vez con `keras3::install_keras()`.
- **Presentaciones.** Se compilan con XeLaTeX vía `tinytex`. Si no hay LaTeX instalado, el script maestro avisa y sigue sin compilar.

Los pasos más demorados son la elección del esquema de validación cruzada (script 06) y las búsquedas de hiperparámetros de los modelos (scripts 09 a 12).

## Estructura del repositorio

```
BDML-PS3/
├── 00_data/
│   ├── raw/                  # datos de Kaggle (no versionados)
│   ├── external/geo_cache/   # copia fija de los datos externos
│   └── kaggle_scores.csv     # puntaje de cada envío a Kaggle
├── 01_code/
│   ├── 00_rundirectory.R     # script maestro
│   ├── 01_ a 06_*.R          # datos, predictores y particiones de validación cruzada
│   ├── 07_ a 12_model_*.R    # las seis familias, un script por familia
│   ├── 13_ a 17_model_*.R    # mejoras sobre LightGBM hasta el mejor modelo
│   ├── 18_ a 20_*.R          # figuras y cifras de las presentaciones
│   └── functions/            # funciones reutilizables
├── 02_outputs/
│   ├── figures/              # figuras de las presentaciones
│   ├── tables/               # tablas, cifras y registro de modelos
│   └── predictions/          # archivos enviados a Kaggle
├── 03_temp/                  # bases limpias y resultados intermedios (no versionados)
└── 04_slides/                # presentaciones (.tex y .pdf) y compile_slides.R
```

### Scripts

| Script | Qué hace |
|---|---|
| `01_get_data.R` | Obtiene los datos de la competencia. |
| `02_clean_data.R` | Limpieza estructural de train y test: marca con indicadores las superficies faltantes, inconsistentes o extremas, sin eliminar observaciones. |
| `03_build_geo_features.R` | Variables de entorno: distancias a TransMilenio, vías, parques, centros comerciales, universidades y centro de empleo; zona verde, restaurantes, pendiente y estrato. |
| `04_build_text_features.R` | Variables de texto: 141 raíces de palabras (regla del 5%), cantidades extraídas (área, baños, parqueaderos, piso) y matriz TF-IDF. |
| `05_build_model_data.R` | Imputación de faltantes y conjuntos de variables: base (17), completo (174) y ampliado (12.174). |
| `06_create_cv_folds.R` | Similitud con Chapinero y particiones de validación cruzada: aleatoria y espacial de 20 regiones, elegida entre 8 esquemas con pseudo-Chapineros. |
| `07_model_linear_regression.R` | Regresión lineal. |
| `08_model_elastic_net.R` | Elastic Net. |
| `09_model_cart.R` | CART (precio total y precio por m²). |
| `10_model_random_forest.R` | Random Forest (precio total y precio por m²). |
| `11_model_boosting.R` | LightGBM, con búsqueda aleatoria de 30 combinaciones de hiperparámetros. |
| `12_model_neural_network.R` | Red neuronal. |
| `13_model_boosting_precio_m2.R` | LightGBM que predice el precio por m². |
| `14_model_blend_elastic_net.R` | Mezcla de LightGBM con Elastic Net. |
| `15_model_boosting_vecinos.R` | LightGBM por m² con el precio de las viviendas vecinas. |
| `16_model_blend_red.R` | Mezcla de LightGBM por m² con la red neuronal; elige el peso de la red. |
| `17_model_blend_vecinos_red.R` | **Mejor modelo:** LightGBM con vecinos mezclado con la red. |
| `18_compare_algorithms.R` | Figuras y cifras del deck de comparación. |
| `19_best_model_analysis.R` | Figuras y cifras del deck del mejor modelo. |
| `20_business_analysis.R` | Análisis de negocio: sobrestimación, regla de compra con margen y dónde no confiar en el modelo. |

### Funciones reutilizables (`01_code/functions/`)

- `fun_save_submission.R`: guarda las predicciones en el formato de Kaggle y verifica el archivo antes del envío.
- `fun_metricas.R`: calcula el MAE, el RMSE y la dirección de los errores, y registra cada variante en `02_outputs/tables/registro_modelos.csv`.
- `fun_osm.R` y `fun_geometrias.R`: descargan capas de OpenStreetMap con copia local y extraen las geometrías de interés.
- `fun_preprocesar_texto.R` y `fun_construir_tfidf.R`: limpieza común de las descripciones y matriz TF-IDF.
- `fun_similitud_chapinero.R`: similitud de cada vivienda con las de Chapinero (propensity score).
- `fun_simular_cv.R`: simulación con pseudo-Chapineros para elegir el esquema de validación cruzada espacial.
- `fun_vecinos.R`: rezago espacial del precio (precio de las viviendas vecinas de train).

## Salidas

Todas se generan con el código, en `02_outputs/`:

- **Figuras** (`figures/`) y **tablas y cifras** (`tables/`) que usan las presentaciones.
- **Predicciones** (`predictions/`): los archivos enviados a Kaggle. El nombre indica el algoritmo y sus hiperparámetros.
- **Registro de modelos** (`tables/registro_modelos.csv`): una fila por variante y tipo de validación cruzada, con su configuración y sus métricas de error fuera de muestra (también en las viviendas más parecidas a Chapinero).
- **Puntajes de Kaggle** (`00_data/kaggle_scores.csv`): puntaje público de cada envío, con su descripción y autor.

## Software

R 4.6.1. Los paquetes principales son `dplyr`, `sf`, `osmdata`, `terra`, `tidytext`, `glmnet`, `rpart`, `ranger`, `lightgbm`, `keras3`, `FNN`, `ggplot2` y `patchwork`.