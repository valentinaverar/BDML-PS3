# Making Money with ML? Predicción de precios de vivienda en Chapinero

**Problem Set 3: predicción de precios de vivienda con modelos de regresión**

**Equipo 03:** Julián Herrera, Andrés Silva, Valentina Vera  
Big Data y Machine Learning para Economía Aplicada (MECA 4107), Universidad de los Andes, 2026-20

---

Este repositorio contiene el código para predecir el precio de las viviendas de Chapinero (Bogotá) con datos de Properati del resto de la ciudad. Las predicciones se evalúan con el error absoluto medio (MAE) en la competencia de Kaggle del curso.

La pregunta que guía el trabajo es: **¿cuál es el mejor modelo para predecir precios en Chapinero, y es lo bastante confiable para decidir qué comprar?** Como en el caso de Zillow, no todos los errores cuestan lo mismo: sobreestimar un precio hace que la empresa pague de más, mientras que subestimarlo, en el peor caso, significa perder un negocio.

## Resultados principales

*(Se completa al final del proyecto.)*

## Instrucciones de replicación

Desde la carpeta raíz del repositorio, en R:

```r
source("01_code/00_rundirectory.R")
```

El script maestro corre todo en orden: preparación de los datos y construcción de predictores (scripts 01 a 06), modelos (07 a 12) y análisis para las presentaciones (13 a 16).

**Datos.** No están en el repositorio. Los datos de la competencia deben guardarse en `00_data/raw/`.

## Estructura del repositorio

```
BDML-PS3/
├── 00_data/
│   ├── raw/                  # datos de Kaggle (no versionados)
│   └── kaggle_scores.csv     # puntaje de cada envío a Kaggle
├── 01_code/
│   ├── 00_rundirectory.R     # script maestro
│   ├── 01_ a 06_*.R          # datos: obtención, limpieza, predictores (OSM y texto), descriptivas y folds
│   ├── 07_ a 12_model_*.R    # un script por familia de algoritmos
│   ├── 13_ a 16_*.R          # comparación, validación espacial, mejor modelo y decisión de compra
│   └── functions/            # funciones reutilizables
├── 02_outputs/
│   ├── figures/              # figuras de las presentaciones
│   ├── tables/               # tablas .tex y registro de modelos
│   └── predictions/          # archivos enviados a Kaggle
├── 03_temp/                  # bases limpias y resultados intermedios (no versionados)
└── 04_slides/                # presentaciones (.tex y .pdf)
```

## Funciones reutilizables

- `functions/fun_save_submission.R`: guarda las predicciones en el formato de Kaggle y verifica el archivo antes del envío (identificadores completos, sin faltantes ni precios no positivos).
- `functions/fun_metricas.R`: calcula el MAE, el RMSE y la dirección de los errores (sobreestimación frente a subestimación), y registra cada variante en `02_outputs/tables/registro_modelos.csv`.

## Salidas

Todas se generan con el código, en `02_outputs/`:

- **Figuras** (`figures/`) y **tablas** (`tables/`, en `.tex`) que usan las presentaciones.
- **Predicciones** (`predictions/`): los archivos enviados a Kaggle. El nombre indica el algoritmo y sus hiperparámetros.
- **Registro de modelos** (`tables/registro_modelos.csv`): una fila por variante, con su configuración, el tipo de validación cruzada (aleatoria o espacial) y sus métricas de error fuera de muestra.

## Software

- R 4.6.1. Los paquetes se instalan y cargan con `pacman::p_load()`.
