############################################################
# Compilar el deck Algorithm Comparison desde la raiz del PS3.
# Ejecutar: source("04_slides/compile_slides.R")
# No vuelve a entrenar modelos; solo actualiza graficos y PDF.
############################################################
local({
  source("01_code/14_compare_algorithms.R")
  raiz <- getwd()
  setwd("04_slides")
  tryCatch(
    tinytex::latexmk("compare_equipo_03.tex", engine = "xelatex", clean = TRUE),
    finally = setwd(raiz)
  )
  message("Presentacion lista en 04_slides/compare_equipo_03.pdf")
})


# Eliminar archivos auxiliares de LaTeX
unlink(list.files("04_slides",
                  pattern = "\\.(aux|log|out|toc|nav|snm|synctex\\.gz|fls|fdb_latexmk|vrb|bbl|blg|lof|lot|xdv)$",
                  full.names = TRUE))