##########################################################
# fun_preprocesar_texto.R
#
# Limpieza de los textos de los anuncios de Properati. La usan
# 04_build_text_features.R (raíces y cantidades) y fun_construir_tfidf()
# (matriz TF-IDF), para que todo el texto pase por la misma limpieza.
##########################################################

fun_preprocesar_texto <- function(texto) {
  #' Limpiar el texto de los anuncios inmobiliarios
  #'
  #' @description \code{fun_preprocesar_texto()} devuelve un vector de texto
  #' nuevo (no modifica title ni description): en minúsculas, sin tildes ni
  #' signos, con los restos de HTML de la fuente corregidos ("ba ntilde o"
  #' -> "bano"), con las palabras que perdieron la ñ o la tilde corregidas
  #' ("bao" -> "bano", "ubicacin" -> "ubicacion") y sin las frases
  #' publicitarias recurrentes de las inmobiliarias ("contactanos",
  #' "visita nuestra pagina web"...). Es una limpieza conservadora: conserva
  #' los números y las palabras de negación ("sin", "no", "con"), que sirven
  #' para extraer cantidades y para los bigramas del TF-IDF.
  #'
  #' @param texto \code{character}. Textos originales (pueden tener NA).
  #'
  #' @returns \code{character} del mismo largo, con el texto limpio. Los NA
  #' se convierten en "".
  texto[is.na(texto)] <- ""
  texto <- stringi::stri_trans_general(tolower(texto), "Latin-ASCII")
  texto <- stringr::str_squish(
    stringr::str_replace_all(texto, "[^a-z0-9]+", " ")
  )
  
  # Restos de HTML de la fuente: "am eacute ricas" -> "americas",
  # "ba ntilde o" -> "bano"; y etiquetas sueltas ("br", "b", "nbsp").
  texto <- stringr::str_replace_all(
    texto, "\\s?([aeiou])(acute|grave|uml|circ)\\s?", "\\1"
  )
  texto <- stringr::str_replace_all(texto, "\\s?ntilde\\s?", "n")
  texto <- stringr::str_replace_all(texto, "\\b(nbsp|amp|quot|br|b)\\b", " ")
  
  # Errores frecuentes observados en train.csv y test.csv.
  correcciones <- c(
    "\\bbaos\\b" = "banos", "\\bbao\\b" = "bano",
    "\\bbalcn\\b" = "balcon", "\\bbalcnes\\b" = "balcones",
    "\\bsaln\\b" = "salon", "\\bhabitacin\\b" = "habitacion",
    "\\biluminacin\\b" = "iluminacion", "\\bpanormica\\b" = "panoramica",
    "\\bportera\\b" = "porteria", "\\brecepcin\\b" = "recepcion",
    "\\bcermica\\b" = "ceramica", "\\bdiseo\\b" = "diseno",
    "\\bantigedad\\b" = "antiguedad", "\\baos\\b" = "anos",
    "\\binformacin\\b" = "informacion", "\\brea\\b" = "area",
    "\\bnios\\b" = "ninos", "\\bjardn\\b" = "jardin",
    "\\bpequeo\\b" = "pequeno", "\\btambin\\b" = "tambien",
    "\\bzona de lavandera\\b" = "zona de lavanderia",
    "\\bzona lavandera\\b" = "zona lavanderia",
    "\\bcuarto de lavandera\\b" = "cuarto de lavanderia",
    # Palabras en "-cion" que perdieron la tilde de la o ("ubicacin").
    "cin\\b" = "cion"
  )
  texto <- stringr::str_replace_all(texto, correcciones)
  
  # Frases comerciales recurrentes verificadas en los anuncios.
  # Se eliminan fragmentos delimitados; nunca el resto de la descripcion.
  publicidad <- c(
    "\\b(?:en )?(?:somos )?rentahouse finca raiz\\b",
    "\\brentahouse (?:cajica )?vende\\b",
    "\\bpads (?:te presenta|tiene el gusto en ofrecer)\\b",
    "\\bcontamos con mas de 51 anos de experiencia en la comercializacion inmobiliaria\\b",
    "\\bdisponemos de una amplia oferta en cundinamarca y antioquia\\b",
    "\\bcontactanos y recibe informacion del inmueble que suenas\\b",
    "\\bpara mayor informacion\\b",
    "\\bcomunicarse con\\b",
    "\\bvisita nuestra pagina web\\b",
    "\\bwww pads com(?: co)?\\b",
    "\\bte interesa conocer mas de nuestro equipo\\b",
    "\\bsiguenos en instagram\\b",
    "\\bvisita nuestr[ao] siti[oa] web\\b",
    "\\bwww rentahousefincaraiz com co\\b",
    "\\brentahousefincariaz co\\b",
    "\\b(?:agenda|agende) (?:tu|su|una) cita\\b",
    "\\b(?:no dudes en )?contactanos\\b",
    "\\bllamanos\\b"
  )
  for (patron in publicidad) {
    texto <- stringr::str_replace_all(texto, patron, " ")
  }
  
  stringr::str_squish(texto)
}
