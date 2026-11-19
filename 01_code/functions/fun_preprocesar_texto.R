# Limpieza conservadora para textos inmobiliarios de Properati (PS3).
# No modifica title/description: devuelve un vector de texto nuevo.
# Conserva numeros y negaciones (sin, no, con) para NLP y regex.

fun_preprocesar_texto <- function(texto) {
  texto[is.na(texto)] <- ""
  texto <- stringi::stri_trans_general(tolower(texto), "Latin-ASCII")
  texto <- stringr::str_squish(
    stringr::str_replace_all(texto, "[^a-z0-9]+", " ")
  )

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
    "\\bnios\\b" = "ninos",
    "\\bzona de lavandera\\b" = "zona de lavanderia",
    "\\bzona lavandera\\b" = "zona lavanderia",
    "\\bcuarto de lavandera\\b" = "cuarto de lavanderia"
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
