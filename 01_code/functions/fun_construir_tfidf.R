# Construir TF-IDF con unigramas y bigramas.
# Se ajusta SOLO con texto_train; sirve también para train/validacion en CV.
fun_construir_tfidf <- function(texto_train, texto_nuevo,
                               min_docs = 5L, max_terminos = 12000L) {
  texto_train <- fun_preprocesar_texto(texto_train)
  texto_nuevo <- fun_preprocesar_texto(texto_nuevo)

  # No quitamos stopwords: "sin", "no" y "con" importan para las negaciones.
  it_train <- text2vec::itoken(texto_train, progressbar = FALSE)
  vocabulario <- text2vec::create_vocabulary(
    it_train, ngram = c(1L, 2L), sep_ngram = "_"
  )
  vocabulario <- text2vec::prune_vocabulary(
    vocabulario, doc_count_min = min_docs,
    doc_proportion_max = 0.90, vocab_term_max = max_terminos
  )
  vectorizador <- text2vec::vocab_vectorizer(vocabulario)

  # Mismas columnas para ambos conjuntos.
  matriz_train <- text2vec::create_dtm(
    text2vec::itoken(texto_train, progressbar = FALSE), vectorizador
  )
  matriz_nuevo <- text2vec::create_dtm(
    text2vec::itoken(texto_nuevo, progressbar = FALSE), vectorizador
  )

  tfidf <- text2vec::TfIdf$new(norm = "l2", sublinear_tf = TRUE)
  matriz_train <- tfidf$fit_transform(matriz_train)
  matriz_nuevo <- tfidf$transform(matriz_nuevo)

  list(train = matriz_train, nuevo = matriz_nuevo,
       vocabulario = vocabulario)
}
