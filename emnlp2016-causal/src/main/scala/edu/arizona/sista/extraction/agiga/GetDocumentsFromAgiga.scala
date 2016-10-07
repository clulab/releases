package preprocessing.agiga

/**
  * Created by bsharp on 3/4/16.
  */
object GetDocumentsFromAgiga extends App {

  // Load in an agiga file, get the document from it
  val fn = "/data/nlp/corpora/agiga/data/xml/nyt_eng_201001.xml.gz"
  val doc = ProcessAgiga.agigaDocToDocument(fn)

  val causal = Set(" cause", "due to", "result", "therefore", " led to", "increased", "decreased")

  val possibleCausalSentences = for {
    s <- doc.sentences
    word <- causal
    if s.getSentenceText().contains(word)
  } yield s

  println (s"${possibleCausalSentences.length} possible causal sentences found")
  println ("")
  possibleCausalSentences.foreach(s => println(s.getSentenceText() + "\n"))

}
