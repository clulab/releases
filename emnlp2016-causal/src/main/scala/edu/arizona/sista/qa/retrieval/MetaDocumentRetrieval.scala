package edu.arizona.sista.qa.retrieval

import edu.arizona.sista.qa.index.TermFilter
import edu.arizona.sista.processors.Document

/**
 * Meta IR engine that combines multiple indices
 * User: mihais
 * Date: 4/4/13
 */
class MetaDocumentRetrieval(
  val indexDir:String,
  val maxDocCount:Int,
  val termFilter:TermFilter,
  val question:Document,
  val syntaxWeight:Double) extends DocumentRetrieval {

  lazy val bowRetriever =
    new BagOfWordsDocumentRetrieval(
      indexDir,
      maxDocCount,
      termFilter,
      question)

  lazy val bodRetriever =
    new BagOfDependenciesDocumentRetrieval(
      indexDir,
      maxDocCount,
      termFilter,
      question)

  def retrieve:List[DocumentCandidate] = {
    val bows = toSelfMap(bowRetriever.retrieve)
    val bods = toSelfMap(bodRetriever.retrieve)

    val result = List.newBuilder[DocumentCandidate]
    for(bow <- bows.keys) {
      val bod = bods.get(bow)
      var score:Double = (1.0 - syntaxWeight) * bow.docScore
      bod.foreach(d => {
        score += syntaxWeight * d.docScore
      })
      result += new DocumentCandidate(
        bow.docid,
        bow.annotation,
        bow.paragraphs,
        score)
    }
    for(bod <- bods.keys) {
      if(! bows.contains(bod)) {
        val score = syntaxWeight * bod.docScore
        result += new DocumentCandidate(
          bod.docid,
          bod.annotation,
          bod.paragraphs,
          score)
      }
    }

    val r = result.result()
    r.slice(0, scala.math.min(maxDocCount, r.size))
  }

  def toSelfMap[T](l:Iterable[T]):Map[T,T] = {
    val m = Map.newBuilder[T, T]
    for(e <- l) {
      m += e -> e
    }
    m.result()
  }
}
