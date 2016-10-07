package edu.arizona.sista.qa.retrieval

import edu.arizona.sista.processors.Document
import edu.arizona.sista.qa.index.{Indexer, TermFilter}
import edu.arizona.sista.struct.{Counters, Counter}

/**
 * Passage scorer using the plain old bag-of-words model
 * User: mihais
 * Date: 3/15/13
 */
class BagOfWordsPassageScorer(
  val termFilter:TermFilter,
  val indexDir:String,
  val query:Document) extends PassageScorer {

  lazy val queryVector = buildQueryVector(query, indexDir)

  def score(answer:AnswerCandidate):Double = {
    val answerVector = buildAnswerVector(answer)

    // the cosine does L2 normalization inside; that's why we use lnn and ltn above
    Counters.cosine(queryVector, answerVector)
  }

  def buildAnswerVector(answer:AnswerCandidate):Counter[String] = {
    val model = new IRModel(null)
    model.lnn(new Counter[String](termFilter.extractValidLemmas(
      answer.doc.annotation, answer.sentenceStart, answer.sentenceEnd)))
  }

  def buildQueryVector(query:Document, indexDir:String):Counter[String] = {
    val dfx = new DocFreqExtractorFromIndex(indexDir, Indexer.TEXT)
    val model = new IRModel(dfx)
    val ltn = model.ltn(new Counter[String](termFilter.extractValidLemmas(
      query, 0, query.sentences.length)))
    dfx.close()
    ltn
  }

}
