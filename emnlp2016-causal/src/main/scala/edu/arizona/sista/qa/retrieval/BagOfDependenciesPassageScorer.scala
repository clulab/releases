package edu.arizona.sista.qa.retrieval

import edu.arizona.sista.qa.index.{Indexer, DependencyBuilder, TermFilter}
import edu.arizona.sista.processors.Document
import edu.arizona.sista.struct.{Counter, Counters}

/**
 *
 * User: mihais
 * Date: 4/4/13
 */
class BagOfDependenciesPassageScorer(
  val termFilter:TermFilter,
  val indexDir:String,
  val query:Document) extends PassageScorer {

  lazy val queryVector = buildQueryVector(query, indexDir)
  lazy val depBuilder = new DependencyBuilder(termFilter)

  def score(answer:AnswerCandidate):Double = {
    val answerVector = buildAnswerVector(answer)

    // the cosine does L2 normalization inside; that's why we use lnn and ltn above
    Counters.cosine(queryVector, answerVector)
  }

  def buildAnswerVector(answer:AnswerCandidate):Counter[String] = {
    val model = new IRModel(null)
    val deps = depBuilder.buildDependencies(
      answer.doc.annotation,
      answer.sentenceStart,
      answer.sentenceEnd)
    model.lnn(new Counter[String](deps.map(x => x.toString)))
  }

  def buildQueryVector(query:Document, indexDir:String):Counter[String] = {
    val dfx = new DocFreqExtractorFromIndex(indexDir, Indexer.DEPENDENCIES)
    val model = new IRModel(dfx)
    val deps = depBuilder.buildDependencies(
      query, 0, query.sentences.length)
    val ltn = model.ltn(new Counter[String](deps.map(x => x.toString)))
    dfx.close()
    ltn
  }
}
