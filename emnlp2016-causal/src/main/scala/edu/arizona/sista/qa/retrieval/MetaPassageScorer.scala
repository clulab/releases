package edu.arizona.sista.qa.retrieval

import edu.arizona.sista.qa.index.TermFilter
import edu.arizona.sista.processors.Document

/**
 *
 * User: mihais
 * Date: 4/4/13
 */
class MetaPassageScorer(
  val termFilter:TermFilter,
  val indexDir:String,
  val query:Document,
  val syntaxWeight:Double) extends PassageScorer {

  lazy val bowScorer = new BagOfWordsPassageScorer(
    termFilter,
    indexDir,
    query)

  lazy val bodScorer = new BagOfDependenciesPassageScorer(
    termFilter,
    indexDir,
    query)

  def score(answer:AnswerCandidate):Double = {
    val bowScore = bowScorer.score(answer)
    val bodScore = bodScorer.score(answer)
    (bodScore * syntaxWeight + (1 - syntaxWeight) * bowScore)
  }
}
