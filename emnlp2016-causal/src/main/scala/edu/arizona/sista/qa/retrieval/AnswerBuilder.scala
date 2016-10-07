package edu.arizona.sista.qa.retrieval

/**
 * Constructs answer candidates from documents retrieved from IR
 * User: mihais
 * Date: 3/15/13
 */
trait AnswerBuilder {
  def mkAnswerCandidates(documents: List[DocumentCandidate]):List[AnswerCandidate]
}
