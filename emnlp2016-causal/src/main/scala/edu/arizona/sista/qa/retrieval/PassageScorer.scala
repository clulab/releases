package edu.arizona.sista.qa.retrieval

import edu.arizona.sista.processors.Document

/**
 * Generic interface for a passage scorer module
 * User: mihais
 * Date: 3/15/13
 */
trait PassageScorer {
  def score(answer:AnswerCandidate):Double
}
