package edu.arizona.sista.qa.segmenter

import edu.arizona.sista.qa.retrieval.AnswerCandidate

/**
 * Generic interface to a segment matcher/scorer module
 * User: peter
 * Date: 4/26/13
 */
trait SegmentMatcher {
  def score(segA:Segment, segB:Segment):Double
}
