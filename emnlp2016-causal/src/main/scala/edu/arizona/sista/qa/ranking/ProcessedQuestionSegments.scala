package edu.arizona.sista.qa.ranking

import edu.arizona.sista.qa.segmenter.Segment
import edu.arizona.sista.struct.Counter

/**
 * Storage class for a question broken into one or more segments
 * User: peter
 * Date: 7/30/13
 */

class ProcessedQuestionSegments (
                                  val questionType:String,
                                  val segments:List[Segment],
                                  val segmentsLTN:List[Counter[String]]) extends ProcessedQuestion {

  val annotation = segments(0).doc
}
