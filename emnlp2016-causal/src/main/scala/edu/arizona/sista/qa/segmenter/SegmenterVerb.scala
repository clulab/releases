package edu.arizona.sista.qa.segmenter

import edu.arizona.sista.processors.Document
import collection.mutable.ArrayBuffer

/**
 * Breaks a text into segments based on verb boundaries
 * User: peter
 * Date: 4/26/13
 */
class SegmenterVerb extends Segmenter {
  val verbFinder = new FinderVerb()

  def segment (doc:Document, sentStartOffset:Int, sentEndOffset:Int):List[Segment] = {
    val verbSegs = verbFinder.find(doc, sentStartOffset, sentEndOffset).toList
    val verbPlusOtherSegs = verbFinder.fillOtherSegments(verbSegs)

    // return segments
    verbPlusOtherSegs
  }

}
