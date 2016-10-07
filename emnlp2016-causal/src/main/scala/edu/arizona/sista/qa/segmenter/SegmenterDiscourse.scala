package edu.arizona.sista.qa.segmenter

import edu.arizona.sista.processors.Document
import collection.mutable.ArrayBuffer

/**
 * Breaks text into segments based on discourse connectives and cue phrases
 * User: peter
 * Date: 4/26/13
 */
class SegmenterDiscourse(mode:String) extends Segmenter {
  val discourseFinder = new FinderDiscourse(mode)

  def segment (doc:Document, sentStartOffset:Int, sentEndOffset:Int):List[Segment] = {
    val discourseSegs = discourseFinder.find(doc, sentStartOffset, sentEndOffset).toList
    val discoursePlusOtherSegs = discourseFinder.fillOtherSegments(discourseSegs)

    // return segments
    discoursePlusOtherSegs
  }

  def segmentWithEOS (doc:Document, sentStartOffset:Int, sentEndOffset:Int):List[Segment] = {
    // Fill "OTHER" segments stops on discourse boundaries, giving two (or more) "OTHERS" in place of one large "OTHER"
    val discourseSegs = discourseFinder.find(doc, sentStartOffset, sentEndOffset).toList
    val discoursePlusOtherSegs = discourseFinder.fillOtherSegmentsEOS(discourseSegs)

    // return segments
    discoursePlusOtherSegs
  }

}

