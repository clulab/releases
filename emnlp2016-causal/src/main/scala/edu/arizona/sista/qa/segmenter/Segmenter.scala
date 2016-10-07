package edu.arizona.sista.qa.segmenter

import edu.arizona.sista.processors.Document
import collection.mutable.ArrayBuffer

/**
 * Generic interface to a text segmenter module
 * User: peter
 * Date: 4/26/13
 */
trait Segmenter {
  def segment (doc:Document, sentStartOffset:Int, sentEndOffset:Int):List[Segment]

  def segment (doc:Document):List[Segment] = segment(doc, 0, doc.sentences.size-1)

  def returnSegmentsWithLabel(inSegs:List[Segment], label:String):List[Segment] = {
    var outSegs = new ArrayBuffer[Segment]
    for (seg <- inSegs) {
      if (seg.label == label) outSegs.append(seg)
    }
    outSegs.toList
  }
}


