package edu.arizona.sista.qa.segmenter

import org.slf4j.LoggerFactory
import edu.arizona.sista.processors.Document
import collection.mutable.ArrayBuffer
import scala.Tuple2



/**
 * Super-class of the segment finders (FinderVerb, FinderDiscourse, etc).
 * User: peter
 * Date: 4/30/13
 */
trait Finder {

  def find(doc:Document, sentStartOffset:Int, sentEndOffset:Int):Iterable[Segment]

  def find(doc:Document):Iterable[Segment] = find(doc, 0, doc.sentences.size-1)

  /**
   * A FinderVerb/FinderDiscourse returns a list of verb or discourse segments.
   * This helper function returns a list that contains those original segments,
   * as well as segments labelled OTHER that are between those original segments.
   * e.g. Original verb segments: [VERB (0, 5) -> (0, 6)], [VERB (1, 2) -> (1, 4)],    end of doc (1, 10).
   * fillOtherSegments: [OTHER (0, 0) -> (0, 5)],  [VERB (0, 5) -> (0. 6)], [OTHER (0, 6]) -> (1, 2)], [VERB (1, 2) -> (1, 4)], [OTHER (1, 4) -> (1, 10)]
   * @param origSegs
   * @return
   */
  def fillOtherSegments (origSegs:List[Segment]):List[Segment] = {
    var newSegs = new ArrayBuffer[Segment]

    // Step 1: Check that the segments list is non empty
    if (origSegs.size == 0) return newSegs.toList
    val doc = origSegs(0).doc
    val endOfDoc = (doc.sentences.size-1, doc.sentences(doc.sentences.size-1).words.size)

    // Step 2: Check for the special case of the first segment beginning at the very start of the document
    if (origSegs(0).startOffset != (0, 0)) {
      newSegs.append(new Segment("OTHER", origSegs(0).doc, (0, 0), origSegs(0).startOffset) )
    }

    // Step 3: Create "OTHER" segments bridging existing segments
    for (i <- 0 until (origSegs.size - 1)) {
      val origSegA = origSegs(i)
      val origSegB = origSegs(i+1)

      // Add first, original segment
      newSegs.append(origSegA)

      if (origSegA.endOffset != origSegB.startOffset) {
        // Create new other segment
        newSegs.append (new Segment("OTHER", origSegA.doc, origSegA.endOffset, origSegB.startOffset) )
      }
    }

    // Step 4: Handle the end case
    val lastSeg = origSegs(origSegs.size-1)
    newSegs.append(lastSeg)
    if (lastSeg.endOffset != endOfDoc) newSegs.append (new Segment("OTHER", lastSeg.doc, lastSeg.endOffset, endOfDoc) )

    // return list of segments
    newSegs.toList
  }


  // TODO: Same as fillOtherSegments, but OTHER segments also stop on EOS boundaries.
  def fillOtherSegmentsEOS (origSegs:List[Segment]):List[Segment] = {
    var filledSegs = new ArrayBuffer[Segment]

    // Step 1: Check that the segments list is non empty
    if (origSegs.size == 0) return filledSegs.toList
    val doc = origSegs(0).doc

    // Step 2: Obtain OTHER segments
    filledSegs = filledSegs ++ fillOtherSegments(origSegs)

    // Step 3: Split "OTHER" tags that span multiple sentences
    var endNum = (filledSegs.size-1)
    var i = 0
    while (i < endNum) {
      val seg = filledSegs(i)

      if (seg.label == "OTHER") {
        // Check if the start and end sentence offsets are different
        if ((seg.startOffset._1 != seg.endOffset._1)) {
          // Because the endOffset is exclusive, skip over cases where the sentenceOffsets differ by 1, and the end tokenOffset is 0
          // e.g. (1, 3) -> (2, 0) skip
          //      (1, 3) -> (2, 5) split
          //      (1, 3) -> (3, 0) split
          if (!((seg.startOffset._1 == seg.endOffset._1 + 1) && (seg.endOffset._2 == 0))) {
            // Delete original segment
            filledSegs.remove(i)
            // Add two new segments
            val origEnd = seg.endOffset
            val newEnd:Tuple2[Int, Int] = (seg.startOffset._1 + 1, 0) //doc.sentences(seg.startOffset._1).size)
            filledSegs.insert(i, new Segment("OTHER", seg.doc, seg.startOffset, newEnd) )
            val newStart:Tuple2[Int, Int] = (seg.startOffset._1 + 1, 0)
            if (newStart != origEnd) {
              filledSegs.insert(i+1, new Segment("OTHER", seg.doc, newStart, origEnd) )
            }
            endNum += 1
          }
        }
      }

      i += 1
    }

    // return list of segments
    filledSegs.toList
  }

}
