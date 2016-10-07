package edu.arizona.sista.qa.segmenter

import edu.arizona.sista.processors.Document
import collection.mutable.ArrayBuffer
import util.control.Breaks._
import org.slf4j.LoggerFactory
import FinderVerb.logger

/**
 * Locates chunks of verbs (and, optionally, adverbs) in a section of text
 * User: peter
 * Date: 4/26/13
 */

class FinderVerb extends Finder {

  def find(doc:Document, sentStartOffset:Int, sentEndOffset:Int):List[Segment] = {
    findVerbModeSelect(doc, sentStartOffset, sentEndOffset, 0)        // "Just report the last verb" mode
  }

  def findVerbModeSelect(doc:Document, sentStartOffset:Int, sentEndOffset:Int, mode:Int):List[Segment] = {
    // Mode: 1 returns the entire segment, 0 returns just the end VB* in large segments containing multiple RB*/VB*'s
    var verbSegs = new ArrayBuffer[Segment]
    var start:Tuple2[Int, Int] = (0, 0)
    var end:Tuple2[Int, Int] = (0, 0)
    var verbActive:Boolean = false
    var verbActiveCount:Int = 0
    var lastVerbIdx:Int = 0
    var sentStart = sentStartOffset
    var sentEnd = sentEndOffset

    // Bound checking
    if (sentStart < 0) sentStart = 0
    if (sentEnd > doc.sentences.size-1) sentEnd = doc.sentences.size-1

    for (sentIdx <- sentStart to sentEnd) {
      val sent = doc.sentences(sentIdx)
      val tags = sent.tags.getOrElse(new Array[String](0))

      for (tokenIdx <- 0 until tags.size) {
        val tag = tags(tokenIdx)

        // Check if the current tag is a verb or adverb
        if (tag.startsWith("VB") || tag.startsWith("RB")) {
          if (verbActive == false) {
            // found new verb
            start = (sentIdx, tokenIdx)
            verbActive = true
          } else {
            // continue counting the verbs
          }
          if (tag.startsWith("VB")) {
            verbActiveCount += 1
            lastVerbIdx = tokenIdx
          }
        }

        // If the current tag is not a verb/adverb, OR we're at the end of the sentence
        if ((!tag.startsWith("VB") && !tag.startsWith("RB")) || (tokenIdx == tags.size-1)) {
          if ((verbActive == true) && (verbActiveCount > 0)) {
            // end of a set of one or more VB* tags
            end = (sentIdx, tokenIdx)
            verbActive = false

            // append new segment to the list of verb segments
            if (mode == 0) {
              // Just return a segment with the last VB
              verbSegs.append (new Segment("VERB", doc, (sentIdx, lastVerbIdx), (sentIdx, lastVerbIdx+1)))
            } else if (mode == 1) {
              // Return entire segment
              verbSegs.append (new Segment("VERB", doc, start, end))
            }

          }
          verbActiveCount = 0
        }
      }
    }

    logger.debug(" * finderVerb(): List of Verb Segments: " + verbSegs.toList)

    // return verb segments
    verbSegs.toList
  }

}

object FinderVerb {
  val logger = LoggerFactory.getLogger(classOf[FinderVerb])
}