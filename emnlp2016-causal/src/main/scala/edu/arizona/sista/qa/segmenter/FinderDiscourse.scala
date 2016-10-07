package edu.arizona.sista.qa.segmenter

import org.slf4j.LoggerFactory
import edu.arizona.sista.processors.Document
import collection.mutable.ArrayBuffer
import scala.Tuple2
import FinderDiscourse.logger

/**
 * Locates chunks of discourse connectives and cue phrases (largely from Daniel Marcu's PhD thesis) in a section of text
 * User: peter
 * Date: 4/30/13
 */

class FinderDiscourse(mode:String) extends Finder {
  // Mode: Selects between FULL, LIMITED, or SINGLE discourse connectives list.  FULL is the default.

  var cueTerms = Array("actually", "after", "again", "already", "also", "although",
    "and", "as", "back", "because", "because of", "before", "but", "by", "certainly",
    "clearly", "earlier", "either", "else", "especially", "even", "except", "finally",
    "fine", "first", "following", "for", "further", "given", "here", "however", "if",
    "in", "including", "instead", "just", "last", "later", "like", "merely", "next",
    "nor", "not", "now", "once", "only", "or", "particularly", "perhaps", "provided",
    "rather", "second", "simply", "since", "so", "soon", "still", "such", "suddenly",
    "that", "then", "therefore", "though", "thus", "too", "true", "until", "well",
    "when", "where", "whether", "which", "while", "who", "without", "yet")

  // Restricted sets of discourse connectives, for faster simulations during development
  if (mode == "MOSTFREQUENT") cueTerms = Array("and", "in", "that", "for", "if", "as", "not", "or", "but", "by", "which", "then", "so", "just", "like", "when")
  if (mode == "INFREQUENT") cueTerms = Array("given", "later", "especially", "although", "yet", "whether", "rather", "including", "perhaps", "finally", "except", "therefore", "provided")
  // Others (legacy)
  if (mode == "SMALL") cueTerms = Array("because", "in", "and", "that", "if", "then", "after", "when", "where", "whether", "which", "while", "who", "without", "yet")
  if (mode == "LIMITED") cueTerms = Array("because", "in", "and", "that", "if")
  if (mode == "SINGLE") cueTerms = Array("because")


  def find(doc:Document, sentStartOffset:Int, sentEndOffset:Int):List[Segment] = {
    // TODO: Modify to detect cue phrases with more than one word (e.g. "because of")
    var discourseSegs = new ArrayBuffer[Segment]
    var start:Tuple2[Int, Int] = (0, 0)
    var end:Tuple2[Int, Int] = (0, 0)
    var sentStart = sentStartOffset
    var sentEnd = sentEndOffset

    // Bound checking
    if (sentStart < 0) sentStart = 0
    if (sentEnd > doc.sentences.size-1) sentEnd = doc.sentences.size-1

    for (sentIdx <- sentStart to sentEnd) {
      val sent = doc.sentences(sentIdx)
      val words = sent.words

      for (tokenIdx <- 0 until words.size) {
        val word = words(tokenIdx).toLowerCase

        // Check if the current tag is a verb
        if (cueTerms.contains(word)) {
          start = (sentIdx, tokenIdx)
          end = (sentIdx, tokenIdx + 1)
          //discourseSegs.append (new Segment("DISCOURSE", doc, start, end))
          discourseSegs.append (new Segment(word, doc, start, end))       // use actual discourse connective as label
        }
      }
    }

    //##logger.debug(" * finderDiscourse(): List of Discourse Segments: " + discourseSegs.toList)

    // return discourse segments
    discourseSegs.toList
  }

}

object FinderDiscourse {
  val logger = LoggerFactory.getLogger(classOf[FinderDiscourse])
}