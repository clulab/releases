package edu.arizona.sista.qa.retrieval

import collection.mutable.ListBuffer

/**
 * Constructs answer candidates using a sliding window of paragraphs
 * User: mihais
 * Date: 3/15/13
 */
class ParagraphAnswerBuilder (val size:Int) extends AnswerBuilder {
  def mkAnswerCandidates(documents: List[DocumentCandidate]):List[AnswerCandidate] = {
    val ab = new ListBuffer[AnswerCandidate]
    for (document <- documents) {
      var offset = 0
      var end = -1
      do {
        val (nextOffset, thisEnd) = nextBlock(document, offset)
        end = thisEnd
        if (end > offset) {
          ab += new AnswerCandidate(document, offset, end)
          offset = nextOffset
        }
      } while(end > 0)
    }
    ab.toList
  }

  def nextBlock(document:DocumentCandidate, offset:Int):(Int, Int) = {
    var parCounter = 0
    var crtOffset = offset
    var nextOffset = 0
    var endOffset = crtOffset + 1
    while(parCounter < size && endOffset > 0) {
      endOffset = findParagraphEnd(document, crtOffset)
      if (endOffset > 0) {
        if (parCounter == 0) nextOffset = endOffset
        crtOffset = endOffset
        parCounter += 1
      }
    }
    (nextOffset, endOffset)
  }

  def findParagraphEnd(document:DocumentCandidate, offset:Int):Int = {
    if (offset >= document.annotation.sentences.length)
      return -1

    var crtPar = document.paragraphs(offset)
    var crtOffset = offset

    while(crtOffset < document.annotation.sentences.length &&
      document.paragraphs(crtOffset) == crtPar)
      crtOffset += 1

    crtOffset
  }
}
