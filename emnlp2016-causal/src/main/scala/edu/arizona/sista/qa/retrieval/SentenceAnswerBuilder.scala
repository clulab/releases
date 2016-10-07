package edu.arizona.sista.qa.retrieval

import collection.mutable.ListBuffer

/**
 * Constructs answer candidates using a sliding window of sentences
 * User: mihais
 * Date: 3/15/13
 */
class SentenceAnswerBuilder (val size:Int) extends AnswerBuilder {
  def mkAnswerCandidates(documents: List[DocumentCandidate]):List[AnswerCandidate] = {
    val ab = new ListBuffer[AnswerCandidate]
    for(document <- documents) {
      var offset = 0
      if(size > document.annotation.sentences.length) {
        val a = new AnswerCandidate(document, 0, document.annotation.sentences.length)
        ab += a
      } else {
        while (offset <= document.annotation.sentences.length - size) {
          val a = new AnswerCandidate(document, offset, offset + size)
          ab += a
          offset += 1
        }
      }
    }
    ab.toList
  }
}
