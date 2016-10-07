package edu.arizona.sista.qa.retrieval

import collection.mutable.ListBuffer
import java.lang.StringBuilder
import edu.arizona.sista.processors.Document

/**
 * Stores one candidate answer, as a contiguous sequence of sentences
 * User: mihais
 * Date: 3/14/13
 */
class AnswerCandidate (
  val doc:DocumentCandidate,

  /** Index of the first sentence that belongs to the answer (inclusive); indexed from 0 */
  val sentenceStart:Int,

  /** Index after the last sentence that belongs to the answer (exclusive); indexed from 0 */
  val sentenceEnd:Int,

  /** Score of this text fragment, as computed by a passage scorer module */
  var answerScore:Double,

  /** Overall answer score, combining document and answer scores */
  var score:Double) extends Serializable {

  def this(d:DocumentCandidate, ss:Int, se:Int) = this(d, ss, se, 0.0, 0.0)
  override def clone = new AnswerCandidate(doc, sentenceStart, sentenceEnd, answerScore, score)

  val sentences = mkSentences(sentenceStart, sentenceEnd)

  val annotation = doc.annotation

  def mkSentences(start:Int, end:Int):List[Int] = {
    val b = new ListBuffer[Int]
    for (i <- start until end) {
      b += i
    }
    b.toList
  }

  def setScore(as:Double, docWeight:Double) {
    answerScore = as
    score = doc.docScore * docWeight + (1.0 - docWeight) * answerScore
  }

  def getText:String = {
    val os = new StringBuilder
    for (s <- sentences) {
      for(i <- 0 until doc.annotation.sentences(s).words.length) {
        os.append(doc.annotation.sentences(s).words(i) + " ")
      }
    }
    os.toString
  }

  def getSentencesString:String = {
    var os = new StringBuilder
    for (i <- 0 until sentences.size) {        // <sentences>
      os.append (sentences(i).toString)
      if (i < (sentences.size - 1)) os.append(",")
    }
    os.toString
  }

  override def equals(other:Any):Boolean = {
    other match {
      case that:AnswerCandidate => (doc == that.doc && sentences == that.sentences)
      case _ => false
    }
  }

  override def hashCode = {
    41 * (41 + doc.hashCode) + sentences.hashCode
  }

  override def toString:String = {
    val os = new StringBuilder
    os.append("docid: " + doc.docid + " sentences: " + getSentencesString + " score: " + score + "\n")
    os.append("text: " + getText)
    os.append("\n")
    os.toString
  }
}
