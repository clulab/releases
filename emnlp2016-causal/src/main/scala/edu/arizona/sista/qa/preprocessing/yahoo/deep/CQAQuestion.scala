package edu.arizona.sista.qa.preprocessing.yahoo.deep

import edu.arizona.sista.processors.Document
import edu.arizona.sista.qa.scorer.{GoldAnswer, Question}
import collection.mutable.ArrayBuffer
import edu.arizona.sista.qa.AlignedText

/**
 * Storage class
 * User: peter
 * Date: 12/2/13
 */

// Storage classes
class CQAQuestion(val text:String,            // Question text
                  val docid:String,           // Question document ID
                  val category:String,        // Question category
                  val quality:String,         // Question quality
                  val answers:Array[CQAAnswer],   // list of answers
                  val content: Option[String] = None
                   ) extends AlignedText {

  def toQuestion(onlyGold:Boolean = true, includeContent: Boolean = false):Question = {
    val goldAnswers = new ArrayBuffer[GoldAnswer]()
    for (answer <- answers) {
      if (answer.gold == true) {
        goldAnswers.append (new GoldAnswer(answer.text, answer.docid, mkSentences(answer)))
      } else if (onlyGold == false) {
        goldAnswers.append (new GoldAnswer(answer.text, answer.docid, mkSentences(answer)))
      }
    }
    new Question(questionText(includeContent), goldAnswers.toArray)
  }

  def mkSentences(answer:CQAAnswer):Array[Int] = {
    if (answer.annotation == null) throw new RuntimeException ("CQAQuestion.mkSentences -- cannot create sentences array, no annotation present")
    val sent = new Array[Int](answer.annotation.sentences.size)
    for (i <- 0 until sent.size) {
      sent(i) = i
    }
    sent
  }

  def goldAnswers = answers.filter(_.gold)

  def questionText(includeContent: Boolean = false): String =
    if (includeContent && content.isDefined)
      text + " " + content.get
    else
      text

  def questionText: String = questionText(false)

  def answersTexts: Iterable[String] = answers.filter(_.gold).map(_.text)
}

class CQAAnswer(val text:String,              // Answer text
                val docid:String,             // Document ID
                var annotation:Document,      // Annotation
                val gold:Boolean = false)     // Is this a gold answer?
