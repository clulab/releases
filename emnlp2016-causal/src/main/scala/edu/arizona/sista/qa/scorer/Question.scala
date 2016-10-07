package edu.arizona.sista.qa.scorer

import collection.mutable.ArrayBuffer
import edu.arizona.sista.qa.AlignedText

// TODO: Convert ArrayBuffer to Array
import java.lang.StringBuilder

/**
 * Stores one question and its gold answers
 * The QuestionParser class reads these objects from the XML file
 * User: peter
 * Date: 3/19/13
 */
class Question (
  val text:String,                            // Question: e.g. "How does DNA replicate?"
  val goldAnswers:Array[GoldAnswer])   // Array of one or more gold answers. e.g. "DNA begins replicating by splitting into..."
  extends AlignedText {

  // Constructor

  // ---------------------------------------------
  //    Methods
  // ---------------------------------------------

  override def toString() : String = {
    val outstring = new StringBuilder();
    val size = goldAnswers.length;

    // Question text
    outstring.append("Question: " + text + "\n");

    // Gold Answer text
    if (size > 0) {
      for (i <- 0 to size-1) {
        outstring.append(" GoldAnswers[" + i + "] : " + goldAnswers(i) + "\n");
      }
    }

    return outstring.toString;

  }

  def questionText: String = text

  def answersTexts: Iterable[String] = goldAnswers.map(_.text)
}
