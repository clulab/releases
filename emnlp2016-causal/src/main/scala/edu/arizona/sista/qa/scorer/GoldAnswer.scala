package edu.arizona.sista.qa.scorer

/**
 * Stores one gold answer for a given question
 * User: peter
 * Date: 3/19/13
 */
class GoldAnswer (
  val text:String,           // Answer text
  val docid:String,          // Document ID

  /**
   * Offsets of the sentences that contain the answer text
   * Offsets start at 0, not at 1 (as they are stored in the gold file)
   * The QuestionParser class manages the above transformation
   */
  val sentenceOffsets:Array[Int]) {

  // Constructor

  // ---------------------------------------------
  //    Methods
  // ---------------------------------------------

  override def toString: String = {
    var outstring = new StringBuilder

    // Gold Answer Description
    outstring.append("Answer: " + text + "\n")
    outstring.append("DocID: " + docid + "\n")
    outstring.append("Sentence Offsets:" + sentenceOffsets.toList + "\n")

    outstring.toString()
  }

}
