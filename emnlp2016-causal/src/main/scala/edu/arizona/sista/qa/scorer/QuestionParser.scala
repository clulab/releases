package edu.arizona.sista.qa.scorer

import collection.mutable.ArrayBuffer
import java.io.File
import nu.xom.Builder


/**
 * 
 * User: peter
 * Date: 3/19/13
 */
class QuestionParser(val debug: Boolean = false) {

  @inline
  def debugPrint(string: String) {
    if (debug) println(string)
  }

  // Constructor

  // ---------------------------------------------
  //    Methods
  // ---------------------------------------------
  def parse(xmlFile:String, moveSentOffset:Boolean = true): List[Question] = {
    // moveSentOffset -- Adjusts sentenceOffsets by -1 as they're read in, so that they're zero indexed.
    var sentOffsetDelta:Int = -1
    if (moveSentOffset == false) sentOffsetDelta = 0

    val parsedQuestions = new ArrayBuffer[Question]()                           // Array of Question/Answer pairs to build

    if (xmlFile == null) {
      throw new RuntimeException ("ERROR: QuestionParser.parse: filename passed is null.  exiting... ")
    }
    val filep = new File(xmlFile)
    if (!filep.exists()) {
      throw new RuntimeException ("ERROR: QuestionParser.parse: file not found. exiting... (filename: " + xmlFile + ")!")
    }

    debugPrint (" * QuestionParser: parse: started... ")
    debugPrint (" * QuestionParser: parse: parsing file (filename: " + xmlFile + ").")

    val b = new Builder()                                                        // XML Parser
    val doc = b.build(filep)
    val root = doc.getRootElement

    debugPrint (" * QuestionParser: parse: started... ")
    val questions = root.getChildElements("question")                            // get all questions
    debugPrint (" * QuestionParser: parse: total questions: " + questions.size)

    // Parse all question / multilple answer pairs
    // For each question
    for (i <- 0 until questions.size) {
      val qe = questions.get(i)
      val questionText = qe.getFirstChildElement("text").getValue               // Read the question text
      debugPrint (" Question [" + i + "] : " + questionText)

      val answers1 = qe.getChildElements("answers")                              // Get the list of "answer"s.  Note that
      val answers = answers1.get(0).getChildElements("answer")                   // the list of "answer"s is encapsulated in an "answers" tag (answers1)
      val goldAnswers = new ArrayBuffer[GoldAnswer]()                              // Temporary list of GoldAnswers for a given question
      debugPrint ("    Total Answers: " + answers.size)

      // For each of N possible answers to that question
      for (j <- 0 until answers.size) {
        val ae = answers.get(j)                                                  // get one answer element
        val answerText = ae.getFirstChildElement("justification").getValue    // read answer text
        val answerDocid = ae.getFirstChildElement("docid").getValue           // read document ID
        val sentences = ae.getFirstChildElement("sentences").getValue          // read sentence numbers
        val sentencesSplit = sentences.split(",")                               // Sentence numbers are read as a comma delimited list.
        val sentenceOffsets = new ArrayBuffer[Int]()                             // Unpack here and store as Array
        for (k <- 0 until sentencesSplit.size) {
          sentenceOffsets.append(sentencesSplit(k).trim.toInt + sentOffsetDelta)
        }

        debugPrint ( "    Answer[" + j + "]: " + answerText)

        val oneGoldAnswer = new GoldAnswer(answerText, answerDocid, sentenceOffsets.toArray)
        goldAnswers.append(oneGoldAnswer)                                       // Store one new GoldAnswer in GoldAnswer list
      }

      parsedQuestions.append (new Question(questionText, goldAnswers.toArray))         // Store parsed question as one Question / GoldAnswer-array pair
    }

    debugPrint (" * QuestionParser: parse: completed... ")

    parsedQuestions.toList                                                      // Return list of Q/A* pairs

  }

  def numQuestions(xmlFile:String):Int = {
    val parsedQuestions = new ArrayBuffer[Question]()                           // Array of Question/Answer pairs to build
    if (xmlFile == null) {
      throw new RuntimeException ("ERROR: QuestionParser.numQuestions: filename passed is null.  exiting... ")
    }
    val filep = new File(xmlFile)
    if (!filep.exists()) {
      throw new RuntimeException ("ERROR: QuestionParser.numQuestions: file not found. exiting... (filename: " + xmlFile + ")!")
    }

    val b = new Builder()                                                        // XML Parser
    val doc = b.build(filep)
    val root = doc.getRootElement

    val questions = root.getChildElements("question")                            // get all questions
    questions.size                                                               // return number of questions

  }


}
