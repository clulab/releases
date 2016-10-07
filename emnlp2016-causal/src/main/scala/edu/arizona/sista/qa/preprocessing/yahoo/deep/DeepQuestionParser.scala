package edu.arizona.sista.qa.preprocessing.yahoo.deep

import collection.mutable.ArrayBuffer
import java.io.{PrintWriter, File}
import nu.xom.Builder
import edu.arizona.sista.qa.scorer.{HistogramString, Question}
import edu.arizona.sista.utils.StringUtils
import org.slf4j.LoggerFactory
import DeepQuestionParser.logger
import edu.arizona.sista.processors.Processor
import edu.arizona.sista.processors.corenlp.CoreNLPProcessor
import edu.arizona.sista.processors.Document
import scala.util.control.Breaks._


/**
 * Created with IntelliJ IDEA.
 * User: peter
 * Date: 12/2/13
 */
class DeepQuestionParser {

  lazy val processor = new CoreNLPProcessor()


  // Fast inexpensive partial annotation
  def mkPartialAnnotation(text:String):Document = {
    val doc = processor.mkDocument(text)
    processor.tagPartsOfSpeech(doc)
    processor.lemmatize(doc)
    doc.clear()
    doc
  }

  def parse(xmlFile: String): Array[CQAQuestion] = {

    val parsedQuestions = new ArrayBuffer[CQAQuestion]() // Array of Question/Answer pairs to build

    if (xmlFile == null) {
      throw new RuntimeException("ERROR: DeepQuestionParser.parse: filename passed is null.  exiting... ")
    }
    val filep = new File(xmlFile)
    if (!filep.exists()) {
      throw new RuntimeException("ERROR: DeepQuestionParser.parse: file not found. exiting... (filename: " + xmlFile + ")!")
    }

    logger.info(" * DeepQuestionParser: parse: started... ")
    logger.info(" * DeepQuestionParser: parse: parsing file (filename: " + xmlFile + ").")

    val b = new Builder() // XML Parser
    val doc = b.build(filep)
    val root = doc.getRootElement

    val questions = root.getChildElements("vespaadd") // get all questions
    println(" * DeepQuestionParser: parse: total questions: " + questions.size)

    // Parse all question / multiple answer pairs
    // For each question
    for (i <- 0 until questions.size) {
      breakable {
        val qe = questions.get(i)
        val qe1 = qe.getChildElements("document")

        //Practice safe parsing...

        //if (qe1.get(0).getFirstChildElement("uri") == null) break()
        //val uri = qe1.get(0).getFirstChildElement("uri").getValue
        val uri = safeGetValue(qe1.get(0).getFirstChildElement("uri"), "UNSPECIFIED")

        //break if we don't have a uri
        if (uri == "UNSPECIFIED") {
          logger.info("bad uri for question [" + i.toString + "]")
          break()
        }

        val subject = safeGetValue(qe1.get(0).getFirstChildElement("subject"), "UNSPECIFIED")
        val content = safeGetValue(qe1.get(0).getFirstChildElement("content"), "UNSPECIFIED")

        //see if we have content...
        if (content == "UNSPECIFIED") logger.info("no content for uri: " + uri)

        val bestAnswer = safeGetValue(qe1.get(0).getFirstChildElement("bestanswer"), "UNSPECIFIED")
        // make sure we have a best answer...
        if (bestAnswer == "UNSPECIFIED") {
          logger.info("no best answer for uri: " + uri)
          break()
        }
        // make sure we have an answer count
        var answercount = safeGetValue(qe1.get(0).getFirstChildElement("answer_count"), "UNSPECIFIED")

        // Other relevant metrics included in Yahoo! Answers file
        val category_broad = safeGetValue(qe1.get(0).getFirstChildElement("cat"), "UNSPECIFIED")
        val category_specific = safeGetValue(qe1.get(0).getFirstChildElement("maincat"), "UNSPECIFIED")
        val quality = safeGetValue(qe1.get(0).getFirstChildElement("qa_quality"), "UNSPECIFIED")
        // Extract answers
        val answerArray = new ArrayBuffer[CQAAnswer]()
        val ae = qe1.get(0).getChildElements("nbestanswers")
        val answers = ae.get(0).getChildElements("answer_item")

        // check for answercount
        if (answercount == "UNSPECIFIED") {
          // count it the other way...
          answercount = answers.size.toString
        }

        // Display question info
        logger.info("URI: " + uri + " \tAnswer Count: " + answercount + " \tCategory: " + category_broad + " \tQuestion: " + subject)

        for (j <- 0 until answers.size) {
          val answer = answers.get(j)
          var isGold: Boolean = false
          if (bestAnswer.compareTo(answer.getValue) == 0) {
            isGold = true
          }

          val docid = uri + "_A" + j.toString

          //create partial annotation of answer
          val annotation = mkPartialAnnotation(answer.getValue)
          val answerElement = new CQAAnswer(answer.getValue, docid, annotation, isGold)

          answerArray.append(answerElement)
        }

        // Add one parsed CQA question to the list
        val question = new CQAQuestion(subject, uri, category_broad, quality, answerArray.toArray)
        parsedQuestions.append(question)
      }

    }

    logger.info(" * DeepQuestionParser: parse: completed... ")

    parsedQuestions.toArray // Return array of Q/A* pairs
  }

  // Safe wrapper for returning XML fields that may or may not exist
  def safeGetValue(element:nu.xom.Element, defaultValue:String):String = {
    if (element == null) return defaultValue
    return element.getValue
  }

  def getWithMinimumAnswers(in:Array[CQAQuestion], minCandidates:Int):Array[CQAQuestion] = {
    val out = new ArrayBuffer[CQAQuestion]()
    for (question <- in) {
      if (question.answers.size >= minCandidates) {
        out.append(question)                            // note: shallow reference copy
      }
    }
    out.toArray
  }

  def getCausal(in:Array[CQAQuestion]):Array[CQAQuestion] = {
    val out = new ArrayBuffer[CQAQuestion]()
    for (question <- in) {
      val qText = question.questionText
      // Heuristics:
      if (qText.toLowerCase.startsWith("what causes")) out.append(question)
      if (qText.toLowerCase.startsWith("what caused")) out.append(question)
      if (qText.toLowerCase.startsWith("what can cause")) out.append(question)
      if (qText.toLowerCase.startsWith("what could cause")) out.append(question)
      if (qText.toLowerCase.startsWith("what would cause")) out.append(question)
      if (qText.toLowerCase.startsWith("what might cause")) out.append(question)
      if (qText.toLowerCase.startsWith("what may cause")) out.append(question)
      if (qText.toLowerCase.startsWith("what is cause")) out.append(question)
      if (qText.toLowerCase.startsWith("what are cause")) out.append(question)
      if (qText.toLowerCase.startsWith("what is the cause")) out.append(question)
      if (qText.toLowerCase.startsWith("what are the causes")) out.append(question)
      if (qText.toLowerCase.startsWith("what result")) out.append(question)
      if (qText.toLowerCase.startsWith("what can result")) out.append(question)
      if (qText.toLowerCase.startsWith("what are result")) out.append(question)
      if (qText.toLowerCase.startsWith("what are the result")) out.append(question)
      if (qText.toLowerCase.startsWith("what is the result")) out.append(question)
      if (qText.toLowerCase.startsWith("what is result")) out.append(question)
      if (qText.toLowerCase.startsWith("what are result")) out.append(question)
      if (qText.toLowerCase.startsWith("what is effect")) out.append(question)
      if (qText.toLowerCase.startsWith("what are effect")) out.append(question)
      if (qText.toLowerCase.startsWith("what is affect")) out.append(question)
      if (qText.toLowerCase.startsWith("what are affect")) out.append(question)
      if (qText.toLowerCase.startsWith("what is the effect")) out.append(question)
      if (qText.toLowerCase.startsWith("what are the effects")) out.append(question)
      if (qText.toLowerCase.startsWith("what affects")) out.append(question)
      if (qText.toLowerCase.startsWith("what effects")) out.append(question)
    }
    out.toArray
  }

  def numQuestions(xmlFile:String):Int = {
    val parsedQuestions = new ArrayBuffer[Question]()                           // Array of Question/Answer pairs to build
    if (xmlFile == null) {
      throw new RuntimeException ("ERROR: DeepQuestionParser.numQuestions: filename passed is null.  exiting... ")
    }
    val filep = new File(xmlFile)
    if (!filep.exists()) {
      throw new RuntimeException ("ERROR: DeepQuestionParser.numQuestions: file not found. exiting... (filename: " + xmlFile + ")!")
    }

    val b = new Builder()                                                        // XML Parser
    val doc = b.build(filep)
    val root = doc.getRootElement

    val questions = root.getChildElements("vespaadd")                            // get all questions
    questions.size                                                               // return number of questions

  }

  def generateCategoryHistogram(questions:Array[CQAQuestion]) {
    val histogram = new HistogramString("Question Categories")
    for (question <- questions) {
      histogram.addData(question.category)
    }

    println ("")
    //histogram.display(new PrintWriter("/dev/null"), 30)
    histogram.display(new PrintWriter("histogram_categories.txt"))
  }

}


object DeepQuestionParser {
  val logger = LoggerFactory.getLogger(classOf[DeepQuestionParser])

  def main(args:Array[String]) {
    val props = StringUtils.argsToProperties(args)
    val parser = new DeepQuestionParser

    //val xmlFile:String = "/home/peter/corpora/how_yahoo_answers/xml/deep/deep.xml"
    val xmlFile:String = "/data1/nlp/corpora/YahooAnswers/FullOct2007/FullOct2007.xml"
    val questions = parser.parse(xmlFile)
    val numQuestions = parser.numQuestions(xmlFile)
    parser.generateCategoryHistogram(questions)
    // Generate histogram of question types


    println ("Number of questions:" + numQuestions)
  }
}

