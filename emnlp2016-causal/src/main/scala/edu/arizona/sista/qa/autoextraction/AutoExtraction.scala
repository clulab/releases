package edu.arizona.sista.qa.autoextraction

import edu.arizona.sista.utils.StringUtils
import org.apache.lucene.document.Document
import edu.arizona.sista.qa.scorer.Question
import nu.xom._
import java.io.FileOutputStream
import org.slf4j.LoggerFactory

/**
 * User: peter
 * Date: 4/24/13
 */

// Storage class
class CandidateQuestion(val text:String, val doc:Document, val sentenceId:Int, val paraId:Int)


class AutoExtraction(indexDir:String) {

//  lazy val documentSerializer = new DocumentSerializer
//  val qStarts = List("how", "why")
//  val corefWords = List("it", "this", "they", "them", "his", "her", "he", "she")
//  val poorQualityAnswers = List("see appendix", "anotherpoorqualityanswer")
//  val poorQualityQuestions = List("this hypothesis", "anotherpoorqualityquestion")
//
//
//  def extractQA():List[Question] = {
//    val qgaPairs = new ArrayBuffer[Question]
//    val questions = findQuestions(qStarts)
//
//    for (i <- 0 until questions.size) {
//      var question = questions(i)
//      // Attempt to repair ambiguities using coreference, if required
//      if (needsCorefResolution(question.text)) question = resolveQuestionCoreference(question)
//
//      val (answer, poorAnswerFlag) = extractAnswerText(question)
//
//      // Discard questions flagged as poor quality
//      if (!poorAnswerFlag) {
//        qgaPairs.append(new Question(question.text, Array[GoldAnswer](answer)))
//      }
//
//    }
//
//    // Return automatically extracted question/GoldAnswer pairs
//    qgaPairs.toList
//  }
//
//  def findQuestions(qStartTerms:List[String]):Array[CandidateQuestion] = {
//    val candidateQuestions = new ArrayBuffer[CandidateQuestion]()
//    val docIterator = new IndexIterator(indexDir)
//
//    logger.debug (" * findQuestions: Started...")
//    // Traverse Lucene index
//    while (docIterator.hasNext) {
//      val doc = docIterator.next()
//      val annotation = documentSerializer.load(doc.getField(Indexer.ANNOTATION).stringValue())
//      val pids = StringUtils.toIntArray(doc.getField(Indexer.PARAGRAPH_IDS).stringValue())
//      val sents = getSentences(annotation)
//
//      // Search for questions within this documents sentences
//      for (i <- 0 until annotation.sentences.size) {
//        breakable {
//          // Step 1: Check if a given sentence contains a question mark
//          if (!sents(i).contains("?")) break()
//
//          // Step 2: Shallow checking -- check if the question has at least 4 words
//          val sent = annotation.sentences(i)
//          val words = sent.words
//          if (words.size <= 3) break()
//
//          // Step 3: Check if the question begins with one of the question Start Terms (e.g. how, why, if, etc)
//          val qStart = words(0).toLowerCase
//          if (!qStartTerms.contains(qStart)) break()
//
//          // Step 4: Check if the question meets basic structure requirements (at least one noun, one verb, etc)
//          // TODO: Implement Step 4
//
//          // Step 5: Check if the question contains one of the terms flagged as indicative of a poor quality question
//          for (pattern <- poorQualityQuestions) {
//            if (sents(i).toLowerCase.indexOf(pattern) > -1) break()
//          }
//
//          // Step 6: Shallow checks have passed, and we've identified a candidate question
//          candidateQuestions.append(new CandidateQuestion(sents(i), doc, i, pids(i)))
//        }
//      }
//    }
//    logger.debug (" * findQuestions: Completed.")
//
//    candidateQuestions.toArray
//  }
//
//
//  def resolveQuestionCoreference(question:CandidateQuestion):CandidateQuestion = {
//    val doc = question.doc
//    val annotation = documentSerializer.load(doc.getField(Indexer.ANNOTATION).stringValue())
//    val words = annotation.sentences(question.sentenceId).words
//    var subs = new ArrayBuffer[CorefSubstitution]
//
//    logger.debug ("-------------------------")
//    logger.debug (" Coreference Resolution: ")
//
//    // Search coreference chains
//    annotation.coreferenceChains.foreach(chains => {
//      for (chain <- chains.getChains) {
//        var active:Boolean = false;
//        // Step 1: Check to see if a given chain contains mentions from the question sentence
//        // Note: Ignore singleton chains
//        for (mention <- chain) {
//          if ((mention.sentenceIndex == question.sentenceId) && (chain.size > 1)) {
//            active = true
//          }
//        }
//
//        if (active == true) {
//          // Step 2: Find the longest coreference term in the list.  This will be what we subsitute
//          // each of the other mentions with.  (e.g. for the list("he", "his", "John Smith"), take
//          // "John Smith")
//          var longestMentionText = new Array[String](0)
//          var longestSize:Int = 0
//          for (mention <- chain) {
//            if ((mention.endOffset - mention.startOffset) > longestSize) {
//              longestMentionText = annotation.sentences(mention.sentenceIndex).words.slice(mention.startOffset, mention.endOffset)
//              longestSize = mention.endOffset - mention.startOffset
//            }
//          }
//
//          // Step 3: Assemble a list of coreference substitutions for the question for this coreference chain
//          for (mention <- chain) {
//            // Check if this mention is in the question sentence
//            if (mention.sentenceIndex == question.sentenceId) {
//              subs.append (new CorefSubstitution(mention.sentenceIndex,
//                                                mention.headIndex,
//                                                mention.startOffset,
//                                                mention.endOffset,
//                                                mention.chainId,
//                                                longestMentionText.clone()))
//            }
//          }
//        }
//
//      }
//    })
//
//    // Assemble new question
//    // Step 4: Sort coreference substitutions by starting offset
//    subs = subs.sortWith(_.startOffset < _.startOffset)
//
//    // Step 5: Construct new text using replacements
//    var idxText:Int = 0
//    var idxSubs:Int = 0
//    var newQuestion = new ArrayBuffer[String]
//    while (idxText < words.size) {
//      // Check for a substitution
//      if (idxSubs < subs.size) {
//        // Substitution -- append text from coreference resolution
//        if (idxText == subs(idxSubs).startOffset) {
//          for (word <- subs(idxSubs).subText) newQuestion.append(word)
//          idxText = subs(idxSubs).endOffset
//          idxSubs += 1
//        } else {
//          newQuestion.append(words(idxText))
//          idxText += 1
//        }
//      } else {
//        // No substitution -- append original question text
//        newQuestion.append(words(idxText))
//        idxText += 1
//      }
//    }
//
//    // Step 6: Create new question text from array of words
//    var newQuestionText:String = ""
//    for (x <- 0 until newQuestion.size) {
//      newQuestionText += newQuestion(x)
//      if (x < (newQuestion.size-1)) newQuestionText += " "
//    }
//
//    logger.debug ("*             Original Question: " + question.text)
//    logger.debug ("* Coreference Resolved Question: " + newQuestionText)
//    logger.debug ("-------------------------")
//
//    return new CandidateQuestion(newQuestionText, doc, question.sentenceId, question.paraId)
//  }
//
//
//  def extractAnswerText(question:CandidateQuestion):(GoldAnswer,Boolean) = {
//    val doc = question.doc
//    val annotation = documentSerializer.load(doc.getField(Indexer.ANNOTATION).stringValue())
//    val pids = StringUtils.toIntArray(doc.getField(Indexer.PARAGRAPH_IDS).stringValue())
//    val docid = doc.getField(Indexer.DOCID).stringValue()
//    val sents = getSentences(annotation)
//
//    var paraCount = 0
//    var paraId = question.paraId
//    var sentIdx = question.sentenceId
//    var poorAnswerFlag:Boolean = false
//
//    var shortAnswerThreshold: Int = 3
//
//    var os = new StringBuilder
//    // Extract answer
//    breakable {
//      while (sentIdx < (sents.size - 1)) {
//        sentIdx += 1
//
//        if (pids(sentIdx) == paraId) {
//          paraCount += 1
//        } else {
//          // Check We've found a large chunk of text for our answer, and have now switched to another paragraph.
//          // If so, stop collecting answer text.
//          if (paraCount >= shortAnswerThreshold) break()
//
//          // If we've added 5 sentences to the answer, and none of the paragraphs those sentences came from
//          // are very long, then we'll stop here
//          if ((sentIdx - question.sentenceId + 1) > 5) {
//            poorAnswerFlag = true                     // Flag potentially poor answer
//            break()
//          }
//
//          // New paragraph -- reset the paragraph counter
//          paraCount = 0
//          paraId = pids(sentIdx)
//        }
//
//        os.append(sents(sentIdx))                       // Build candidate answer
//
//        // Flag answers that contain patterns that suggest poor answer quality
//        for (pattern <- poorQualityAnswers) {
//          if (sents(sentIdx).toLowerCase().indexOf(pattern) > -1) poorAnswerFlag = true
//        }
//
//      }
//    }
//
//    // If the answer is very short, flag it as likely being of poor quality
//    if ((sents.size - sentIdx) < shortAnswerThreshold) poorAnswerFlag = true
//
//
//    val sIds = new ArrayBuffer[Int]()
//    for (i <- (question.sentenceId + 1) to sentIdx) sIds += i
//
//    // Return new Gold Answer
//    (new GoldAnswer(os.toString(), docid, sIds.toArray), poorAnswerFlag)
//
//  }


  def writeGoldQuestionsToXML(GoldQAPairs:List[Question], filename:String, sentenceOffset:Int = 0) {
    // SentenceOffset -- GoldQAPairs are stored starting at 0, but legacy files are written starting with 1 (and decremented by 1 when read in).
    // When writing a 0 indexed list, set the sentenceOffset to +1.
    // When writing a 1 indexed list, set the sentenceOffset to 0.

    val root = new Element("root")
    val xmlDoc = new nu.xom.Document(root)

    // Step 1: Assemble XML Document
    for (question <- GoldQAPairs) {
      val questionElem = new Element("question")
      val textElem = new Element("text")
      val answersElem = new Element("answers")

      root.appendChild(questionElem)                            // Root
      questionElem.appendChild(textElem)                        // <question>
      textElem.appendChild(question.text)                       // <text>
      questionElem.appendChild(answersElem)                     // <answers>

      for (answer <- question.goldAnswers) {
        val answerElem = new Element("answer")
        val justElem = new Element("justification")
        val docidElem = new Element("docid")
        val sentencesElem = new Element("sentences")
        val os = new StringBuilder

        justElem.appendChild(answer.text)                       // <justification>
        docidElem.appendChild(answer.docid)                     // <docid>

        for (i <- 0 until answer.sentenceOffsets.size) {        // <sentences>
          os.append(answer.sentenceOffsets(i) + sentenceOffset)
          if (i < (answer.sentenceOffsets.size-1)) os.append(", ")
        }
        sentencesElem.appendChild(os.toString())

        // Append to <answer> element                           // <answer>
        answerElem.appendChild(justElem)
        answerElem.appendChild(docidElem)
        answerElem.appendChild(sentencesElem)
        // Append <answer> to <answers> element
        answersElem.appendChild(answerElem)
      }
    }

    // Step 2: Write XML File
    val fileXML = new FileOutputStream(filename)
    val serializer = new Serializer(fileXML, "ISO-8859-1")
    serializer.setIndent(4)
    serializer.setMaxLength(80)
    serializer.write(xmlDoc)
    fileXML.close()

  }


//  private def getSentences(doc:edu.arizona.sista.processors.Document): Array[String] = {
//    val sents = new ArrayBuffer[String]
//    for (s <- doc.sentences) sents += s.getSentenceText()
//
//    sents.toArray
//  }
//
//  private def needsCorefResolution(text:String):Boolean = {
//    val words = text.split(" ")
//    for (word <- words) {
//      if (corefWords.contains(word.toLowerCase)) return true
//    }
//    false
//  }

}


// Entry point
object AutoExtraction {
  val logger = LoggerFactory.getLogger(classOf[AutoExtraction])

//  def main(args:Array[String]) {
//    val props = StringUtils.argsToProperties(args)
//    val indexDir = props.getProperty("index")
//    val AQG = new AutoExtraction(indexDir)
//
//    logger.info("* Starting Automatic Question/Answer Pair Extraction... ")
//    val autoGoldQuestions = AQG.extractQA()
//    for (question <- autoGoldQuestions) {
//      logger.info (" Question Text:" + question.text)
//      for (answer <- question.goldAnswers) {
//        logger.info (" * AnswerText:" + answer.text)
//        logger.info (" * AnswerSentOffsets:" + answer.sentenceOffsets.toList)
//        logger.info (" * AnswerDOCID:" + answer.docid)
//      }
//      logger.info ("")
//    }
//
//    logger.info ("* Exporting to XML file (filename = " + props.getProperty("autoqa_xml_out") + " )")
//    AQG.writeGoldQuestionsToXML(autoGoldQuestions, props.getProperty("autoqa_xml_out"))
//    logger.info ("* Completed... ")
//
//  }

}


// Storage Class
class CorefSubstitution (
  /** Index of the sentence containing this mentions; starts at 0 */
  val sentenceIndex:Int,
  /** Token index for the mention head word; starts at 0 */
  val headIndex:Int,
  /** Start token offset in the sentence; starts at 0 */
  val startOffset:Int,
  /** Offset of token immediately after this mention; starts at 0 */
  val endOffset:Int,
  /** Id of the coreference chain containing this mention; -1 if singleton mention */
  val chainId:Int,
  /** Substitution text */
  val subText:Array[String])

