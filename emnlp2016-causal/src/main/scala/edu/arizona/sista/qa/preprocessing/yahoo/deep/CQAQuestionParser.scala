package edu.arizona.sista.qa.preprocessing.yahoo.deep

import java.util.Properties

import edu.arizona.sista.qa.translation.FreeTextAlignmentUtils

import collection.mutable.ArrayBuffer
import java.io._
import nu.xom.{Serializer, Element, Builder}
import edu.arizona.sista.qa.scorer.Question
import org.slf4j.LoggerFactory
import edu.arizona.sista.utils.StringUtils
import CQAQuestionParser.logger
import edu.arizona.sista.processors.{DocumentSerializer, Processor, Document}
import edu.arizona.sista.processors.corenlp.CoreNLPProcessor
import edu.arizona.sista.processors.DocumentSerializer._
import edu.arizona.sista.qa.autoextraction.AutoExtraction
import scala.concurrent.forkjoin.ForkJoinPool
import scala.collection.parallel.ForkJoinTaskSupport
import scala.util.Random

/**
 * Load and save CQAQuestion arrays in XML format
 * User: peter
 * Date: 12/2/13
 */
class CQAQuestionParser(val nThreads: Option[Int] = None) {
  // Constructor
  lazy val queryProcessor:Processor = new CoreNLPProcessor()
  lazy val documentSerializer = new DocumentSerializer

  // For compatibility with QuestionParser
  def parse(xmlFile:String):Array[Question] = {
    val CQAQuestions = load(xmlFile, true)
    CQAtoQuestions(CQAQuestions)
  }

  // Convert CQAQuestions array to Questions array
  def CQAtoQuestions(in:Array[CQAQuestion], onlyGold:Boolean = true):Array[Question] = {
    val out = new ArrayBuffer[Question]()
    for (cqaQuestion <- in) {
      out.append (cqaQuestion.toQuestion(onlyGold))
    }
    return out.toArray
  }

  def load(xmlFile:String, annotate:Boolean = true): Array[CQAQuestion] = {

    if (xmlFile == null) {
      throw new RuntimeException ("ERROR: CQAQuestionParser.parse: filename passed is null.  exiting... ")
    }
    val filep = new File(xmlFile)
    if (!filep.exists()) {
      throw new RuntimeException ("ERROR: CQAQuestionParser.parse: file not found. exiting... (filename: " + xmlFile + ")!")
    }

    logger.debug (" * CQAQuestionParser: parse: started... ")
    logger.info (" * CQAQuestionParser: parse: parsing file (filename: " + xmlFile + ").")

    if (annotate == false) {
      logger.debug ("WARNING: Annotation cannot currently be read successfully from CQA file. ")
    }

    val b = new Builder()                                                        // XML Parser
    val doc = b.build(filep)
    val root = doc.getRootElement

    println (" * CQAQuestionParser: parse: started... ")
    val questions = root.getChildElements("question")                            // get all questions
    println (" * CQAQuestionParser: parse: total questions: " + questions.size)

    def parseQuestion(i: Int) = {
      val qe = questions.get(i)
      val questionText = qe.getFirstChildElement("text").getValue                 // Read the question text
      val questionDocid = qe.getFirstChildElement("docid").getValue               // Read the question docid
      val questionQuality = qe.getFirstChildElement("quality_score").getValue     // Read the question quality score
      val questionCategory = qe.getFirstChildElement("category").getValue         // Read the question category

      val questionContent = optGetValue(qe.getFirstChildElement("content"))

      val answers1 = qe.getChildElements("answers")                               // Get the list of "answer"s.  Note that
      val answers = answers1.get(0).getChildElements("answer")                    // the list of "answer"s is encapsulated in an "answers" tag (answers1)

      val answerArray = new ArrayBuffer[CQAAnswer]
      // For each of N possible answers to that question
      for (j <- 0 until answers.size) {
        val ae = answers.get(j)
        val answerText = ae.getFirstChildElement("text").getValue
        val answerDocid = ae.getFirstChildElement("docid").getValue               // Answer-specific document ID (usually question ID w/A_n appended)
        val answerGold = ae.getFirstChildElement("gold").getValue
        val answerAnnotation = ae.getFirstChildElement("annotation").getValue

        var isGold:Boolean = false
        if (answerGold.toLowerCase == "true") {
          isGold = true
        }

        if (annotate) {
          // Perform annotation
          answerArray.append ( new CQAAnswer(answerText, answerDocid, mkPartialAnnotation(answerText), isGold) )
        } else {
          if (answerAnnotation.compareTo("null") != 0) {
            // Read annotation from file

            // TODO: Annotation cannot currently be read in successfully.  The XML file removes some of the formatting tags (like the tab delimiters between fields).
            //val annotation = documentSerializer.load( answerAnnotation )
            //answerArray.append ( new CQAAnswer(answerText, answerDocid, annotation, isGold) )
            answerArray.append ( new CQAAnswer(answerText, answerDocid, null, isGold) )
          } else {
            // Do not annotate, and store null reference for annotation
            answerArray.append ( new CQAAnswer(answerText, answerDocid, null, isGold) )
          }
        }
      }

      new CQAQuestion(questionText, questionDocid, questionCategory, questionQuality, answerArray.toArray, questionContent)

    }

    // Parse all question / multiple answer pairs
    // For each question
    val pc = (0 until questions.size).par
    nThreads.foreach(k => pc.tasksupport = new ForkJoinTaskSupport(new ForkJoinPool(k)))

    // Parse all question / multiple answer pairs
    // For each question
    val parsedQuestions = for  {
      i <- pc
    } yield parseQuestion(i)

    logger.info (" * CQAQuestionParser: parse: completed... ")

    parsedQuestions.toArray                                                      // Return array of Q/A* pairs

  }

  private def read(r:BufferedReader): Array[String] = {
    val line = r.readLine()
    // println("READ LINE: [" + line + "]")
    if (line.length == 0) return new Array[String](0)
    line.split(SEP)
  }

  def save(questions:Iterable[CQAQuestion], filename:String, annotate:Boolean = true) {
    // Save to XML
    val root = new Element("root")
    val xmlDoc = new nu.xom.Document(root)

    // Step 1: Assemble XML Document
    for (question <- questions) {
      val questionElem = new Element("question")
      val textElem = new Element("text")
      val docidElem = new Element("docid")
      val qualityElem = new Element("quality_score")
      val categoryElem = new Element("category")
      val answersElem = new Element("answers")

      textElem.appendChild(question.text)
      docidElem.appendChild(question.docid)
      qualityElem.appendChild(question.quality)
      categoryElem.appendChild(question.category)

      root.appendChild(questionElem)
      questionElem.appendChild(textElem)
      questionElem.appendChild(docidElem)
      questionElem.appendChild(qualityElem)
      questionElem.appendChild(categoryElem)

      question.content.foreach(
        content => {
          val contentElem = new Element("content")
          contentElem.appendChild(content)
          questionElem.appendChild(contentElem)
        }
      )

      questionElem.appendChild(answersElem)                     // <answers>

      for (answer <- question.answers) {
        val answerElem = new Element("answer")
        val ansTextElem = new Element("text")
        val ansDocidElem = new Element("docid")
        val ansIsGoldElem = new Element("gold")
        val annotationElem = new Element("annotation")

        val os = new StringBuilder

        ansTextElem.appendChild(answer.text)                       // <text>
        ansDocidElem.appendChild(answer.docid)                     // <docid>
        ansIsGoldElem.appendChild(answer.gold.toString)            // <gold>

        if (answer.annotation != null) {                           // <annotation>
          annotationElem.appendChild(documentSerializer.save(answer.annotation))
        } else {
          if (annotate) {
            val annotation = mkPartialAnnotation(answer.text)
            annotationElem.appendChild(documentSerializer.save(annotation))
          } else {
            annotationElem.appendChild("null")
          }
        }


        // Append to <answer> element                           // <answer>
        answerElem.appendChild(ansTextElem)
        answerElem.appendChild(ansDocidElem)
        answerElem.appendChild(ansIsGoldElem)
        answerElem.appendChild(annotationElem)

        // Append <answer> to <answers> element
        answersElem.appendChild(answerElem)
      }
    }

    // Step 2: Write XML File
    val fileXML = new FileOutputStream(filename)
    val serializer = new Serializer(fileXML, "ISO-8859-1")
    serializer.setIndent(4)
//    serializer.setMaxLength(80)
    serializer.write(xmlDoc)
    fileXML.close()

  }


  // Safe wrapper for returning XML fields that may or may not exist
  def safeGetValue(element:nu.xom.Element, defaultValue:String):String =
    optGetValue(element).getOrElse(defaultValue)

  def optGetValue(element:nu.xom.Element): Option[String] = {
    if (element == null) None
    else Some(element.getValue)
  }

  def mkPartialAnnotation(text:String):Document = {
    val doc = queryProcessor.mkDocument(text)
    queryProcessor.tagPartsOfSpeech(doc)
    queryProcessor.lemmatize(doc)
    doc.clear
    doc
  }

  def createHistogram (cqaFilename:String, pw:PrintWriter, bins:Int=50) {
    val questions = load(cqaFilename, false)     // Load CQA questions
    var average:Double = 0.0

    // Initialize histogram
    val hist = new Array[Int](bins)
    for (i <- 0 until hist.size) {
      hist(i) = 0
    }

    // Create histogram
    for (i <- 0 until questions.size) {
      val numAnswers = questions(i).answers.size
      if (numAnswers < bins) {
        hist(numAnswers) += 1
      } else {
        hist(bins-1) += 1       // Last bin is a catch-all
      }
      average += numAnswers
    }
    average /= questions.size

    pw.println ("Histogram:")
    pw.println ("-----------------------")
    for (i <- 0 until hist.size) {
      pw.println ("Bin: " + i + " \t Count: " + hist(i))
    }
    pw.println ("-----------------------")
    pw.println (" Average: " + average.formatted("%3.3f"))

  }


  def createHistogramSentences (cqaFilename:String, pw:PrintWriter, bins:Int=50) {
    val questions = load(cqaFilename, false)     // Load CQA questions
    var average:Double = 0.0
    var numAnswers:Int = 0

    // Initialize histogram
    val hist = new Array[Int](bins)
    for (i <- 0 until hist.size) {
      hist(i) = 0
    }

    // Create histogram
    for (i <- 0 until questions.size) {

      for (answer <- questions(i).answers) {
        val annotation = mkPartialAnnotation( answer.text )
        val numSentences = annotation.sentences.size

        if (numSentences < bins) {
          hist(numSentences) += 1
        } else {
          hist(bins-1) += 1       // Last bin is a catch-all
        }
        average += numSentences
        numAnswers += 1
      }


    }
    average /= numAnswers.toDouble

    pw.println ("Histogram of Number of Sentences per Question:")
    pw.println ("-----------------------")
    for (i <- 0 until hist.size) {
      pw.println ("Bin: " + i + " \t Count: " + hist(i))
    }
    pw.println ("-----------------------")
    pw.println (" Average: " + average.formatted("%3.3f"))
    pw.println (" numAnswers: " + numAnswers)
  }

  def makeFolds(questions:Array[CQAQuestion],
                whichFold:Int,
                numFolds:Int = 5,
                numFoldsTrain:Int = 3,
                numFoldsDev:Int = 1,
                numFoldsTest:Int = 1
               ):(Array[CQAQuestion],Array[CQAQuestion],Array[CQAQuestion]) = {
//    val numFolds = StringUtils.getInt(props, "crossvalidation.folds", 5)
//    val numFoldsTrain = StringUtils.getInt(props, "crossvalidation.foldsTrain", 3)
//    val numFoldsDev = StringUtils.getInt(props, "crossvalidation.foldsDev", 1)
//    val numFoldsTest = StringUtils.getInt(props, "crossvalidation.foldsTest", 1)
//    val debugLimitQuestions = StringUtils.getInt(props, "debug.LimitQuestions", 0)

    val qPerFold:Int = math.floor(questions.size.toDouble / numFolds.toDouble).toInt
    var remainder = questions.size - (qPerFold * numFolds)

    // Error checking
    if ((numFoldsTrain + numFoldsDev + numFoldsTest) != numFolds) {
      throw new RuntimeException("ERROR: MultipleChoiceEntryPoint.makeFolds: 'folds' should sum to 'foldsTrain + foldsDev + foldsTest' in properties file. ")
    }
    if (whichFold >= numFolds) throw new RuntimeException("ERROR: MultipleChoiceEntryPoint.makeFolds: 'whichFold' must not exceed 'numFolds-1' ")

    // Create folds
    val qFolds = new Array[Array[CQAQuestion]](numFolds)
    var start:Int = 0
    for (i <- 0 until numFolds) {
      if (remainder > 0) {
        qFolds(i) = questions.slice(start, start + qPerFold + 1)
        start += qPerFold + 1
        remainder -= 1
      } else {
        qFolds(i) = questions.slice(start, start + qPerFold)
        start += qPerFold
      }
    }

    // Create train/dev/test question files
    var offset = whichFold
    // Train
    val qTrain = new ArrayBuffer[CQAQuestion]()
    for (i <- 0 until numFoldsTrain) {
      qTrain.insertAll(qTrain.size, qFolds((offset)%numFolds))
      //if ((debugLimitQuestions > 0) && (qTrain.size > debugLimitQuestions)) qTrain.remove(debugLimitQuestions, qTrain.size - debugLimitQuestions)
      offset += 1
    }

    // Dev
    val qDev = new ArrayBuffer[CQAQuestion]()
    for (i <- 0 until numFoldsDev) {
      qDev.insertAll(qDev.size, qFolds((offset)%numFolds))
      //if ((debugLimitQuestions > 0) && (qDev.size > debugLimitQuestions)) qDev.remove(debugLimitQuestions, qDev.size - debugLimitQuestions)
      offset += 1
    }

    // Test
    val qTest = new ArrayBuffer[CQAQuestion]()
    for (i <- 0 until numFoldsTest) {
      qTest.insertAll(qTest.size, qFolds((offset)%numFolds))
      //if ((debugLimitQuestions > 0) && (qTest.size > debugLimitQuestions)) qTest.remove(debugLimitQuestions, qTest.size - debugLimitQuestions)
      offset += 1
    }

    // Return train, dev, and test folds
    (qTrain.toArray, qDev.toArray, qTest.toArray)
  }

  def filterMathAndSports(in:Array[CQAQuestion]): Array[CQAQuestion] = {
    val out = new ArrayBuffer[CQAQuestion]
    val filterList = Array(" function ", "divide", "minus", "+", "=", "vs", "multiply", "subtract")
    for (q <- in) {
      var include:Boolean = true
      for (stopWord <- filterList) {
        if (q.questionText.contains(stopWord)) {
          include = false
        }
      }
      if (include) {
        out.append(q)
      } else {
        println ("Filtered out: " + q.questionText)
      }

    }
    out.toArray
  }

}


object CQAQuestionParser {
  val logger = LoggerFactory.getLogger(classOf[CQAQuestionParser])

  def main(args:Array[String]) {
    val props = StringUtils.argsToProperties(args)
    val deepParser = new DeepQuestionParser
    val CQAParser = new CQAQuestionParser

/*
    // Action 1: Take a CQA file in Yahoo! format and convert it to our internal format
    val questions = deepParser.parse("/home/peter/corpora/how_yahoo_answers/xml/deep/deep.xml")
    CQAParser.save(questions, "cqa_questions_yadeep_all.xml", false)
*/

/*
    // Action 2: Take a CQA file in Yahoo! format and extract a subset of questions from it
    val minAnswers = 4
    val files = FreeTextAlignmentUtils.findFiles("/data/nlp/corpora/yahoo_chunked/chunks", "xml")
    //val numQuestions = 1000000
    //val questionsAll = deepParser.parse("/home/peter/corpora/how_yahoo_answers/xml/deep/deep.xml")
    for (fileID <- files.indices) {
      val filename = files(fileID).getAbsolutePath
      val questionsAll = deepParser.parse(filename)
      val questionsFiltered = deepParser.getWithMinimumAnswers(questionsAll, minAnswers)

      // Causal:
      val causalQuestions = deepParser.getCausal(questionsFiltered)
      for (question <- causalQuestions) {
        println (question.toQuestion().toString())
      }

      CQAParser.save(causalQuestions, s"cqa_questions_yadeep_min4_causal$fileID.giza.xml", false)


//      var rand = new ArrayBuffer[CQAQuestion]()
//      rand = rand ++ questionsFiltered
//      extractionUtils.Random.setSeed(10)     // Set to static seed
//      val questionsRandomized = extractionUtils.Random.shuffle(rand).toArray
//
//      //val questions = questionsRandomized.slice(0, numQuestions)
//      val questions = questionsRandomized.slice(10001, questionsRandomized.size)
//      //val questions = questionsRandomized
//
//      CQAParser.save(questions, "cqa_questions_yadeep_min4_all.giza.xml", false)

    }

*/

/*
    //val questions = CQAParser.load("cqa_questions_yadeep_min4_all.giza.xml", true)
    val qFiles = FreeTextAlignmentUtils.findFiles("/data/nlp/corpora/cqa_yahoo_causal_may2/extractedQuestions", "xml")
    val questions = for {
      file <- qFiles
      filePath = file.getAbsolutePath

    } yield CQAParser.load(filePath, annotate = true)
    //CQAParser.load("/data/nlp/corpora/yahoo/cqa_questions_yadeep_min4_causal.giza.xml", true)
    val xmlsave = new AutoExtraction("")      // TODO: Refactor to place XML save function in a better spot
    //xmlsave.writeGoldQuestionsToXML(CQAParser.CQAtoQuestions(questions).toList, "cqa_questions_yadeep_min4_all.q.giza.xml", 1)
    xmlsave.writeGoldQuestionsToXML(CQAParser.CQAtoQuestions(questions.flatten).toList, "/data/nlp/corpora/cqa_yahoo_causal_may2/cqa_questions_yadeep_min4_causal.q.giza.xml", 1)

*/
/*
    // Action 3: Test -- Load a previously saved CQA file
    val questions = CQAParser.load("cqa_questions_yadeep_min4_10k.xml", true)
    //val questions = CQAParser.load("cqa_questions_yadeep_all.xml", true)
    for (i <- 0 until 10) {
      println ("Question[" + i + "]: " + questions(i).text)
      println ("  DOCID[" + i + "]: " + questions(i).docid)
      println ("  Category[" + i + "]: " + questions(i).category)
      for (j <- 0 until questions(i).answers.size) {
        println ("  Answer[" + j + "]: " + questions(i).answers(j).text)
        println ("    DOCID[" + j + "]: " + questions(i).answers(j).docid)
        println ("    Gold[" + j + "]: " + questions(i).answers(j).gold)
        val tags = questions(i).answers(j).annotation.sentences(0).tags.get
        println ("    Annotation: " + tags(0))
      }
    }
*/

/*
    // Action 4: Create train/dev/test folds
    val filenamePrefix = "/data/nlp/corpora/cqa_yahoo_causal_may2/cqa_questions_yadeep_min4_causal"
    val qFiles = FreeTextAlignmentUtils.findFiles("/data/nlp/corpora/cqa_yahoo_causal_may2/extractedQuestions", "xml")
    val questions1 = for {
      file <- qFiles
      filePath = file.getAbsolutePath
    } yield CQAParser.load(filePath, annotate = true)

    //val filenamePrefix = "cqa_questions_yadeep_min4_500"
    //val questions = CQAParser.load(filenamePrefix + ".xml", true)
    val questions2 = questions1.flatten
    // Filter out math and sports questions
    val questions3 = CQAParser.filterMathAndSports(questions2)

    val rand = new Random(seed = 426)
    val questions = rand.shuffle(questions3.toTraversable).toArray
    CQAParser.save(questions, filenamePrefix + ".cqa.all.xml", false)

    val size = questions.size
    val train = questions.slice(0, size/2)
    val dev = questions.slice(size/2, (size/2) + (size/4))
    val test = questions.slice((size/2) + (size/4), size)
    println (s"NumQuestions -- train: ${train.length}\tdev: ${dev.length}\ttest: ${test.length}")
    // Save CQAQuestion folds
    CQAParser.save(train, filenamePrefix + ".filtered.cqa.train.xml", false)
    CQAParser.save(dev, filenamePrefix + ".filtered.cqa.dev.xml", false)
    CQAParser.save(test, filenamePrefix + ".filtered.cqa.test.xml", false)

    // Save regular Question file folds
    val xmlsave = new AutoExtraction("")      // TODO: Refactor to place XML save function in a better spot
    xmlsave.writeGoldQuestionsToXML(CQAParser.CQAtoQuestions(train).toList, filenamePrefix + ".filtered.q.train.xml", 1)
    xmlsave.writeGoldQuestionsToXML(CQAParser.CQAtoQuestions(dev).toList, filenamePrefix + ".filtered.q.dev.xml", 1)
    xmlsave.writeGoldQuestionsToXML(CQAParser.CQAtoQuestions(test).toList, filenamePrefix + ".filtered.q.test.xml", 1)
*/
//    /*
    // Action 4b (alternative): Create crossvalidation folds
    val filenamePrefix = "/data/nlp/corpora/cqa_yahoo_causal_may2/cqa_questions_yadeep_min4_causal"
    val qFiles = FreeTextAlignmentUtils.findFiles("/data/nlp/corpora/cqa_yahoo_causal_may2/extractedQuestions", "xml")
    val questions1 = for {
      file <- qFiles
      filePath = file.getAbsolutePath
    } yield CQAParser.load(filePath, annotate = true)

    //val filenamePrefix = "cqa_questions_yadeep_min4_500"
    //val questions = CQAParser.load(filenamePrefix + ".xml", true)
    val questions2 = questions1.flatten

    // Filter questions with Math/sports
    val questions3 = CQAParser.filterMathAndSports(questions2)

    val numFolds = 5
    for (i <- 0 until numFolds) {
      val rand = new Random(seed = i)
      val questions = rand.shuffle(questions3.toTraversable).toArray
      //CQAParser.save(questions, filenamePrefix + ".cqa.all.xml", false)
      //val numFolds: Int = 2
      val numTrain: Int = 3
      val numDev: Int = 1
      val numTest:Int = 1
      val (train, dev, test) = CQAParser.makeFolds(questions, i, numFolds, numTrain, numDev, numTest)

//      val train = questions.slice(0, size/2)
//      val dev = questions.slice(size/2, (size/2) + (size/4))
//      val test = questions.slice((size/2) + (size/4), size)
      println (s"NumQuestions -- train: ${train.length}\tdev: ${dev.length}\ttest: ${test.length}")
      // Save CQAQuestion folds
      CQAParser.save(train, filenamePrefix + s".filtered.CV.${numFolds}Folds.$i" + ".cqa.train.xml", false)
      CQAParser.save(dev, filenamePrefix + s".filtered.CV.${numFolds}Folds.$i" + ".cqa.dev.xml", false)
      CQAParser.save(test, filenamePrefix + s".filtered.CV.${numFolds}Folds.$i" + ".cqa.test.xml", false)

      // Save regular Question file folds
      val xmlsave = new AutoExtraction("")      // TODO: Refactor to place XML save function in a better spot
      xmlsave.writeGoldQuestionsToXML(CQAParser.CQAtoQuestions(train).toList, filenamePrefix + s".filtered.CV.${numFolds}Folds.$i" + ".q.train.xml", 1)
      xmlsave.writeGoldQuestionsToXML(CQAParser.CQAtoQuestions(dev).toList, filenamePrefix + s".filtered.CV.${numFolds}Folds.$i" + ".q.dev.xml", 1)
      xmlsave.writeGoldQuestionsToXML(CQAParser.CQAtoQuestions(test).toList, filenamePrefix + s".filtered.CV.${numFolds}Folds.$i" + ".q.test.xml", 1)
    }
//*/

/*
    // Action 5: Create faux Lucene index (for IR term frequency weighting) from CQA file
    //val filenamePrefix = "cqa_questions_yadeep_min4_10k"
    //val index_out = "/home/peter/corpora/cqa_index_test10k"
    val filenamePrefix = "/data/nlp/corpora/cqa_yahoo_causal_may2/cqa_questions_yadeep_min4_causal.cqa.all"
    val index_out = "/data/nlp/corpora/cqa_yahoo_causal_may2/index"
    val dryrun:Boolean = false
    val partialAnnotation:Boolean = true                              // "true" just does tokenizing, POS tagging, and lemmatization -- much faster

    val questions = CQAParser.load(filenamePrefix + ".xml", true)     // Load CQA questions
    val qaPairs = CQAParser.CQAtoQuestions(questions, false)          // Turn into Questions array, but containing all answers (not just gold)
    println (" Adding to index... ( " + index_out + " )")
    val indexer = new Indexer("", index_out)          // Collection dir is blank as we're creating an index directly from the GoldQA array
    indexer.indexFromGoldAnswers(qaPairs, dryrun, partialAnnotation)

    val pw = new PrintWriter("YA_Causal_numAnsCand_histogram.txt")
    val pw2 = new PrintWriter("YA_Causal_numAnsSents_histogram.txt")
    CQAParser.createHistogram(filenamePrefix + ".xml", pw, 50)
    CQAParser.createHistogramSentences(filenamePrefix + ".xml", pw2, 50)
    pw.close()
    pw2.close()
*/
/*
    // Action 6: Create histogram of number of answers per question in CQA file
    val filenamePrefix = "cqa_questions_yadeep_min4_10k"
    val pw = new PrintWriter(System.out, true)
    CQAParser.createHistogram(filenamePrefix + ".xml", pw, 50)
*/

/*
    // Action 7: Count occurrances of discourse markers in answers in CQA file
    val filenamePrefix = "cqa_questions_yadeep_min4_10k"
    val pw = new PrintWriter(System.out, true)
    val questions = CQAParser.load(filenamePrefix + ".xml", true)     // Load CQA questions
    val qaPairs = CQAParser.CQAtoQuestions(questions, false)          // Turn into Questions array, but containing all answers (not just gold)
    val discourseFinder = new FinderDiscourse("FULL")
    var numMarkers:Int = 0
    var numAnswers:Int = 0

    for (qaPair <- qaPairs) {
      for (ga <- qaPair.goldAnswers) {
        val annotation = CQAParser.mkPartialAnnotation(ga.text)
        val markers = discourseFinder.find(annotation, 0, annotation.sentences.size)
        numMarkers += markers.size
        numAnswers += 1
      }
    }

    println ("Number of Makers: " + numMarkers)
    println ("Number of Answers: " + numAnswers)
    println ("Average: " + (numMarkers.toDouble / numAnswers.toDouble))
*/


/*
    // Action 8: Create histogram of number of answers per question in CQA file
    val filenamePrefix = "/data1/nlp/corpora/qa/yahoo_cqa10k/train/cqa_questions_yadeep_min4_10k.cqa.train"
    val pw = new PrintWriter(System.out, true)
    CQAParser.createHistogramSentences(filenamePrefix + ".xml", pw, 50)
*/


    //    println ("Number of questions:" + questions.size)
  }
}
