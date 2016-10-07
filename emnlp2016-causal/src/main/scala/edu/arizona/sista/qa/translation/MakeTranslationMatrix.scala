package edu.arizona.sista.qa.translation

import org.slf4j.LoggerFactory
import scala.sys.process._
import edu.arizona.sista.utils.StringUtils
import edu.arizona.sista.qa.scorer.{Question, QuestionParser}
import java.io._
import edu.arizona.sista.processors.{Sentence, DocumentSerializer, Document, Processor}
import MakeTranslationMatrix._
import edu.arizona.sista.struct.Lexicon
import edu.arizona.sista.struct.Counter
import scala.collection.mutable
import edu.arizona.sista.qa.AlignedText
import edu.arizona.sista.qa.preprocessing.yahoo.deep.{CQAQuestion, CQAQuestionParser}
import edu.arizona.sista.processors.fastnlp.FastNLPProcessor
import scala.collection.mutable.ArrayBuffer
//import edu.arizona.sista.qa.discourse.{DiscourseExplorer, DiscourseUnit}
import scala.io.Source
//import edu.arizona.sista.qa.preprocessing.agiga.{AgigaSerializedDiscourseReader}
//import edu.arizona.sista.qa.preprocessing.AgigaFileSlice
//import edu.arizona.sista.mc.preprocessing.tessellatedgraph.GigawordDocFreq
import scala.util.control.NonFatal
import scala.util.Random
import edu.arizona.sista.qa.index.TermFilter
import java.util.Properties
//import edu.arizona.sista.qa.translation.FreeTextPreprocessing.{PreprocessAgiga, PreprocessTextbook}

/**
 * Use gold file to generate translation table using GIZA++, then convert table into a single-file that's easily loaded by the translation package
 * User: peter, mihai
 * Date: 8/2/13
 */

class MakeTranslationMatrix() {}

object MakeTranslationMatrix {
  lazy val termFilter = new TermFilter

  val logger = LoggerFactory.getLogger(classOf[MakeTranslationMatrix])
  lazy val questionParser = new QuestionParser

  val QTEMP = "questions_temp"
  val ATEMP = "answers_temp"

  // Make the temp files which are input to Giza
  def makeGizaQuestionsFile(
                             questions: Array[_ <: AlignedText],
                             filenameQuestions: String,
                             filenameAnswers:String,
                             viewType:String,
                             sanitizeForWord2Vec:Boolean) {
    val pwQuestions = new PrintWriter(filenameQuestions)
    val pwAnswers = new PrintWriter(filenameAnswers)
    logger.debug ("* makeGizaQuestionsFile: Started...")
    logger.debug (" Questions to process: " + questions.size)

    var qNum:Int = 0
    var aNum:Int = 0
    for (question <- questions) {
      val viewQ = FreeTextAlignmentUtils.mkView(question.questionText, viewType, sanitizeForWord2Vec)

      if(! viewQ.empty) {
        val featureStringQ:String = viewQ.toString
        logger.debug(" Question [" + qNum + "]: " + featureStringQ)

        for (text <- question.answersTexts) {
          val viewA = FreeTextAlignmentUtils.mkView(text, viewType, sanitizeForWord2Vec)

          if(! viewA.empty) {
            val featureStringA = viewA.toString
            pwQuestions.println (featureStringQ)    // Question lemmas, printed once for each gold answer!
            pwAnswers.println (featureStringA)      // Answer lemmas
            aNum += 1
          }
        }
        qNum += 1
      }
    }

    pwQuestions.close()
    pwAnswers.close()

    logger.debug (s"* makeGizaQuestionsFile: Completed, for $qNum questions and $aNum Q/A pairs.")

  }

  // Make the Giza Config file
  def mkGizaConfig(wdir:String,
                   config:String,
                   src:String,
                   dst:String,
                   sent:String,
                   out:String,
                   ml:Int,
                   maxFertility:Int = 10) {
    val fn = wdir + File.separator + config
    val os = new PrintWriter(new FileWriter(fn))
    os.println(s"S: $wdir/$src.vcb")
    os.println(s"T: $wdir/$dst.vcb")
    os.println(s"C: $wdir/$sent")
    os.println("model1iterations: 5")
    os.println("model2iterations: 0")
    os.println("model3iterations: 0")
    os.println("model4iterations: 0")
    os.println("hmmiterations: 0")
    os.println("model1dumpfrequency: 1")
    os.println("model2dumpfrequency: 1")
    os.println("model345dumpfrequency: 1")
    os.println(s"ml: $ml")                  //Soft max - can make smaller here, but can't make bigger w/o changing the C code
    os.println(s"o: $wdir/$out")
    //os.println(s"maxfertility $maxFertility")
    os.close()
  }

  // Convert the temp files made into a format which Giza takes in
  def mkGizaInput(wdir:String, srcName:String, dstName:String) {
    println (srcName + "___" + dstName)
    val srcLexicon = new Lexicon[String]
    val srcCounts = new Counter[String]
    val dstLexicon = new Lexicon[String]
    val dstCounts = new Counter[String]

    // keeps track of how many times we've seen target (i.e., question by default) words
    //   in sentences that do not contain it
    val nonSelfSents = new Counter[String]()

    val src = new BufferedReader(new FileReader(wdir + File.separator + srcName))
    val dst = new BufferedReader(new FileReader(wdir + File.separator + dstName))
    val out = new PrintWriter(new FileWriter(wdir + File.separator + srcName + "_" + dstName + ".snt"))
    var done = false
    while(! done) {
      val srcLine = src.readLine()
      val dstLine = dst.readLine()
      if(srcLine == null || dstLine == null) {
        done = true
      } else {
        out.println(1)
        saveLine(out, srcLine, srcLexicon, srcCounts)
        saveLine(out, dstLine, dstLexicon, dstCounts)

        countTrainingSents(dstLine, srcLine, nonSelfSents)
      }
    }
    src.close()
    dst.close()

    out.close()

    saveVocab(srcLexicon, srcCounts, wdir + File.separator + srcName + ".vcb")
    saveVocab(dstLexicon, dstCounts, wdir + File.separator + dstName + ".vcb")
  }

  private def countTrainingSents(dstLine:String, srcLine:String, nonSelfSents:Counter[String]) {
    val dstTokens = dstLine.split("\\s+").toSet
    //val srcTokens = srcLine.split("\\s+").toSet

    for(dstToken <- dstTokens) {
      //if(! srcTokens.contains(dstToken)) nonSelfSents.incrementCount(dstToken)
      nonSelfSents.incrementCount(dstToken)
    }
  }

  def saveVocab(lexicon:Lexicon[String], counts:Counter[String], fn:String) {
    val os = new PrintWriter(new FileWriter(fn))
    for(i <- 0 until lexicon.size) {
      val w = lexicon.get(i)
      os.println(s"${i + 1} $w ${counts.getCount(w).toInt}")
    }
    os.close()
  }

  def saveLine(out:PrintWriter, line:String, lexicon:Lexicon[String], counts:Counter[String]) {
    val tokens = line.split("\\s+")
    var first = true
    for(token <- tokens) {
      if(! first) out.print(" ")
      out.print(lexicon.add(token) + 1)
      counts.incrementCount(token)
      first = false
    }
    out.println()
  }

  /*
    * Methods for Making/Saving Priors
   */

  /**
   * Creates prior probabilities from the original questions and answers
   * This is safest way to compute priors. Using giza vocabularies is not accurate,
   *   especially for qtoa, because the questions that have multiple gold answers
   *   are repeated in the Giza dataset.
   * @param questions Questions with gold answers
   * @param wdir Working dir for Giza
   * @param prefix Prefix to append to all files generated
   * @param viewType Representation of text, e.g., "lemmas"
   */
  def mkPriors(
                questions:Iterable[_ <: AlignedText],
                wdir:String,
                prefix:String,
                viewType:String,
                sanitizeForWord2Vec:Boolean,
                gizaLengthLimit:Int = -1) {
    logger.debug("Computing priors...")
    var total = 0
    val counts = new Counter[String]
    for (question <- questions) {
      val viewQ = FreeTextAlignmentUtils.mkView(question.questionText, viewType, sanitizeForWord2Vec)
      for (answerText <- question.answersTexts) {
        val viewA = FreeTextAlignmentUtils.mkView(answerText, viewType, sanitizeForWord2Vec)
        for (f <- viewA.features) {
          total += 1
          counts.incrementCount(f)
        }
      }

    }

    var infix:String = "_atoq"
    if (gizaLengthLimit != -1) infix += "_ml" + gizaLengthLimit

    logger.debug("Priors computed for a collection of size " + total)
    savePriors(counts, total, wdir + File.separator + prefix + infix + ".priors")
  }


  def savePriors(counts:Counter[String], total:Int, fn:String) {
    val os = new PrintWriter(new FileWriter(fn))
    val keys = counts.keySet.toList.sorted
    for(key <- keys) {
      val prior = counts.getCount(key) / total.toDouble
      os.println(key + " " + prior)
    }
    os.close()
    logger.debug("Priors saved in file: " + fn)
  }



  /*
    * Methods for Making Matrices
   */

  /**
   * Main entry point
   * @param questions A list of gold QA pairs
   * @param filePrefix the filename for the GIZA++ translation matrix output
   * @param viewType How to represent the text in questions and answers
   */
  def makeTranslationMatrix(
                             questions:Array[_ <: AlignedText],
                             filePrefix:String,
                             mode:String = "atoq",
                             viewType:String = "lemmas",
                             gizaLengthLimit:Int = 1001,
                             gizaPath:String = "GIZA++",
                             sanitizeForWord2Vec:Boolean) {

    val (wdir, transPrefix) = FreeTextAlignmentUtils.extractDir(filePrefix)
    logger.debug(s"Using working dir $wdir and prefix $transPrefix")

    makeGizaQuestionsFile(questions,
      wdir + File.separator + MakeTranslationMatrix.QTEMP,
      wdir + File.separator + MakeTranslationMatrix.ATEMP,
      viewType,
      sanitizeForWord2Vec)

    // Create translation matrix
    makeGizaTranslationFile(wdir, transPrefix, mode = mode, gizaPath = gizaPath, gizaLengthLimit = gizaLengthLimit, buildFilenames = true)

    // Create priors
    mkPriors(questions, wdir, transPrefix, viewType, sanitizeForWord2Vec, gizaLengthLimit = gizaLengthLimit)
  }


  //### Peter: Temporary, takes Q/A filenames as parameters
  def makeTranslationMatrixManual( filenameQ:String,
                             filenameA:String,
                             questions:Array[AlignedText],
                             filePrefix:String,
                             viewType:String = "lemmas",
                             mode:String = "atoq",
                             gizaPath:String = "GIZA++",
                             sanitizeForWord2Vec:Boolean) {

    val (wdir, transPrefix) = FreeTextAlignmentUtils.extractDir(filePrefix)
    logger.debug(s"Using working dir $wdir and prefix $transPrefix")

    // Copy Q/A filenames to temporary filenames used by the rest of the class
    FreeTextAlignmentUtils.exe ("cp " + wdir + File.separator + filenameQ + " " + wdir + File.separator + MakeTranslationMatrix.QTEMP)
    FreeTextAlignmentUtils.exe ("cp " + wdir + File.separator + filenameA + " " + wdir + File.separator + MakeTranslationMatrix.ATEMP)

    // Create translation matrix
    makeGizaTranslationFile(wdir, transPrefix, mode = mode, gizaPath = gizaPath)

    // Create priors
    mkPriors(questions, wdir, transPrefix, viewType, sanitizeForWord2Vec)
  }



  // Loads up the temp src/dst files, converts to the required format for Giza, makes the config file, and
  def makeGizaTranslationFile(
                               wdir:String,
                               transPrefix:String,
                               mode:String,
                               gizaLengthLimit:Int = 1001,
                               buildFilenames:Boolean = false,
                               providedSrcFilename:String = "",
                               providedDstFilename:String = "",
                               gizaPath:String = "GIZA++",
                               keepGizaFiles:Boolean = true,
                               maxFertility:Int = 10) {

    var filenameSrc:String = ""
    var filenameDst:String = ""

    // If set to build the filenames, build them based on designated parameters
    if (buildFilenames){
      if (mode.toLowerCase == "qtoa") {
        filenameSrc = QTEMP
        filenameDst = ATEMP
      } else if (mode.toLowerCase == "atoq") {
        filenameSrc = ATEMP
        filenameDst = QTEMP
      } else {
        throw new RuntimeException("ERROR: unknown mode " + mode)
      }

    } else {
      // If not building filenames (i.e. they're provided)
      filenameSrc = providedSrcFilename
      filenameDst = providedDstFilename
    }

    //make informative names for matrix storage
    val filenamePrefix = transPrefix + "_" + mode + "_ml" + gizaLengthLimit
    val outFile = filenamePrefix + "_gizaout"

    // Step 1: create vocabularies and file with aligned sentences
    mkGizaInput(wdir, filenameSrc, filenameDst)

    // Step 2: Make Giza config file
    val filenameSentences = filenameSrc + "_" + filenameDst + ".snt"
    val configFile = filenamePrefix + ".config"
    mkGizaConfig(wdir, configFile, filenameSrc, filenameDst, filenameSentences, outFile, gizaLengthLimit, maxFertility)

    // Step 3: Call Giza++ to generate the translation matrix
    FreeTextAlignmentUtils.exe(s"$gizaPath $wdir/$configFile")

    // Step 4: Generate the actual translation matrix (in our format) from Giza's output
    // We save only the string-based matrix here, ignoring giza's vocabularies (we create our own later)
    makeStringBasedTable(
      s"$wdir/$outFile.t1.5",
      s"$wdir/$outFile.trn.src.vcb",
      s"$wdir/$outFile.trn.trg.vcb",
      s"$wdir/$filenamePrefix.matrix")

    // Step 5 (optional): delete all giza output files
    if(! keepGizaFiles) {
      val files = new File(wdir).listFiles()
      files.foreach(f => if(f.getName.contains("_gizaout")) f.delete())
    }
  }

  /**
   * Creates the translation table that we use in the rest of our code, from GIZA's output
   * The format of each line in the generated file is: dstWord srcWord p(dstWord|srcWord)
   * @param gizaOutputFile
   * @param srcVocabFile
   * @param dstVocabFile
   * @param outputFile
   */
  def makeStringBasedTable(gizaOutputFile:String,
                           srcVocabFile:String,
                           dstVocabFile:String,
                           outputFile:String) {
    val srcVocab = loadGizaVocab(srcVocabFile)
    val dstVocab = loadGizaVocab(dstVocabFile)
    val os = new PrintStream(new FileOutputStream(outputFile))
    for(line <- io.Source.fromFile(gizaOutputFile).getLines()) {
      val bits = line.trim.split("\\s+")
      assert(bits.length == 3)
      val sid = bits(0).toInt // source token on the first position!
      val did = bits(1).toInt // target token on the second position!
      val probDgivenS = bits(2).toDouble // // p(target|source)

      // skip NULL words (id == 0). not actually used in the model
      if(sid != 0 && did != 0) {
        val sw = srcVocab.get(sid)
        assert(sw.isDefined)
        val dw = dstVocab.get(did)
        assert(dw.isDefined)

        os.println(s"${sw.get} ${dw.get} $probDgivenS")
      }
    }
    os.close()
  }


  /*
    * Methods for indexing, loading, and saving files
   */

  def loadGizaVocab(vocabFile:String):Map[Int, String] = {
    val v = new mutable.HashMap[Int, String]()
    for(line <- io.Source.fromFile(vocabFile).getLines()) {
      val bits = line.trim.split("\\s+")
      assert(bits.length == 3)
      v.put(bits(0).toInt, bits(1))
    }
    v.toMap
  }



   def main(args:Array[String]) {

     val props = StringUtils.argsToProperties(args)

     val sanitizeForW2V = StringUtils.getBool(props, "translation.w2v_sanitization", true)
     val mode = props.getProperty("translation.mode", "atoq")                                 // supported are atoq, qtoa, ston, and ntos
     val filenameMatrix = props.getProperty("translation.matrix_output_filename", "translation_matrix.txt")
     val gizaPath = props.getProperty("translation.gizapath", "GIZA++")
     val desiredView = props.getProperty("view.view", "lemmas")
     val questionsFilename = props.getProperty("translation.train_questions") // for Making a matrix from YA questions & answers
     val questionLimit = StringUtils.getInt(props, "translation.limit_train_questions", 0)

     logger.info(s"View: $desiredView")

     // Step 1: Load training questions
     logger.info("Reading questions...  ")
     var questions = questionParser.parse(questionsFilename).toArray

     // Step 2: If enabled, Limit training set size for debug purposes
     if ((questionLimit != 0) && (questionLimit < (questions.size - 1))) questions = questions.slice(0, questionLimit)

     // Step 3: Create a translation matrix from the list of questions
     makeTranslationMatrix(
       questions = questions,
       filePrefix = filenameMatrix,
       mode = mode,
       viewType = desiredView,
       gizaPath = gizaPath,
       sanitizeForWord2Vec = sanitizeForW2V)

     logger.info("Matrix generation complete.")

   }

}
