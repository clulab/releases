package edu.arizona.sista.qa.ranking

import java.util.Properties
import edu.arizona.sista.learning._
import java.io.PrintWriter
import edu.arizona.sista.utils.{VersionUtils, StringUtils}
import collection.mutable.{ListBuffer, ArrayBuffer}
import edu.arizona.sista.qa.scorer.{Question, Histogram, Scorer, Scores}
//import edu.arizona.sista.qa.discourse._
import edu.arizona.sista.qa.QA
import edu.arizona.sista.utils.MathUtils.softmax
import scala.{Array, Tuple2}
import edu.arizona.sista.processors.{Processor, Document}
import edu.arizona.sista.qa.retrieval._
import edu.arizona.sista.qa.translation.{TranslationMultisetModel, TranslationRelationModel}
import org.slf4j.LoggerFactory
import edu.arizona.sista.qa.preprocessing.yahoo.deep.{CQAQuestion, CQAQuestionParser}
import util.control.Breaks._
import scala.Tuple2
import edu.arizona.sista.qa.index.TermFilter
import edu.arizona.sista.qa.word2vec.{Word2VecMultisetModel, Word2VecRelationModel, CNNRelationModel}
import edu.arizona.sista.qa.baselines.RelationLookupModel


/**
 * Modified version of RankerVoting that will simply combine features of submodels using a MultiModel
 * User: dfried, peter
 * Date: 8/27/13
 */
class RankerCombined(props:Properties) extends RankerVoting(props:Properties) {
  /*
   * Main entry point -- orchestrates a complete training, tuning, and test run of a model
   */
  override def doRun(workunitID:String) {
    val classifierProperties = new Properties()
    logger.info("Worker: Workunit (WID: " + workunitID + " ) started...")

    // Step 1: Load properties used internally
    val trainMethod = props.getProperty("ranker.train_method", "incremental").toLowerCase
    val answersFromIR = StringUtils.getInt(props, "ranker.answersfromIR", 20)

    // Step 1A: Append thread IDs to any filenames used internally by the RankingClassifier to make them thread safe
    val svm_c = props.getProperty("ranker.svm_c", "0.01")     // extract C parameter from properties file
    classifierProperties.put("c", svm_c)
    classifierProperties.put("debugFile", "./debug-features-w" + workunitID )
    classifierProperties.setProperty("modelFile", "model" + workunitID + ".dat")
    classifierProperties.setProperty("trainFile", "train" + workunitID + ".dat")
    val classifierClass = props.getProperty("classifierClass", "")
    if (classifierClass != "") classifierProperties.setProperty("classifierClass", classifierClass)

    // Step 2: Perform model selection
    val modelSelection = props.getProperty("ranker.model", "cusp").split(",")

    var subModels = new ArrayBuffer[RankingModel]

    if (modelSelection.contains("word2vecmultiset")) {
      subModels.append( new Word2VecMultisetModel(props))
    }
    if (modelSelection.contains("word2vecrelation")) {
      subModels.append( new Word2VecRelationModel(props))
    }
    if (modelSelection.contains("relationlookup")) {
      subModels.append( new RelationLookupModel(props))
    }
    if (modelSelection.contains("translationrelation")) {
      subModels.append( new TranslationRelationModel(props))
    }
    if (modelSelection.contains("cnnrelation")) {
      subModels.append( new CNNRelationModel(props))
    }

    models.append(new MultiModel(props, subModels))

    if (models.size == 0) throw new RuntimeException("ERROR: RankerCombined: No recognized models were specified for ranker.model in the properties file. ")


    // Step 3: Load Training, Development, and Test questions
    logger.info ("Reading questions...  ")
    // Training questions
    val questionsTrainFilename = props.getProperty("gold")
    var questionsTrainSize = questionParser.numQuestions(questionsTrainFilename)
    // If enabled, Limit training set size for debug purposes
    val questionTrainLimit = StringUtils.getInt(props, "ranker.limit_train_questions", 0)
    if ((questionTrainLimit != 0) && (questionTrainLimit < (questionsTrainSize - 1))) questionsTrainSize = questionTrainLimit

    // Development questions
    var questionsDevFilename = props.getProperty("dev", "")
    var questionsDevSize = 0
    if (questionsDevFilename != "") {
      questionsDevSize = questionParser.numQuestions(questionsDevFilename)
      // If enabled, Limit training set size for debug purposes
      val questionDevLimit = StringUtils.getInt(props, "ranker.limit_dev_questions", 0)
      if ((questionDevLimit != 0) && (questionDevLimit < (questionsDevSize - 1))) questionsDevSize = questionDevLimit
    }

    // Test questions
    var questionsTestFilename = props.getProperty("test")
    var questionsTestSize = questionParser.numQuestions(questionsTestFilename)
    // If enabled, Limit training set size for debug purposes
    val questionTestLimit = StringUtils.getInt(props, "ranker.limit_test_questions", 0)
    if ((questionTestLimit != 0) && (questionTestLimit < (questionsTestSize - 1))) questionsTestSize = questionTestLimit

    logger.info ("Reading questions complete.  ")

    // Step 4: Setup a verbosely named log file with the parameters for the experiment in the filename.
    var incrementalTrainingFile: String = generateVerboseFilename(props, questionsTrainSize.toString, questionsTestSize.toString, svm_c)
    props.setProperty ("output_report_filename", incrementalTrainingFile)


    logger.info ("Opening report file... (filename = " + incrementalTrainingFile + " )")
    val pw = new PrintWriter(incrementalTrainingFile)
    // print out the git revision to make it easier to replicate results
    VersionUtils.gitRevision.foreach(r => {
      pw.println(s"git revision $r")
    })
    displayProperties(props, pw)
    pw.println ("------")
    pw.flush()

    pw.println ("Pre-training")
    displayMemoryUsage(pw)

    // Step 5: Run Experiment

    val loadClassifiers = props.getProperty("ranker.load_classifiers", "")
    if (loadClassifiers != "") {
      // Load classifiers
      val filenamesClassifiers = loadClassifiers.split(",")
      logger.info("Loading Classifiers")
      for (filename <- filenamesClassifiers) {
        classifiers.append(SVMRankingClassifier.loadFrom(filename))
      }
    } else {
      // Train new classifiers
      logger.info("Running training... ")
      trainV(questionsTrainFilename, answersFromIR, classifierProperties, false, pw)

      // Save classifiers using default names
      logger.info("Saving classifiers...")
      for (classifierIdx <- 0 until classifiers.size) {
        val prefix = incrementalTrainingFile.substring(0, incrementalTrainingFile.size - 4)
        val filename = prefix + ".classifier" + classifierIdx + ".svm"
        classifiers(classifierIdx).saveTo(filename)
      }
    }

    pw.println ("Post-training, Pre-tuning")
    System.gc()                   // Manually perform garbage collection
    displayMemoryUsage(pw)

    // Step 6: Run tuning procedure
    if (questionsDevFilename != "") {
      doTuningProcedureV(answersFromIR, questionsDevFilename, true, pw)      // tuning
      pw.flush()
    } else {
      logger.info ("No development question set specified -- skipping tuning... ")
      pw.println (" NOTE: No development question set specified -- skipping over tuning procedure")
    }

    pw.println ("Post-tuning, Pre-test")
    System.gc()                   // Manually perform garbage collection
    displayMemoryUsage(pw)

    // Step 7: Run test procedure
    logger.info ("Running test procedure... ")
    pw.println (" ======================================================================================================================= ")
    pw.println ("           Testing performance on test set... ")
    pw.println (" ======================================================================================================================= ")


    doTestProcedureV(answersFromIR, questionsTestFilename, true, pw)      // test
    pw.flush()

    // Step 9: Cleanup
    System.gc()                   // Manually perform garbage collection
    displayMemoryUsage(pw)
    pw.close()
    println ("Workunit complete... ")
    logger.info("Worker: Workunit (WID: " + workunitID + " ) completed...")

  }

}


object RankerCombined {
  val logger = LoggerFactory.getLogger(classOf[RankerCombined])   // Correct?

  def loadFrom(filenamesClassifiers:Array[String], props:Properties):RankerCombined = {
    // Create new RankingVoter
    val ranker = new RankerCombined(props)

    // Step 2: Load models
    logger.info ("Loading Models")
    val modelSelection = props.getProperty("ranker.model", "cusp").split(",")
    if (ranker.models.size == 0) throw new RuntimeException("ERROR: RankerCombined: No recognized models were specified for ranker.model in the properties file. ")

    // Step 3: Load classifiers
    logger.info ("Loading Classifiers")
    for (filename <- filenamesClassifiers) {
      ranker.classifiers.append( SVMRankingClassifier.loadFrom(filename) )
    }
    if (ranker.classifiers.size == 0) {
      throw new RuntimeException("ERROR: RankerCombined: Number of clasifiers to load must be non-zero")
    }

    // Step 4: Misc error checking
    if (ranker.models.size != ranker.classifiers.size) {
      throw new RuntimeException("ERROR: RankerCombined: Number of models to load must be equal to the number of classifiers (models=" + ranker.models.size + " rankers=" + ranker.classifiers.size + ")")
    }

    logger.debug ("loadFrom Complete -- returning ranker")
    ranker
  }

}


