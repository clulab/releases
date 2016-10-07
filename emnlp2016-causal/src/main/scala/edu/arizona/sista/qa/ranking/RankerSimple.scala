package edu.arizona.sista.qa.ranking

import java.util.Properties
import java.io.PrintWriter
import edu.arizona.sista.learning.{SVMRankingClassifier, RankingClassifier}
import edu.arizona.sista.utils.StringUtils
import collection.mutable.ArrayBuffer
import edu.arizona.sista.qa.scorer.{Histogram, Scorer, Scores}

/**
 * Created with IntelliJ IDEA.
 * User: peter
 * Date: 7/31/13
 */

class RankerSimple(props:Properties) extends Ranker(props:Properties) {
  var classifier:RankingClassifier[String] = null // just a placeholder until train()

  //## SPECIFIC TO SIMPLE
  def train(questionsFilename:String,
            model:RankingModel,
            maxAnswersFromIR:Int,
            classifierProperties:Properties,
            crossValidate:Boolean = true,
            pw:PrintWriter) {

    // Step 0: create the classifier using classifierProperties
    classifier = RankingClassifier.apply(classifierProperties)   // default classifier

    // Step 1: Randomize order of training questions
    //         Do not randomize if we use only the first few questions (for replicability)
    val questionTrainLimit = StringUtils.getInt(props, "ranker.limit_train_questions", 0)
    var questions = questionParser.parse(questionsFilename).toArray
    if(questionTrainLimit == 0)
      questions = randomizeQuestionOrder(questions)

    // If enabled, Limit training set size for debug purposes
    if ((questionTrainLimit != 0) && (questionTrainLimit < (questions.size - 1)))
      questions = questions.slice(0, questionTrainLimit)
    logger.debug("Training with {} questions.", questions.size)

    val queryAnnotations = mkAnnotations(questions)
    val candidates = mkCandidates(queryAnnotations, maxAnswersFromIR)
    val dataset = mkDataset(model, questions, queryAnnotations, candidates)
    classifier.train(dataset)

    logger.debug("crossvalidate", questions.size)
    if(crossValidate) {
      val scores = RankingClassifier.crossValidate[String](dataset, classifierProperties)
      logger.debug("rerank", questions.size)
      val rerankedCandidates = rerank(scores, candidates)
      beforeAndAfterScores(questions, candidates, rerankedCandidates)
      logger.debug("Cross validation complete.")
    }

    // Display weights in debug output
    classifier.displayModel(pw)
    displayMemoryUsage(pw)

    logger.debug("train finished", questions.size)
  }


  def doTestProcedure(
                       model:RankingModel,
                       maxAnswersFromIR:Int,
                       questionsFilename:String,
                       detailedOutput:Boolean,
                       saveTestFile:Boolean,
                       pw:PrintWriter) {

    // save the datums from the test file for offline experiments
    if(saveTestFile && classifier.isInstanceOf[SVMRankingClassifier[String]])
      classifier.asInstanceOf[SVMRankingClassifier[String]].openEvalFile()

    // Step 1: Randomize order of testing questions
    //         Do not randomize if we use only the first few questions (for replicability)
    val questionTestLimit = StringUtils.getInt(props, "ranker.limit_test_questions", 0)
    var questions = questionParser.parse(questionsFilename).toArray
    if(questionTestLimit == 0)
      questions = randomizeQuestionOrder(questions)

    // If enabled, Limit testing set size for debug purposes
    if ((questionTestLimit != 0) && (questionTestLimit < (questions.size - 1)))
      questions = questions.slice(0, questionTestLimit)

    // Display summary statistics for test questions passed in
    logger.debug("Testing with {} questions.", questions.size)
    pw.println("")
    pw.println(" - - - - - - - - - - - - - - - - - - - - - - - - - -")
    pw.println("")
    pw.println(" Testing with {} questions", questions.size)
    pw.println("")

    // Step 2: Generate model features from test questions
    val queryAnnotations = mkAnnotations(questions)
    val candidates = mkCandidates(queryAnnotations, maxAnswersFromIR)
    val testDataset = mkDataset(model, questions, queryAnnotations, candidates)

    // Step 3: Evaluate test questions on trained classifier
    logger.info("Evaluating test questions...")
    val scores = test(testDataset, classifier)
    var rerankedCandidates = rerank(scores, candidates)
    val (scoresBefore, scoresAfter) = beforeAndAfterScores(questions, candidates, rerankedCandidates)
    logger.info("Evaluation complete.")
    if(saveTestFile && classifier.isInstanceOf[SVMRankingClassifier[String]])
      classifier.asInstanceOf[SVMRankingClassifier[String]].closeEvalFile()

    // Step 3A: Compute statistical significance
    val scoreSetBefore = new ArrayBuffer[Scores]
    val scoreSetAfter = new ArrayBuffer[Scores]
    for (i <- 0 until questions.size) {
      val question = questions(i)
      scoreSetBefore.append( scorer.computeScores(question, candidates(i).toList) )
      scoreSetAfter.append( scorer.computeScores(question, rerankedCandidates(i).toList) )
    }
    val scoreSetPValues = scorer.computeBootstrapResampling(scoreSetBefore.toArray, scoreSetAfter.toArray, 100000)   // Use 10000 bootstrap samples

    // Step 4: Store summary result (TODO: Remove this part?)
    if (detailedOutput == true) {
      val scoreDelta = scoresAfter.sent.overlapAt1 - scoresBefore.sent.overlapAt1        // use Precision@1 (Sentence) as scoring metric
      pw.println(" Test set score performance: ")
      pw.println("\t before: " + (scoresBefore.sent.overlapAt1 * 100).formatted("%3.3f") + "%")
      pw.println("\t after: " + (scoresAfter.sent.overlapAt1 * 100).formatted("%3.3f") + "%")
      pw.println("\t delta: " + (scoreDelta * 100).formatted("%3.3f") + "%")
      pw.println("")

      logger.debug(" Test set score performance: ")
      logger.debug("\t before: " + (scoresBefore.sent.overlapAt1 * 100).formatted("%3.3f") + "%")
      logger.debug("\t after: " + (scoresAfter.sent.overlapAt1 * 100).formatted("%3.3f") + "%")
      logger.debug("\t delta: " + (scoreDelta * 100).formatted("%3.3f") + "%")
    }

    // Step 5: Detailed question analysis reporting and histogram generation

    // Initialize score data structures
    var avgScoreSetBaseline = new Scores()
    var avgScoreSetExperimental = new Scores()
    val oneScorer = new Scorer()
    val histogramAt1 = new Histogram("Paragraph Precision @1")
    val histogramMRR = new Histogram("Paragraph MRR")

    // Generate a detailed report for each question in the test set.  Concurrently build up data to populate the
    // summary histograms appended to the bottom of the detailed report.
    for (i <- 0 until questions.size) {
      val question = questions(i)
      val scoreSetBaseline = oneScorer.computeScores(question, candidates(i).toList)
      val scoreSetExperimental = oneScorer.computeScores(question, rerankedCandidates(i).toList)

      // Question analysis (scores and top answers for baseline and experimental methods on each question)
      if (detailedOutput == true) {
        pw.println("================================================================================")
        pw.println(" Question[" + i + "] : + " + question.text)
        pw.println("================================================================================")

        // score
        oneScorer.analysis (question, candidates(i).toList, scoreSetBaseline, pw, i)
        oneScorer.analysisTextMethodCompare(rerankedCandidates(i).toList, pw, "Experimental", scoreSetExperimental, scoreSetBaseline)
      }

      // Averaging
      avgScoreSetBaseline.addToAverage(scoreSetBaseline)
      avgScoreSetExperimental.addToAverage(scoreSetExperimental)

      // Histogram computation and binning
      histogramAt1.addData(scoreSetExperimental.para.overlapAt1 - scoreSetBaseline.para.overlapAt1, i.toString)
      histogramMRR.addData(scoreSetExperimental.para.overlapMRR - scoreSetBaseline.para.overlapMRR, i.toString)

    }

    // Step 6: Display histograms
    pw.println (" - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
    histogramAt1.display(pw)
    pw.println (" - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
    histogramMRR.display(pw)


    // Step 7: Display average scores (and their statistical significance) across the test set for the baseline and experimental runs.
    pw.println (" - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
    oneScorer.analysisTextSummary(pw, "Baseline", avgScoreSetBaseline)
    oneScorer.analysisTextSummaryCompare(pw, "Experimental", avgScoreSetExperimental, avgScoreSetBaseline)
    oneScorer.analysisTextSummaryPValues(pw, "P-Values (Experimental)", scoreSetPValues)

    pw.println ("End of doTestProcedure")
    displayMemoryUsage(pw)


  }




  def doTuningProcedure(
                         model:RankingModel,
                         maxAnswersFromIR:Int,
                         questionsFilename:String,
                         detailedOutput:Boolean,
                         pw:PrintWriter) {

    // tuning is currently done only for feature based models
    // TODO: for kernel-based models, maybe we should tune C here?
    if(model.usesKernels) return

    // Step 1: Randomize order of development questions
    var questions = randomizeQuestionOrder(questionParser.parse(questionsFilename).toArray)

    // If enabled, Limit development set size for debug purposes
    val questionDevLimit = StringUtils.getInt(props, "ranker.limit_dev_questions", 0)
    if ((questionDevLimit != 0) && (questionDevLimit < (questions.size - 1))) questions = questions.slice(0, questionDevLimit)

    // Display summary statistics for development questions passed in
    logger.info("Tuning with {} questions.", questions.size)
    pw.println("")
    pw.println(" - - - - - - - - - - - - - - - - - - - - - - - - - -")
    pw.println("")
    pw.println(" Tuning with {} questions", questions.size)
    pw.println("")

    // Step 2: Generate model features from development questions to use for tuning
    val queryAnnotations = mkAnnotations(questions)
    val candidates = mkCandidates(queryAnnotations, maxAnswersFromIR)
    val testDataset = mkDataset(model, questions, queryAnnotations, candidates)


    // Step 3: Evaluate development questions using the trained classifier, at a variety of different weight-clipping
    // (or, loosely interpretted, regularization) thresholds.  Note the top-performing threshold.
    var topScore:Double = 0.0
    var topThresh:Double = 0.0
    val clipThresholds = Array[Double](0.75, 0.5, 0.25, 0.1, 0.075, 0.05, 0.025, 0.01, 0.0075, 0.005, 0.0025, 0.001, 0.0001, 0.0)
    pw.println (" ======================================================================================================================= ")
    pw.println (" ============================================== TUNING ================================================================= ")
    pw.println (" ======================================================================================================================= ")

    // Step 3A: For each threshold
    for (c <- 0 until clipThresholds.size) {
      // Select threshold
      val clipThresh = clipThresholds(c)
      pw.println(" ======================================================================================================================= ")
      pw.println("         Weight clip threshold: " + clipThresh + " * IRweight")
      pw.println(" ======================================================================================================================= ")

      // Step 3B: Clip the weights to current threshold
      classifier match {
        case c: SVMRankingClassifier[String] => c.clipWeightsRelativeToOneFeature(clipThresh, "ir")
      }

      pw.flush()

      // Step 3C: Evaluate the performance of the clipped model on the development questions
      logger.debug("Evaluating development questions with on model with clipped weights...")
      val scores = test(testDataset, classifier)
      var rerankedCandidates = rerank(scores, candidates)
      val (scoresBefore, scoresAfter) = beforeAndAfterScores(questions, candidates, rerankedCandidates)
      logger.debug("Evaluation complete.")

      // Step 3D: Compute statistical significance of development performance
      val scoreSetBefore = new ArrayBuffer[Scores]
      val scoreSetAfter = new ArrayBuffer[Scores]
      for (i <- 0 until questions.size) {
        val question = questions(i)
        scoreSetBefore.append(scorer.computeScores(question, candidates(i).toList))
        scoreSetAfter.append(scorer.computeScores(question, rerankedCandidates(i).toList))
      }
      val scoreSetPValues = scorer.computeBootstrapResampling(scoreSetBefore.toArray, scoreSetAfter.toArray, 100000)   // Bootstrap resampling, 10000 iterations


      // Step 3E: Output summary of performance for current threshold
      if (detailedOutput == true) {
        // Step 4: Store summary result
        val scoreDelta = scoresAfter.sent.overlapAt1 - scoresBefore.sent.overlapAt1 // use Precision@1 (Sentence) as scoring metric
        pw.println(" Test set score performance: ")
        pw.println("\t before: " + (scoresBefore.sent.overlapAt1 * 100).formatted("%3.3f") + "%")
        pw.println("\t after: " + (scoresAfter.sent.overlapAt1 * 100).formatted("%3.3f") + "%")
        pw.println("\t delta: " + (scoreDelta * 100).formatted("%3.3f") + "%")
        pw.println("")

        logger.debug(" Test set score performance: ")
        logger.debug("\t before: " + (scoresBefore.sent.overlapAt1 * 100).formatted("%3.3f") + "%")
        logger.debug("\t after: " + (scoresAfter.sent.overlapAt1 * 100).formatted("%3.3f") + "%")
        logger.debug("\t delta: " + (scoreDelta * 100).formatted("%3.3f") + "%")
      }

      // Step 3F: Compute histogram of performance at current threshold
      // Initialization
      var avgScoreSetBaseline = new Scores()
      var avgScoreSetExperimental = new Scores()
      val oneScorer = new Scorer()
      val histogramAt1 = new Histogram("Paragraph Precision @1")
      val histogramMRR = new Histogram("Paragraph MRR")

      // Histogram computation (compute across each question in the development set)
      for (i <- 0 until questions.size) {
        val question = questions(i)
        val scoreSetBaseline = oneScorer.computeScores(question, candidates(i).toList)
        val scoreSetExperimental = oneScorer.computeScores(question, rerankedCandidates(i).toList)

        // Averaging
        avgScoreSetBaseline.addToAverage(scoreSetBaseline)
        avgScoreSetExperimental.addToAverage(scoreSetExperimental)

        // Histogram computation and binning
        histogramAt1.addData(scoreSetExperimental.para.overlapAt1 - scoreSetBaseline.para.overlapAt1, i.toString)
        histogramMRR.addData(scoreSetExperimental.para.overlapMRR - scoreSetBaseline.para.overlapMRR, i.toString)

      }

      // Output detailed histogram performance report
      pw.println (" - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
      histogramAt1.display(pw)
      pw.println (" - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
      histogramMRR.display(pw)

      // Step 3G: Display average scores (and their statistical significance) across the test set for the baseline and experimental runs at this weight-clipping threshold.
      pw.println(" - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
      oneScorer.analysisTextSummary(pw, "Baseline", avgScoreSetBaseline)
      oneScorer.analysisTextSummaryCompare(pw, "Experimental", avgScoreSetExperimental, avgScoreSetBaseline)
      oneScorer.analysisTextSummaryPValues(pw, "P-Values (Experimental)", scoreSetPValues)

      pw.flush()

      // Step 3H: Track the best performing model and weight clipping threshold
      if (avgScoreSetExperimental.para.overlapAt1 > topScore) {
        topScore = avgScoreSetExperimental.para.overlapAt1
        topThresh = clipThresh
      }

    }


    // Step 4: Once all weight-clipping thresholds have been evaluated, pick the top performing threshold, and set the classifier to that threshold.
    pw.println ("")
    pw.println (" ======================================================================================================================= ")
    pw.println (" Top performing clip threshold:")
    pw.println (" thresh: " + topThresh + " \t score(MRR): " + topScore )
    pw.println (" Tuning procedure complete. ")
    pw.println (" ======================================================================================================================= ")

    classifier match {
      case c: SVMRankingClassifier[String] => c.clipWeightsRelativeToOneFeature(topThresh, "ir")
    }


    pw.println ("End of doTuningProcedure()")
    displayMemoryUsage(pw)

    // Return the tuned, weight-clipped classifier.
    // classifier

  }


}
