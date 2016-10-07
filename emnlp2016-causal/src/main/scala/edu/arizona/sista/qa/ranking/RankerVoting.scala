package edu.arizona.sista.qa.ranking

import java.util.Properties
import edu.arizona.sista.learning._
import java.io.PrintWriter
import edu.arizona.sista.utils.{VersionUtils, StringUtils}
import collection.mutable.{ListBuffer, ArrayBuffer}
import edu.arizona.sista.qa.scorer.{Question, Histogram, Scorer, Scores}
//import edu.arizona.sista.qa.discourse._
import edu.arizona.sista.utils.MathUtils.softmax
import scala.{Array, Tuple2}
import edu.arizona.sista.processors.{Processor, Document}
import edu.arizona.sista.qa.retrieval._
import edu.arizona.sista.qa.translation.{TranslationMultisetModel}
import org.slf4j.LoggerFactory
import edu.arizona.sista.qa.preprocessing.yahoo.deep.{CQAQuestion, CQAQuestionParser}
import util.control.Breaks._
import edu.arizona.sista.qa.index.TermFilter
import edu.arizona.sista.qa.word2vec._
import edu.arizona.sista.processors.fastnlp.FastNLPProcessor
import scala.Some
import edu.arizona.sista.qa.preprocessing.yahoo.YahooKerasUtils


/**
 * Voting ranker that allows combining multiple models and/or classifiers together into one reranking framework
 * User: peter
 * Date: 8/27/13
 */
class RankerVoting(props: Properties) extends RankerSimple(props: Properties) {
  val classifiers = new ArrayBuffer[RankingClassifier[String]]
  // Classifiers
  val models = new ArrayBuffer[RankingModel] // Ranking models

  // for rescaling the datasets to be within a consistent range
  // rescaleFeatures is populated during training (one entry for each model), and used during testing
  val rescaleFeatures = StringUtils.getBool(props, "ranker.rescale_features.enabled", false)
  var scaleRanges = new ArrayBuffer[ScaleRange[String]]
  var rescaleLower = StringUtils.getDouble(props, "ranker.rescale_features.lower", -1.0)
  var rescaleUpper = StringUtils.getDouble(props, "ranker.rescale_features.upper", 1.0)

  var lambda: Double = StringUtils.getDouble(props, "ranker.lambda", 0.5)

  // CQA-specific variables
  var CQAEnabled: Boolean = StringUtils.getBool(props, "cqa_enabled", false)
  lazy val CQAQuestionParser = new CQAQuestionParser
  var CQAIndex = Array.empty[CQAQuestion]
  if (CQAEnabled) CQAIndex = buildCQAIndex()
  // All CQA Questions across the train/dev/test corpus
  //lazy val queryProcessor:Processor = new CoreNLPProcessor()
  lazy val queryProcessor: Processor = new FastNLPProcessor()

  // CQA-Specific
  def buildCQAIndex(): Array[CQAQuestion] = {
    val filename = props.getProperty("cqaindex", "")
    if (filename == "") throw new RuntimeException("ERROR: Could not build CQA index. ('cqaindex' property is empty)")
    logger.info("Building and annotating CQA Index.  This may take several minutes.")
    CQAQuestionParser.load(filename, true) // Load and annotate
  }

  // CQA-Specific
  def CQAIRScore(queryAnnotation: Document, answers: Array[AnswerCandidate]): Array[AnswerCandidate] = {
    val indexDir = props.getProperty("index")
    val docSyntaxWeight = StringUtils.getDouble(props, "retrieval.doc_syntax_weight", 0.0)
    val answerSyntaxWeight = StringUtils.getDouble(props, "retrieval.answer_syntax_weight", 0.0)
    val termFilter = new TermFilter

    var paragraphScorer: PassageScorer = null
    if (answerSyntaxWeight == 0.0) {
      // no syntax; just use the BOW model
      paragraphScorer =
        new BagOfWordsPassageScorer(
          termFilter,
          indexDir,
          queryAnnotation)
    } else {
      // meta model combines BOW with syntax
      paragraphScorer =
        new MetaPassageScorer(
          termFilter,
          indexDir,
          queryAnnotation,
          answerSyntaxWeight)
    }

    for (answer <- answers) {
      val answerScore = paragraphScorer.score(answer)
      answer.setScore(answerScore, 0.0) // Set answer score, and set docWeight to 0.0 (so the combined answer score is entirely weighted to this score)
    }

    answers.sortBy(-_.score)
  }

  // CQA-Specific
  def mkCandidatesCQA(questions: Array[Question], questionAnnotation: Array[Document]): Array[Array[AnswerCandidate]] = {
    val candidates = new ArrayBuffer[Array[AnswerCandidate]]

    // For each question
    for (qIdx <- 0 until questions.size) {
      val question = questions(qIdx)
      println("qIdx: " + qIdx + " question:" + question.text)

      // Step 1: Find question ID
      val answer = question.goldAnswers(0) // Note, here just takes the first gold answer.  CQA currently should only ever have one gold answer.
      val docidParts = answer.docid.split("_") // CQA DocID format is <docid>_A<answer_num>.
      val queryDocid = docidParts(0) // extract <docid> portion

      // Step 2: Use DocID to find question and answer candidates in CQA Index
      var CQAQuestion1: CQAQuestion = null
      breakable {
        for (i <- 0 until CQAIndex.size) {
          if (CQAIndex(i).docid.compareTo(queryDocid) == 0) {
            CQAQuestion1 = CQAIndex(i)
            break
          }
        }
      }
      if (CQAQuestion1 == null) throw new RuntimeException("mkCandidates:ERROR: Query with docid (" + queryDocid + ") not found in CQA Index.")

      // Step 3: Generate a list of answer candidates from the CQA Index
      val answersPreIR = new ArrayBuffer[AnswerCandidate]()
      for (i <- 0 until CQAQuestion1.answers.size) {
        // make annotation
        val annotation = mkPartialAnnotation(CQAQuestion1.answers(i).text)
        val docCandidate = new DocumentCandidate(CQAQuestion1.answers(i).docid,
          annotation,
          mkFakePids(annotation), // paragraph IDs
          0.0) // document score

        val candidate = new AnswerCandidate(docCandidate, 0, annotation.sentences.size)
        // TODO: Candidate score needs to be set to its IR score
        answersPreIR.append(candidate)
      }

      // Append IR scores to each candidate, and sort based in IR scores
      val answers = CQAIRScore(questionAnnotation(qIdx), answersPreIR.toArray)

      // Step 4: Sort candidates based on their IR score
      val baseline = props.getProperty("cqa_baseline", "ir")

      if (baseline == "ir") {
        // Step 4A: Baseline: IR
        // Default -- CQAIRScore already scores questions based on their IR scores
        candidates += answers

      } else if (baseline == "random") {
        // Step 4B: Baseline: Randomized
        var rand = new ArrayBuffer[AnswerCandidate]
        rand = rand ++= answers
        val answersShuffled = util.Random.shuffle(rand).toArray
        candidates += answersShuffled.toArray

      } else {
        // Unknown
        throw new RuntimeException("ERROR: Unknown cqa_baseline in properties file")
      }

      // Step 5: Add sorted candidates to the candidates array
      //candidates += answers.toArray
    }
    assert(candidates.size == questions.size)
    candidates.toArray

  }


  def mkFakePids(doc: Document): Array[Int] = {
    val pids = new Array[Int](doc.sentences.size)
    for (i <- 0 until pids.size) {
      pids(i) = 1 // note: are paragraph IDs 0 or 1 indexed?
    }
    pids
  }

  def mkPartialAnnotation(text: String): Document = {
    val doc = queryProcessor.mkDocument(text)
    queryProcessor.tagPartsOfSpeech(doc)
    queryProcessor.lemmatize(doc)
    doc.clear
    doc
  }


  def trainV(questionsFilename: String,
             maxAnswersFromIR: Int,
             classifierProperties: Properties,
             crossValidate: Boolean = true,
             pw: PrintWriter) {

    // Step 0: Ensure that we have a classifier for each model.
    if (classifiers.size < models.size) {
      for (i <- 0 until (models.size - classifiers.size)) {
        classifiers.append(RankingClassifier.apply(classifierProperties))
      }
    }

    // Step 1: Randomize order of training questions
    //         Do not randomize if we use only the first few questions (for replicability)
    val questionTrainLimit = StringUtils.getInt(props, "ranker.limit_train_questions", 0)
    var questions = questionParser.parse(questionsFilename).toArray
    if (questionTrainLimit == 0)
      questions = randomizeQuestionOrder(questions)

    // If enabled, Limit training set size for debug purposes
    if ((questionTrainLimit != 0) && (questionTrainLimit < (questions.size - 1)))
      questions = questions.slice(0, questionTrainLimit)
    logger.debug("Training with {} questions.", questions.size)

    val queryAnnotations = mkAnnotations(questions)

    var candidates = Array.empty[Array[AnswerCandidate]]
    if (CQAEnabled == true) {
      candidates = mkCandidatesCQA(questions, queryAnnotations)
    } else {
      candidates = mkCandidates(queryAnnotations, maxAnswersFromIR)
    }

    // Optionally dump the info to be input to keras (processed offline)
    val dumpQuestionInfoForKeras = StringUtils.getBool(props, "ranker.dump_keras_input_enabled", false)
    val kerasDumpDir = props.getProperty("ranker.dump_to_directory")
    if (dumpQuestionInfoForKeras) {
      YahooKerasUtils.dumpYahooDataforKeras(kerasDumpDir, "train", candidates, questions, queryAnnotations)
    }


    for (i <- 0 until models.size) {
      val model = models(i)
      val dataset = mkDataset(model, questions, queryAnnotations, candidates)
      classifiers(i).train(dataset)
      pw.println("Model " + i + " Weights")
      classifiers(i).displayModel(pw)
      if (rescaleFeatures)
        scaleRanges.append(Datasets.svmScaleRankingDataset(dataset, rescaleLower, rescaleUpper))
    }

/*
      // Temporarily disabled -- we rarely make use of crossvalidation
      logger.debug("crossvalidate", questions.size)
      if(crossValidate) {


        val scores = RankingClassifier.crossValidate[String](dataset, classifierProperties)
        logger.debug("rerank", questions.size)
        val rerankedCandidates = rerank(scores, candidates)
        beforeAndAfterScores(questions, candidates, rerankedCandidates)
        logger.debug("Cross validation complete.")
      }
*/

    // Display weights in debug output
    displayMemoryUsage(pw)

    logger.debug("train finished", questions.size)
  }


  // Combines the scores from multiple models (Model x Question x Answer Candidate) into a single set of scores (Question x Answer Candidate)
  def combineScores(scores: Array[Array[Array[Double]]]): Array[Array[Double]] = {
    val outScores = new Array[Array[Double]](scores(0).size)
    for (a <- 0 until scores(0).size) {
      outScores(a) = new Array[Double](scores(0)(a).size)
    }

    // Add all the scores for a given answer candidate together
    for (i <- 0 until scores.size) {
      // Lambda model
      var weight = 1.0
      if (scores.size == 2) {
        if (i == 0) weight = lambda
        if (i == 1) weight = 1 - lambda
      }

      // Combine scores
      val scoreSet = scores(i)
      for (q <- 0 until scoreSet.size) {
        for (ac <- 0 until scoreSet(q).size) {
          if (i == 0) {
            outScores(q)(ac) = scoreSet(q)(ac) * weight
          } else {
            outScores(q)(ac) += scoreSet(q)(ac) * weight
          }
        }
      }

    }

    outScores
  }


  // Returns an array of indicies of reranked scores.  Used to generate voting/reranking debug output in doTestProcedureV.
  def rerankIndices(scores: Array[Double]): Array[Int] = {
    val after = new Array[Int](scores.size)

    val queryRanks = new ListBuffer[(Int, Double)]
    for (i <- 0 until scores.size) {
      queryRanks += new Tuple2(i, scores(i))
    }
    val sortedRanks = queryRanks.toList.sortBy(0 - _._2).toArray

    val ranked = new Array[Double](scores.size)
    for (i <- 0 until sortedRanks.size) {
      after(sortedRanks(i)._1) = i + 1
    }

    after
  }


  def doTestProcedureV(
                        maxAnswersFromIR: Int,
                        questionsFilename: String,
                        detailedOutput: Boolean,
                        pw: PrintWriter) {


    // Step 1: Randomize order of testing questions
    //         Do not randomize if we use only the first few questions (for replicability)
    val questionTestLimit = StringUtils.getInt(props, "ranker.limit_test_questions", 0)
    var questions = questionParser.parse(questionsFilename).toArray
    if (questionTestLimit == 0)
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

    var candidates = Array.empty[Array[AnswerCandidate]]
    if (CQAEnabled == true) {
      candidates = mkCandidatesCQA(questions, queryAnnotations)
    } else {
      candidates = mkCandidates(queryAnnotations, maxAnswersFromIR)
    }

    // Optionally dump the info to be input to keras (processed offline)
    val dumpQuestionInfoForKeras = StringUtils.getBool(props, "ranker.dump_keras_input_enabled", false)
    val kerasDumpDir = props.getProperty("ranker.dump_to_directory")
    if (dumpQuestionInfoForKeras) {
      YahooKerasUtils.dumpYahooDataforKeras(kerasDumpDir, "test", candidates, questions, queryAnnotations)
    }


    val scores = new ArrayBuffer[Array[Array[Double]]]
    val scoresProb = new ArrayBuffer[Array[Array[Double]]]

    // Step 3: Evaluate test questions on each trained classifier model in the voting ranker
    logger.info("Evaluating questions...")

    for (i <- 0 until models.size) {
      logger.debug("Evaluating test questions (Model " + i + ") ...")
      val model = models(i) // Select one model
      val maybeRescale = rescaleFeatures match {
          case true => Some(scaleRanges(i))
          case false => None
        }
      val testDataset = mkDataset(model, questions, queryAnnotations, candidates, maybeRescale) // Generate test dataset
      if (maybeRescale.isDefined) {
        val pwScaleRange = new PrintWriter("/lhome/bsharp/causal/yahoo/EA/test_scaleRange.txt")
        val sr = maybeRescale.get
        sr.saveTo(pwScaleRange)
        pwScaleRange.close()
      }
      val scoresModel = test(testDataset, classifiers(i)) // Evaluate model on candidates
      scores.append(scoresModel) // Store model scores

      // While we generate the raw scores above, here we also generate the normalized scores (or probabilities),
      // which are conceptually better to combine and give the voting system higher performance.
      val scoresProbTemp = new Array[Array[Double]](scoresModel.size)
      for (j <- 0 until scoresProbTemp.size) scoresProbTemp(j) = softmax(scoresModel(j).toIterable).toArray
      scoresProb.append(scoresProbTemp)
    }


    // Step 3A: Combine the scores from each model into a single score
    logger.debug("Combining scores... ")
    var combinedScoresRaw = combineScores(scores.toArray)
    var combinedScoresProb = combineScores(scoresProb.toArray)
    val rerankedCandidates = rerank(combinedScoresProb, candidates) // Rerank based on combined probability scores
    val (scoresBefore, scoresAfter) = beforeAndAfterScores(questions, candidates, rerankedCandidates)
    logger.debug("Evaluation complete.")

    // Optional Error Analysis Output
    //val errorFilenamePrefix = "errorout"
    if (StringUtils.getBool(props, "erroranalysis.enabled", false) == true) {
      //val errorFilenamePrefix = props.getProperty("output_report_filename")
      val errorFilenamePrefix = "/lhome/bsharp/causal/yahoo/EA/EA_V+cB"
      createErrorAnalysis(models.toArray, questions, queryAnnotations, rerankedCandidates, errorFilenamePrefix)
    }

    // Step 3A: Compute statistical significance
    val scoreSetBefore = new ArrayBuffer[Scores]
    val scoreSetAfter = new ArrayBuffer[Scores]
    for (i <- 0 until questions.size) {
      val question = questions(i)
      scoreSetBefore.append(scorer.computeScores(question, candidates(i).toList))
      scoreSetAfter.append(scorer.computeScores(question, rerankedCandidates(i).toList))
    }
    val scoreSetPValues = scorer.computeBootstrapResampling(scoreSetBefore.toArray, scoreSetAfter.toArray, 100000) // Use 10000 bootstrap samples

    // Step 4: Store summary result (TODO: Remove this part?)
    if (detailedOutput == true) {
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
        oneScorer.analysis(question, candidates(i).toList, scoreSetBaseline, pw, i)
        oneScorer.analysisTextMethodCompare(rerankedCandidates(i).toList, pw, "Experimental", scoreSetExperimental, scoreSetBaseline)

        // DEBUG: Include ranking information for each model, as well as combined ranking information
        for (j <- 0 until scores.size) {

          val scoreSet = scores(j) // Questions x Answer Candidates (raw)
          val scoresAC = scoreSet(i) // Answer Candidates
          val scoreSetProb = scoresProb(j) // Questions x Answer Candidates (prob)
          val scoresACProb = scoreSetProb(i) // Answer Candidates

          val indices = rerankIndices(scoresAC)
          val finalOrder = new Array[(Int, Int, Double, Double)](indices.size)
          for (k <- 0 until indices.size) {
            finalOrder(k) = (k + 1, indices(k), scoresAC(k), scoresACProb(k))
          }
          val sortedOrder = finalOrder.sortBy(_._2)
          pw.print("Ranking (Model " + j + ") Reordered Indices: \t")
          for (k <- 0 until sortedOrder.size) {
            pw.print(sortedOrder(k)._1.formatted("%3d") + "(" + sortedOrder(k)._3.formatted("%3.4f") + "a, " + sortedOrder(k)._4.formatted("%3.4f") + "n)")
          }
          pw.println("")
          pw.println("")
        }

        // DEBUG: Include final ranking information for the combined voting model
        val indices = rerankIndices(combinedScoresProb(i))
        val finalOrder = new Array[(Int, Int, Double, Double)](indices.size)
        for (k <- 0 until indices.size) {
          finalOrder(k) = (k + 1, indices(k), combinedScoresRaw(i)(k), combinedScoresProb(i)(k))
        }
        val sortedOrder = finalOrder.sortBy(_._2)
        pw.print("Ranking (Model C) Reordered Indices: \t")
        for (k <- 0 until sortedOrder.size) {
          pw.print(sortedOrder(k)._1.formatted("%3d") + "(" + sortedOrder(k)._3.formatted("%3.4f") + "a, " + sortedOrder(k)._4.formatted("%3.4f") + "n)")
        }
        pw.println("")
        pw.println("")

      }

      // Averaging
      avgScoreSetBaseline.addToAverage(scoreSetBaseline)
      avgScoreSetExperimental.addToAverage(scoreSetExperimental)

      // Histogram computation and binning
      histogramAt1.addData(scoreSetExperimental.para.overlapAt1 - scoreSetBaseline.para.overlapAt1, i.toString)
      histogramMRR.addData(scoreSetExperimental.para.overlapMRR - scoreSetBaseline.para.overlapMRR, i.toString)

    }

    // Step 6: Display histograms
    pw.println(" - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
    histogramAt1.display(pw)
    pw.println(" - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
    histogramMRR.display(pw)


    // Step 7: Display average scores (and their statistical significance) across the test set for the baseline and experimental runs.
    pw.println(" - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
    oneScorer.analysisTextSummary(pw, "Baseline", avgScoreSetBaseline)
    oneScorer.analysisTextSummaryCompare(pw, "Experimental", avgScoreSetExperimental, avgScoreSetBaseline)
    oneScorer.analysisTextSummaryPValues(pw, "P-Values (Experimental)", scoreSetPValues)

    // Step 8: Write scores arrays to file, so that more statistical comparisons can be run later
    val statsFilenameBaseline: String = props.getProperty("output_report_filename") + ".stats.baseline_test"
    scorer.saveScoresArray(scoreSetBefore.toArray, statsFilenameBaseline)
    val statsFilenameAfter: String = props.getProperty("output_report_filename") + ".stats.model_combined_test"
    scorer.saveScoresArray(scoreSetAfter.toArray, statsFilenameAfter)


    pw.println("End of doTestProcedure")
    displayMemoryUsage(pw)

  }


  def doTestProcedureVotingManual(
                                   maxAnswersFromIR: Int,
                                   questions: Array[Question]): (Array[Array[AnswerCandidate]], Array[Scores], Array[Scores]) = {
    // A stripped-down test procedure that simply returns a list of answer candidates, as well as P@1/MRR scores.  Intended for use with QA.eval() and the API

    // Step 1: Generate baseline IR candidates
    val queryAnnotations = mkAnnotations(questions)

    var candidates = Array.empty[Array[AnswerCandidate]]
    if (CQAEnabled == true) {
      candidates = mkCandidatesCQA(questions, queryAnnotations)
    } else {
      candidates = mkCandidates(queryAnnotations, maxAnswersFromIR)
    }


    val scores = new ArrayBuffer[Array[Array[Double]]]
    val scoresProb = new ArrayBuffer[Array[Array[Double]]]

    val oneScorer = new Scorer()
    val scoresBaseline = new ArrayBuffer[Scores]
    val scoresExperimental = new ArrayBuffer[Scores]

    // Step 2: Generate model features for test questions, and evaluate test questions on each trained classifier model in the voting ranker
    logger.info("Evaluating test questions...")
    for (i <- 0 until models.size) {
      logger.debug("Evaluating test questions (Model " + i + ")...")
      val model = models(i) // Select one model
      val maybeRescale = rescaleFeatures match {
          case true => Some(scaleRanges(i))
          case false => None
        }
      val testDataset = mkDataset(model, questions, queryAnnotations, candidates, maybeRescale) // Generate test dataset
      val scoresModel = test(testDataset, classifiers(i)) // Evaluate model on candidates
      scores.append(scoresModel) // Store model scores

      // While we generate the raw scores above, here we also generate the normalized scores (or probabilities),
      // which are conceptually better to combine and give the voting system higher performance.
      val scoresProbTemp = new Array[Array[Double]](scoresModel.size)
      for (j <- 0 until scoresProbTemp.size) scoresProbTemp(j) = softmax(scoresModel(j).toIterable).toArray
      scoresProb.append(scoresProbTemp)
    }

    // Step 3: Combine the scores from each model into a single score using the voting model
    logger.debug("Combining scores... ")
    var combinedScoresProb = combineScores(scoresProb.toArray)
    val rerankedCandidates = rerank(combinedScoresProb, candidates) // Rerank based on combined probability scores

    // Step 4: Compute P@1 and MRR Scores for each question
    for (i <- 0 until questions.size) {
      val question = questions(i)
      scoresBaseline.append(oneScorer.computeScores(question, candidates(i).toList))
      scoresExperimental.append(oneScorer.computeScores(question, rerankedCandidates(i).toList))
    }

    logger.debug("Complete... ")
    return (rerankedCandidates, scoresBaseline.toArray, scoresExperimental.toArray)

  }


  // Supports tuning procedures
  def reportDelta(before: Scores, after: Scores, tuningMetric: String, pw: PrintWriter) {
    if (tuningMetric == "para_p1") {
      pw.println(" Test set score performance (Paragraph P@1): ")
      pw.println("\t before: " + (before.para.overlapAt1 * 100).formatted("%3.3f") + "%")
      pw.println("\t after: " + (after.para.overlapAt1 * 100).formatted("%3.3f") + "%")
      pw.println("\t delta: " + ((after.para.overlapAt1 - before.para.overlapAt1) * 100).formatted("%3.3f") + "%")

      pw.println(" Test set score performance (Paragraph P@1): ")
      logger.debug("\t before: " + (before.para.overlapMRR * 100).formatted("%3.3f") + "%")
      logger.debug("\t after: " + (after.para.overlapMRR * 100).formatted("%3.3f") + "%")
      logger.debug("\t delta: " + ((after.para.overlapAt1 - before.para.overlapAt1) * 100).formatted("%3.3f") + "%")
    }

    if (tuningMetric == "para_mrr") {
      pw.println(" Test set score performance (Paragraph MRR): ")
      pw.println("\t before: " + (before.para.overlapMRR * 100).formatted("%3.3f") + "%")
      pw.println("\t after: " + (after.para.overlapMRR * 100).formatted("%3.3f") + "%")
      pw.println("\t delta: " + ((after.para.overlapMRR - before.para.overlapMRR) * 100).formatted("%3.3f") + "%")

      logger.debug(" Test set score performance (Paragraph MRR): ")
      logger.debug("\t before: " + (before.para.overlapMRR * 100).formatted("%3.3f") + "%")
      logger.debug("\t after: " + (after.para.overlapMRR * 100).formatted("%3.3f") + "%")
      logger.debug("\t delta: " + ((after.para.overlapMRR - before.para.overlapMRR) * 100).formatted("%3.3f") + "%")
    }
    pw.println("")
  }


  def tuneClippingWeights(
                           questions: Array[Question],
                           queryAnnotations: Array[Document],
                           candidates: Array[Array[AnswerCandidate]],
                           testDatasets: Array[RankingDataset[String]],
                           pw: PrintWriter) {

    val clipThresholds = Array[Double](0.75, 0.5, 0.25, 0.1, 0.075, 0.05, 0.025, 0.01, 0.0075, 0.005, 0.0025, 0.001, 0.0001, 0.0)
    val tuningMetric = props.getProperty("ranker.tuning_metric", "para_p1")
    val votingMethod = props.getProperty("voting.method", "UNSPECIFIED")
    var topScore: Double = 0.0
    var topThresh: Double = 0.0
    var topScoreSetAfter = new Array[Scores](0)
    var baselineScoreSet = new Array[Scores](0)

    var numClassifiers = models.size
    if (votingMethod == "oneclassifier") numClassifiers = 1 // the "oneclassifier" method has many models but one classifier


    for (modelIdx <- 0 until numClassifiers) {
      topScore = 0.0
      topThresh = 0.0

      // For each clipping threshold for a given model
      for (c <- 0 until clipThresholds.size) {
        // Select threshold
        val clipThresh = clipThresholds(c)
        pw.println(" ======================================================================================================================= ")
        pw.println("         Model + " + modelIdx + " Weight clip threshold: " + clipThresh + " * IRweight")
        pw.println(" ======================================================================================================================= ")
        pw.flush()

        // Step 3B: Clip the weights to current threshold
        classifiers(modelIdx) match {
          // Clip weights of classifier
          case cl: SVMRankingClassifier[String] => cl.clipWeightsRelativeToOneFeature(clipThresh, "ir")
        }

        // Step 3C: Evaluate the performance of the clipped model on the development questions
        logger.debug("Evaluating development questions with on model with clipped weights...")
        val scoresModel = test(testDatasets(modelIdx), classifiers(modelIdx)) // Evaluate model on candidates
        var rerankedCandidates = rerank(scoresModel, candidates)
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
        val scoreSetPValues = scorer.computeBootstrapResampling(scoreSetBefore.toArray, scoreSetAfter.toArray, 100000) // Bootstrap resampling, 10000 iterations

        // Step 3E: Output summary of performance for current threshold
        val scoreDelta = scoresAfter.para.overlapMRR - scoresBefore.para.overlapMRR // use Precision@1 (Sentence) as scoring metric
        reportDelta(scoresBefore, scoresAfter, tuningMetric, pw)

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
        pw.println(" - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
        histogramAt1.display(pw)
        pw.println(" - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
        histogramMRR.display(pw)

        // Step 3G: Display average scores (and their statistical significance) across the test set for the baseline and experimental runs at this weight-clipping threshold.
        pw.println(" - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
        oneScorer.analysisTextSummary(pw, "Baseline", avgScoreSetBaseline)
        oneScorer.analysisTextSummaryCompare(pw, "Experimental", avgScoreSetExperimental, avgScoreSetBaseline)
        oneScorer.analysisTextSummaryPValues(pw, "P-Values (Experimental)", scoreSetPValues)

        pw.flush()

        // Step 3H: Track the best performing model and weight clipping threshold
        if (tuningMetric == "para_p1") {
          if (avgScoreSetExperimental.para.overlapAt1 > topScore) {
            topScore = avgScoreSetExperimental.para.overlapAt1
            topThresh = clipThresh
            topScoreSetAfter = scoreSetAfter.toArray
            baselineScoreSet = scoreSetBefore.toArray // Note: this doesn't need to be saved every time, since the baseline doesn't change -- clean up later
          }
        } else if (tuningMetric == "para_mrr") {
          if (avgScoreSetExperimental.para.overlapMRR > topScore) {
            topScore = avgScoreSetExperimental.para.overlapMRR
            topThresh = clipThresh
            topScoreSetAfter = scoreSetAfter.toArray
            baselineScoreSet = scoreSetBefore.toArray // Note: this doesn't need to be saved every time, since the baseline doesn't change -- clean up later
          }
        }

      }


      // Step 4: Once all weight-clipping thresholds have been evaluated, pick the top performing threshold, and set the classifier to that threshold.
      pw.println("")
      pw.println(" ======================================================================================================================= ")
      pw.println(" Model " + modelIdx + " Top performing clip threshold:")
      if (tuningMetric == "para_p1") pw.println(" thresh: " + topThresh + " \t score(Paragraph P@1): " + topScore)
      if (tuningMetric == "para_mrr") pw.println(" thresh: " + topThresh + " \t score(Paragraph MRR): " + topScore)
      pw.println(" Tuning procedure complete. ")
      pw.println(" ======================================================================================================================= ")

      classifiers(modelIdx) match {
        case cl: SVMRankingClassifier[String] => cl.clipWeightsRelativeToOneFeature(topThresh, "ir")
      }

      // Save top array of scores, so we can come back to compute more statistics on them later
      val statsFilenameBaseline: String = props.getProperty("output_report_filename") + ".stats.baseline_dev"
      scorer.saveScoresArray(baselineScoreSet, statsFilenameBaseline)
      val statsFilenameAfter: String = props.getProperty("output_report_filename") + ".stats.model" + modelIdx + "_tuned"
      scorer.saveScoresArray(topScoreSetAfter, statsFilenameAfter)
    }

  }

  def tuneLambda(
                  questions: Array[Question],
                  queryAnnotations: Array[Document],
                  candidates: Array[Array[AnswerCandidate]],
                  pw: PrintWriter) {

    val lambdaValues = Array[Double](1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.0)
    val tuningMetric = props.getProperty("ranker.tuning_metric", "para_p1")
    var topScore: Double = 0.0
    var topThresh: Double = 0.0
    var topScoreSetAfter = new Array[Scores](0)

    for (w <- 0 until lambdaValues.size) {
      // Select threshold
      val lambdaValue = lambdaValues(w)
      lambda = lambdaValue

      logger.debug("Lambda = " + lambdaValue)
      pw.println(" ======================================================================================================================= ")
      pw.println("         Lambda: " + lambdaValue)
      pw.println(" ======================================================================================================================= ")
      pw.flush()

      // Step 3C: Evaluate the performance of the clipped model on the development questions
      logger.debug("Evaluating development questions with a new lambda value for tuning...")
      val scores = new ArrayBuffer[Array[Array[Double]]]
      val scoresProb = new ArrayBuffer[Array[Array[Double]]]

      logger.debug("Generating Scores...")
      for (i <- 0 until models.size) {
        logger.debug("Generating scores: Model " + i)
        val model = models(i) // Select one model
        val maybeRescale = rescaleFeatures match {
            case true => Some(scaleRanges(i))
            case false => None
          }
        val testDataset = mkDataset(model, questions, queryAnnotations, candidates, maybeRescale) // Generate test dataset
        val scoresModel = test(testDataset, classifiers(i)) // Evaluate model on candidates
        scores.append(scoresModel) // Store model scores

        // Normalized scores (probabilities)
        val scoresProbTemp = new Array[Array[Double]](scoresModel.size)
        for (j <- 0 until scoresProbTemp.size) scoresProbTemp(j) = softmax(scoresModel(j).toIterable).toArray
        scoresProb.append(scoresProbTemp)
      }

      // Combine individual model scores to generate voting model score and rankings
      var combinedScoresProb = combineScores(scoresProb.toArray)
      val rerankedCandidates = rerank(combinedScoresProb, candidates)
      val (scoresBefore, scoresAfter) = beforeAndAfterScores(questions, candidates, rerankedCandidates)

      // Step 3D: Compute statistical significance of development performance
      val scoreSetBefore = new ArrayBuffer[Scores]
      val scoreSetAfter = new ArrayBuffer[Scores]
      for (i <- 0 until questions.size) {
        val question = questions(i)
        scoreSetBefore.append(scorer.computeScores(question, candidates(i).toList))
        scoreSetAfter.append(scorer.computeScores(question, rerankedCandidates(i).toList))
      }
      val scoreSetPValues = scorer.computeBootstrapResampling(scoreSetBefore.toArray, scoreSetAfter.toArray, 100000) // Bootstrap resampling, 10000 iterations


      // Step 3E: Output summary of performance for current threshold
      val scoreDelta = scoresAfter.sent.overlapMRR - scoresBefore.sent.overlapMRR // use Precision@1 (Sentence) as scoring metric
      reportDelta(scoresBefore, scoresAfter, tuningMetric, pw)

      // Step 3F: Compute histogram of performance at current threshold
      // Initialization
      var avgScoreSetBaseline = new Scores()
      var avgScoreSetExperimental = new Scores()
      val oneScorer = new Scorer()
      val histogramAt1 = new Histogram("Sentence Precision @1")
      val histogramMRR = new Histogram("Sentence MRR")

      // Histogram computation (compute across each question in the development set)
      for (i <- 0 until questions.size) {
        val question = questions(i)
        val scoreSetBaseline = oneScorer.computeScores(question, candidates(i).toList)
        val scoreSetExperimental = oneScorer.computeScores(question, rerankedCandidates(i).toList)

        // Averaging
        avgScoreSetBaseline.addToAverage(scoreSetBaseline)
        avgScoreSetExperimental.addToAverage(scoreSetExperimental)

        // Histogram computation and binning
        histogramAt1.addData(scoreSetExperimental.sent.overlapAt1 - scoreSetBaseline.sent.overlapAt1, i.toString)
        histogramMRR.addData(scoreSetExperimental.sent.overlapMRR - scoreSetBaseline.sent.overlapMRR, i.toString)
      }

      // Output detailed histogram performance report
      pw.println(" - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
      histogramAt1.display(pw)
      pw.println(" - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
      histogramMRR.display(pw)

      // Step 3G: Display average scores (and their statistical significance) across the test set for the baseline and experimental runs at this weight-clipping threshold.
      pw.println(" - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
      oneScorer.analysisTextSummary(pw, "Baseline", avgScoreSetBaseline)
      oneScorer.analysisTextSummaryCompare(pw, "Experimental", avgScoreSetExperimental, avgScoreSetBaseline)
      oneScorer.analysisTextSummaryPValues(pw, "P-Values (Experimental)", scoreSetPValues)

      pw.flush()

      // Step 3H: Track the best performing model and weight clipping threshold
      if (tuningMetric == "sent_p1") {
        if (avgScoreSetExperimental.sent.overlapAt1 >= topScore) {
          topScore = avgScoreSetExperimental.sent.overlapAt1
          topThresh = lambdaValue
          topScoreSetAfter = scoreSetAfter.toArray
        }
      } else if (tuningMetric == "sent_mrr") {
        if (avgScoreSetExperimental.sent.overlapMRR >= topScore) {
          topScore = avgScoreSetExperimental.sent.overlapMRR
          topThresh = lambdaValue
          topScoreSetAfter = scoreSetAfter.toArray
        }
      }
    }

    // Step 4: Once all lambda values have been evaluated, pick the top performing value
    pw.println("")
    pw.println(" ======================================================================================================================= ")
    pw.println("        Top performing lambda:")
    if (tuningMetric == "sent_p1") pw.println(" lambda: " + topThresh + " \t score(Paragraph P@1): " + topScore)
    if (tuningMetric == "sent_mrr") pw.println(" lambda: " + topThresh + " \t score(Paragraph MRR): " + topScore)
    pw.println(" Tuning procedure complete. ")
    pw.println(" ======================================================================================================================= ")

    lambda = topThresh

    // Save top array of scores, so we can come back to compute more statistics on them later
    val statsFilenameAfter: String = props.getProperty("output_report_filename") + ".stats.model_combined_lambda_tuned"
    scorer.saveScoresArray(topScoreSetAfter, statsFilenameAfter)

  }

  def doTuningProcedureV(
                          maxAnswersFromIR: Int,
                          questionsFilename: String,
                          detailedOutput: Boolean,
                          pw: PrintWriter) {

    // tuning is currently done only for feature based models
    // TODO: for kernel-based models, maybe we should tune C here?
    for (model <- models) if (model.usesKernels) return

    val votingMethod = props.getProperty("voting.method", "UNSPECIFIED")

    // Step 1: Randomize order of development questions
    var questions = randomizeQuestionOrder(questionParser.parse(questionsFilename).toArray)

    // If enabled, Limit development set size for debug purposes
    val questionDevLimit = StringUtils.getInt(props, "ranker.limit_dev_questions", 0)
    if ((questionDevLimit != 0) && (questionDevLimit < (questions.size - 1))) questions = questions.slice(0, questionDevLimit)

    // Step 2: Generate model features from development questions to use for tuning
    val queryAnnotations = mkAnnotations(questions)
    var candidates = Array.empty[Array[AnswerCandidate]]
    if (CQAEnabled) {
      candidates = mkCandidatesCQA(questions, queryAnnotations)
    } else {
      candidates = mkCandidates(queryAnnotations, maxAnswersFromIR)
    }


    // Step 3: Pre-generate model features
    val testDatasets = new ArrayBuffer[RankingDataset[String]]()
    // Model features distributed across different classifiers
    for (i <- 0 until models.size) {
      val model = models(i) // Select one model
      val maybeRescale = rescaleFeatures match {
          case true => Some(scaleRanges(i))
          case false => None
        }
      val testDataset = mkDataset(model, questions, queryAnnotations, candidates, maybeRescale) // Generate test dataset
      testDatasets.append(testDataset) // Generate test dataset
    }

    // Step 4: Output log file header
    pw.println(" ======================================================================================================================= ")
    pw.println(" ============================================== TUNING ================================================================= ")
    pw.println(" ======================================================================================================================= ")
    pw.println(" Tuning with {} questions", questions.size)
    pw.println("")
    logger.info("Tuning with {} questions.", questions.size)


    // Step 5: Call each tuning procedure as required

    // Step 5A: Clip SVM Weights
    // Not required?
    //tuneClippingWeights(models, questions, queryAnnotations, candidates, testDatasets.toArray, pw)

    // Step 5B: Tune lambda (weight) parameter between models
    if ((votingMethod == "combineprobs") && (models.size > 1)) {
      tuneLambda(questions, queryAnnotations, candidates, pw)
    }


    pw.println("End of doTuningProcedure()")
    displayMemoryUsage(pw)

  }


  def displayProperties(props: Properties, pw: PrintWriter) {
    pw.println("")
    pw.println("----------------------------------------------")
    pw.println("Experiment Properties: ")
    props.list(pw)

  }

  def createModel(model: String) {
    // method to handle model appending

    // model types that should match with Word2VecViewModel
    val word2vecMatches = "word2vec|bigrams|head_dep|head_rel_dep|ordered_head_dep|ordered_head_rel_dep".r
    // for combo vectors
    val w2vCombo = "w2v-baseline|head_dep-combo|head_rel_dep-combo".r

    // standardize case to minimize user error
    model.toLowerCase match {

      // translation model
      case "translationmultiset" => models.append(new TranslationMultisetModel(props))

      // handle unimplemented models...
      case _ => throw new RuntimeException(s"ERROR: $model not supported yet!")
    }
  }

  /*
   * Main entry point -- orchestrates a complete training, tuning, and test run of a model
   */
  override def doRun(workunitID: String) {
    val classifierProperties = new Properties()
    logger.info("Worker: Workunit (WID: " + workunitID + " ) started...")

    // Step 1: Load properties used internally
    val trainMethod = props.getProperty("ranker.train_method", "incremental").toLowerCase
    val answersFromIR = StringUtils.getInt(props, "ranker.answersfromIR", 20)

    // Step 1A: Append thread IDs to any filenames used internally by the RankingClassifier to make them thread safe
    val svm_c = props.getProperty("ranker.svm_c", "0.01") // extract C parameter from properties file
    classifierProperties.put("c", svm_c)
    classifierProperties.put("debugFile", "./debug-features-w" + workunitID)
    classifierProperties.setProperty("modelFile", "model" + workunitID + ".dat")
    classifierProperties.setProperty("trainFile", "train" + workunitID + ".dat")
    val classifierClass = props.getProperty("classifierClass", "")
    if (classifierClass != "") classifierProperties.setProperty("classifierClass", classifierClass)

    // Step 2: Perform model selection
    val modelSelection = props.getProperty("ranker.model", "cusp").split(",")
    // append relevant models
    for (model <- modelSelection) createModel(model)
    // if no models...
    if (models.size == 0) throw new RuntimeException("ERROR: RankerVoting: No recognized models were specified for ranker.model in the properties file. ")

    // Step 3: Load Training, Development, and Test questions
    logger.info("Reading questions...  ")
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

    logger.info("Reading questions complete.  ")

    // Step 4: Setup a verbosely named log file with the parameters for the experiment in the filename.
    var incrementalTrainingFile: String = generateVerboseFilename(props, questionsTrainSize.toString, questionsTestSize.toString, svm_c)
    props.setProperty("output_report_filename", incrementalTrainingFile)


    logger.info("Opening report file... (filename = " + incrementalTrainingFile + " )")
    val pw = new PrintWriter(incrementalTrainingFile)
    // print out the git revision to make it easier to replicate results
    VersionUtils.gitRevision.foreach(r => {
      pw.println(s"git revision $r")
    })
    displayProperties(props, pw)
    pw.println("------")
    pw.flush()

    pw.println("Pre-training")
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

    pw.println("Post-training, Pre-tuning")
    System.gc() // Manually perform garbage collection
    displayMemoryUsage(pw)

    // Step 6: Run tuning procedure
    if (questionsDevFilename != "") {
      doTuningProcedureV(answersFromIR, questionsDevFilename, true, pw) // tuning
      pw.flush()
    } else {
      logger.info("No development question set specified -- skipping tuning... ")
      pw.println(" NOTE: No development question set specified -- skipping over tuning procedure")
    }

    pw.println("Post-tuning, Pre-test")
    System.gc() // Manually perform garbage collection
    displayMemoryUsage(pw)

    // Step 7: Run test procedure
    logger.info("Running test procedure... ")
    pw.println(" ======================================================================================================================= ")
    pw.println("           Testing performance on test set... ")
    pw.println(" ======================================================================================================================= ")


    doTestProcedureV(answersFromIR, questionsTestFilename, true, pw) // test
    pw.flush()

    // Step 9: Cleanup
    System.gc() // Manually perform garbage collection
    displayMemoryUsage(pw)
    pw.close()
    println("Workunit complete... ")
    logger.info("Worker: Workunit (WID: " + workunitID + " ) completed...")

  }

}


object RankerVoting {
  val logger = LoggerFactory.getLogger(classOf[RankerVoting]) // Correct?



}


