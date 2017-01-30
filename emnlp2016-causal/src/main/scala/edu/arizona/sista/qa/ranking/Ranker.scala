package edu.arizona.sista.qa.ranking

import java.util.concurrent.atomic.AtomicInteger

import edu.arizona.sista.processors.Document
import edu.arizona.sista.qa.retrieval.AnswerCandidate
import edu.arizona.sista.qa.word2vec.ErrorAnalysisCausal
import collection.mutable.{ListBuffer, ArrayBuffer}
import edu.arizona.sista.qa.scorer._
import edu.arizona.sista.struct.Counter
import edu.arizona.sista.learning._
import edu.arizona.sista.utils.StringUtils
import java.util.Properties
import edu.arizona.sista.qa.QA
import util.Random
import java.io.{File, FileReader, PrintWriter}
import org.slf4j.LoggerFactory
import scala.Tuple2
import edu.arizona.sista.qa.discourse.{DiscourseModelNGram}
import scala.collection.parallel.ForkJoinTaskSupport
import scala.collection.mutable
import edu.arizona.sista.qa.ranking.cache.{JSONFeatureCache, DBFeatureCache, FeatureCache}

/**
 * Created with IntelliJ IDEA.
 * User: peter
 * Date: 7/31/13
 */


abstract class Ranker(val props:Properties) {
  val logger = LoggerFactory.getLogger(classOf[Ranker])   // Correct?

  ///** QA system used during training to generate candidates */
  var qa:QA = new QA(props)
  /** Generic scorer used during training to assign answer scores */
  lazy val scorer:Scorer = new Scorer()
  /** Parser for gold files; used during training, tuning */
  lazy val questionParser = new QuestionParser

  val parallelize = StringUtils.getBool(props, "ranker.parallelize", false)
  val nThreads = StringUtils.getIntOption(props, "ranker.n_threads")

  val cacheFeatures = StringUtils.getBool(props, "ranker.cache_features", false)

  val readOnlyCache = StringUtils.getBool(props, "ranker.cache_read_only", false)

  val featureCache: Option[FeatureCache] = if (cacheFeatures) {
    val cacheFilename = props.getProperty("ranker.cache_filename")
    if (cacheFilename == null) sys.error("must pass a cache filename if cache_features is set to true")
    if (cacheFilename.endsWith("db"))
      Some(new DBFeatureCache(cacheFilename))
    else if (cacheFilename.endsWith("json"))
      Some(new JSONFeatureCache(cacheFilename))
    else sys.error("cache_filename must end with db (for SQLite databse) or json (for json file)")
  } else None

  val featureNamesByModel = new mutable.HashMap[RankingModel,
    Set[String]] with mutable.SynchronizedMap[RankingModel, Set[String]] {
    override def default(rankingModel: RankingModel) = rankingModel.featureNames
  }

  /** Constants */
  val VERY_SMALL_SCORE_DIFF = 0.0001

  //------------------------------------------------------------
  //       Main Functions (training, tuning, and testing)
  //------------------------------------------------------------

  /*
   * Training Function
   */
  def train(questionsFilename:String,
            model:RankingModel,
            maxAnswersFromIR:Int,
            classifierProperties:Properties,
            crossValidate:Boolean = true,
            pw:PrintWriter)

  /*
   * Testing and evaluation function
   */
  def test( testDataset:RankingDataset[String],
            classifier:RankingClassifier[String],
            generateProbabilities:Boolean = false,
            gamma:Double = 1.0):Array[Array[Double]] = {

    val scores = new Array[Array[Double]](testDataset.size)

    for(i <- 0 until testDataset.size) {
      val queryDatums = testDataset.mkQueryDatums(i)            // Generate model features from test question
      // Keep track of qids, in case we save them for offline testing
      if(classifier.isInstanceOf[SVMRankingClassifier[String]])
        classifier.asInstanceOf[SVMRankingClassifier[String]].setQid(i + 1)
      // Generate score for a given question
      if(generateProbabilities) {
        scores(i) = classifier.probabilitiesOf(queryDatums, gamma).toArray
      } else {
        scores(i) = classifier.scoresOf(queryDatums).toArray
      }
    }

    scores
  }


  /*
   * Test and evaluation procecedure
   */
  def doTestProcedure(
    model:RankingModel,
    maxAnswersFromIR:Int,
    questionsFilename:String,
    detailedOutput:Boolean,
    saveTestFile:Boolean,
    pw:PrintWriter)

  /*
   * Parameter tuning procedure
   */
  def doTuningProcedure(
    model:RankingModel,
    maxAnswersFromIR:Int,
    questionsFilename:String,
    detailedOutput:Boolean,
    pw:PrintWriter)

  /*
   * Main entry point -- orchestrates a complete training, tuning, and test run of a model
   */
  def doRun(workunitID:String) {
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
    classifierProperties.setProperty("testFile", "test" + workunitID + ".dat")
    classifierProperties.setProperty("keepIntermediateFiles", "true")

    // Step 2: Perform model selection
    val modelSelection = props.getProperty("ranker.model", "cusp")
    val model: RankingModel = new DiscourseModelNGram(props)
//    if (modelSelection == "cusp") {
//      model = new DiscourseModelCusp(props)
//    } else if (modelSelection == "ngram") {
//      model = new DiscourseModelNGram(props)
//    } else if (modelSelection == "tree") {
//      model = new DiscourseModelTree(props)
//    } else if (modelSelection == "treek") {
//      model = new DiscourseModelTreeKernel(props)
//    }

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

    logger.info ("Opening report file... (filename = " + incrementalTrainingFile + " )")
    val pw = new PrintWriter(incrementalTrainingFile)
    pw.println ("------")
    pw.flush()

    pw.println ("Pre-training")
    displayMemoryUsage(pw)

    // Step 5: Run Experiment
    logger.info ("Running training... ")
    train(questionsTrainFilename, model, answersFromIR, classifierProperties, false, pw)

    pw.println ("Post-training, Pre-tuning")
    System.gc()                   // Manually perform garbage collection
    displayMemoryUsage(pw)

    // Step 6: Run tuning procedure
    if (questionsDevFilename != "") {
      doTuningProcedure(model, answersFromIR, questionsDevFilename, true, pw)      // tuning
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

    doTestProcedure(model, answersFromIR, questionsTestFilename, true, true, pw)      // test
    pw.flush()

    pw.println ("Post-test, Pre-verification")
    System.gc()                   // Manually perform garbage collection
    displayMemoryUsage(pw)

    // Step 8: As a verification step, re-run test procedure on training data
    pw.println (" ======================================================================================================================= ")
    pw.println ("           Verifying performance on training set (below)... ")
    pw.println (" ======================================================================================================================= ")

    doTestProcedure(model, answersFromIR, questionsTrainFilename, false, false, pw)     // training set
    pw.flush()

    pw.println ("Post-verification")
    System.gc()                   // Manually perform garbage collection
    displayMemoryUsage(pw)


    // Step 9: Cleanup
    pw.close()
    println ("Workunit complete... ")
    logger.info("Worker: Workunit (WID: " + workunitID + " ) completed...")

  }



  //----------------------------
  //    Supporting Functions
  //----------------------------

  def mkCandidates(questions:Array[Document], maxAnswersFromIR:Int):Array[Array[AnswerCandidate]] = {
    val candidates = new ArrayBuffer[Array[AnswerCandidate]]
    for(question <- questions) {
      // fetch answer candidates from IR
      val answers = qa.answer(question, maxAnswersFromIR).toArray
      candidates += answers
    }
    assert(candidates.size == questions.size)
    candidates.toArray
  }

  def mkAnnotations(questions:Iterable[Question]):Array[Document] = {
    val annotations = new ArrayBuffer[Document]
    for(question <- questions) {
      val queryAnnotation = qa.queryProcessor.annotate(question.text)
      annotations += queryAnnotation
    }
    annotations.toArray
  }


  def addPerfectPredictorFeature(question:Question, answer:AnswerCandidate, features:Counter[String]):Counter[String] = {
    // A method to convince yourself that the SVM is working properly.
    var maxScore:Double = 0.0
    for (ga <- question.goldAnswers) {
      val score = scorer.sentOverlapF1(answer, ga)
      if (score > maxScore) maxScore = score
    }

    // Make perfect predictor feature
    if (maxScore > 0) {
      features.setCount("perfect_predictor", 1)
    }
    features
  }


  def mkDataset( model:RankingModel,
                 questions:Array[Question],
                 queryAnnotations:Array[Document],
                 candidates:Array[Array[AnswerCandidate]],
                 rangeToScaleBy: Option[ScaleRange[String]] = None):RankingDataset[String] = {

    assert(candidates.size == questions.size)
    assert(queryAnnotations.size == questions.size)

    val dataset = if(model.usesKernels) new RVFKRankingDataset[String] else new RVFRankingDataset[String]

    val questionsProcessed: AtomicInteger = new AtomicInteger(0)

    def mkDatums(question: Question, answers: Array[AnswerCandidate], processedQuestions: Array[ProcessedQuestion])  = {
      logger.debug ("-----------------------------------------------------------------------------------")
      logger.debug ("* Question[]: " + question.text)
      // assign scores to these answers based on gold answers
      val ranks = mkRanks(answers, question.goldAnswers)
      var offset = 0
      val queryDatums = new ListBuffer[Datum[Int, String]]

      // make datums and add to dataset
      for(j <- 0 until answers.size) {
        val answer = answers(j)
        logger.debug ("* AnswerCandidate[" + j + "]")

        // Add features from one or more processedQuestion representations
        var features = new Counter[String]
        var kernel:String = null
        for (processedQuestion <- processedQuestions) {
          //todo: change to Try.toOption wrapped block, flatmapped, in case model doesn't have featureNames implemented
          val (f, k) = featureCache.map(
            cache => (cache.getFeatures(processedQuestion.asInstanceOf[ProcessedQuestionSegments],
            answer,
            featureNamesByModel(model),
            { model.mkFeatures(answer, processedQuestion, None) }, // this should only be called if features not in cache
            readOnlyCache = readOnlyCache
            ))
          ).getOrElse(model.mkFeatures(answer, processedQuestion, None))

          // TODO: this supports only one processedQuestion per question!
          if(k != null && kernel == null) kernel = k
          features = features + f
        }
        println(s"summed features: ${features.toShortString}")
        println(s"ranks: ${ranks(offset)}")

        //## TEST: Add "perfect predictor" feature, for testing
        if ( StringUtils.getBool(props, "ranker.add_perfect_predictor", false) ) {
          features = addPerfectPredictorFeature(question, answer, features)
        }

        // possibly scale the features, if we passed in a ScaleRange
        rangeToScaleBy.foreach(scaleRange => features = Datasets.svmScaleDatum(features, scaleRange))

        if(model.usesKernels) {
          assert(kernel != null)
          val datum = new RVFKDatum[Int, String](ranks(offset), features, kernel)
          queryDatums += datum
        } else {
          val datum = new RVFDatum[Int, String](ranks(offset), features)
          queryDatums += datum
        }
        offset += 1
      }
      queryDatums
    }

    val processedQuestions = queryAnnotations.map(model.mkProcessedQuestion)

    val questionIterable = (questions, candidates, processedQuestions).zipped

    val queryData: Traversable[Iterable[Datum[Int, String]]] = if (parallelize) {
      val pc = questionIterable.par
      nThreads.foreach {
        n => pc.tasksupport = new ForkJoinTaskSupport(new scala.concurrent.forkjoin.ForkJoinPool(n))
      }
      pc.map(pair => mkDatums(pair._1, pair._2, pair._3)).seq
    } else {
      questionIterable.map(mkDatums)
    }
    for (datum <- queryData) dataset += datum

    featureCache.foreach(cache => {
      println(s"${cache.cacheHits} total cache hits")
      println(s"${cache.cacheMisses} total cache misses")
      if (! readOnlyCache) {
        println("writing cache")
        cache.writeCache
        println("write complete")
      }
    })

    dataset
  }


  /**
   * Re-rank the answers in "before" using the scores produced by the classifier
   * @param scores Classifier scores
   * @param before Answers before re-ranking
   * @return Answers after re-ranking
   */
  def rerank(scores:Array[Array[Double]],
             before:Array[Array[AnswerCandidate]]): Array[Array[AnswerCandidate]] = {
    val after = new Array[Array[AnswerCandidate]](before.size)

    for(queryOffset <- 0 until before.size) {
      val queryRanks = new ListBuffer[(Int, Double)]
      for(i <- 0 until scores(queryOffset).size) {
        queryRanks += new Tuple2(i, scores(queryOffset)(i))
      }
      val sortedRanks = queryRanks.toList.sortBy(0 - _._2).toArray

      val ranked = new Array[AnswerCandidate](before(queryOffset).size)
      //for(i <- 0 until sortedRanks.size) {
      for(i <- 0 until before(queryOffset).size) {
        // Need deep copy
        ranked(i) = before(queryOffset)(sortedRanks(i)._1).clone
        ranked(i).score = sortedRanks(i)._2
      }
      after(queryOffset) = ranked

    }

    after
  }

  def beforeAndAfterScores(questions:Array[Question],
                           before:Array[Array[AnswerCandidate]],
                           after:Array[Array[AnswerCandidate]]):(Scores, Scores) = {
    var scoreSetBefore = new Scores()
    var scoreSetAfter = new Scores()

    for (i <- 0 until questions.size) {
      val question = questions(i)
      scoreSetBefore = scoreSetBefore + scorer.computeScores(question, before(i).toList)
      scoreSetAfter = scoreSetAfter + scorer.computeScores(question, after(i).toList)
    }
    // Compute averages
    scoreSetBefore = scoreSetBefore / questions.size.toDouble
    scoreSetAfter = scoreSetAfter / questions.size.toDouble

    // Report summary statistics
    logger.info (" ** beforeAndAfterScores(): Summary BEFORE " +
      "(sentence: P@1:" + (scoreSetBefore.sent.overlapAt1*100).formatted("%3.2f") +
      "  MRR:" + (scoreSetBefore.sent.overlapMRR*100).formatted("%3.2f") +
      ")  (paragraph: P@1:" + (scoreSetBefore.para.overlapAt1*100).formatted("%3.2f") +
      "  MRR:" + (scoreSetBefore.para.overlapMRR*100).formatted("%3.2f") + ") " )

    logger.info (" ** beforeAndAfterScores(): Summary  AFTER " +
      "(sentence: P@1:" + (scoreSetAfter.sent.overlapAt1*100).formatted("%3.2f") +
      "  MRR:" + (scoreSetAfter.sent.overlapMRR*100).formatted("%3.2f") +
      ")  (paragraph: P@1:" + (scoreSetAfter.para.overlapAt1*100).formatted("%3.2f") +
      "  MRR:" + (scoreSetAfter.para.overlapMRR*100).formatted("%3.2f") +  ") " )

    (scoreSetBefore, scoreSetAfter)
  }


  def mkRanks(answers:Array[AnswerCandidate], goldAnswers:Array[GoldAnswer]):Array[Int] = {
    //mkRanksContinuous(answers, goldAnswers)
    //mkRanksTwoBins(answers, goldAnswers)
    //mkRanksArbitraryBins(answers, goldAnswers, List[Double](0, 0.01, 0.5, 0.75, 0.9, 1.0))
    mkRanksArbitraryBins(answers, goldAnswers, List[Double](0, 0.01, 0.5, 0.9, 1.0))
  }


  def mkRanksTwoBins(answers:Array[AnswerCandidate], goldAnswers:Array[GoldAnswer]):Array[Int] = {
    val scores = new ArrayBuffer[(Int, Double)]
    var offset = 0
    for(a <- answers) {
      scores += new Tuple2(offset, scorer.maxSentOverlapF1ForCandidate(a, goldAnswers))
      offset += 1
    }

    val sortedScores = scores.toList.sortBy(_._2)
    logger.debug("Sorted scores: " + sortedScores)

    var ranks = new ListBuffer[(Int, Int)]()
    for(score <- sortedScores) {
      // we use two bins, bin1: [0, 0.5) and bin2: [0.5, 1)
      if(score._2 < 0.5) {
        ranks += new Tuple2(score._1, 1)
      } else {
        ranks += new Tuple2(score._1, 2)
      }
    }
    logger.debug("Ranks: " + ranks)

    val result = new Array[Int](answers.size)
    for(r <- ranks) {
      assert(r._2 > 0)
      result(r._1) = r._2
    }
    result
  }


  def mkRanksArbitraryBins(answers:Array[AnswerCandidate], goldAnswers:Array[GoldAnswer], bins:List[Double]):Array[Int] = {
    // bins example: bins = List[Double](0, 0.01, 0.5, 0.75, 0.9, 1.0)
    val scores = new ArrayBuffer[(Int, Double)]
    var offset = 0
    for(a <- answers) {
      scores += new Tuple2(offset, scorer.maxSentOverlapF1ForCandidate(a, goldAnswers))
      offset += 1
    }

    val sortedScores = scores.toList.sortBy(_._2)
    logger.debug("Sorted scores: " + sortedScores)

    var prevScore = -1.0
    var crtRank = 1
    var ranks = new ListBuffer[(Int, Int)]()
    for(score <- sortedScores) {
      for (binNum <- 0 until bins.size) {
        if(bins(binNum) <= score._2) crtRank = binNum + 1
      }
      ranks += new Tuple2(score._1, crtRank)
      prevScore = score._2
    }
    logger.debug("Ranks: " + ranks)

    val result = new Array[Int](answers.size)
    for(r <- ranks) {
      assert(r._2 > 0)
      result(r._1) = r._2
    }
    result
  }


  def mkRanksContinuous(answers:Array[AnswerCandidate], goldAnswers:Array[GoldAnswer]):Array[Int] = {
    val scores = new ArrayBuffer[(Int, Double)]
    var offset = 0
    for(a <- answers) {
      scores += new Tuple2(offset, scorer.maxSentOverlapF1ForCandidate(a, goldAnswers))
      offset += 1
    }

    val sortedScores = scores.toList.sortBy(_._2)
    logger.debug("Sorted scores: " + sortedScores)

    var prevScore = -1.0
    var crtRank = 0
    var ranks = new ListBuffer[(Int, Int)]()
    for(score <- sortedScores) {
      if(score._2 > prevScore + VERY_SMALL_SCORE_DIFF) {
        crtRank += 1
      }
      ranks += new Tuple2(score._1, crtRank)
      prevScore = score._2
    }
    logger.debug("Ranks: " + ranks)

    val result = new Array[Int](answers.size)
    for(r <- ranks) {
      assert(r._2 > 0)
      result(r._1) = r._2
    }
    result
  }



  //------------------------
  //    Helper Functions
  //------------------------

  // Returns the index of the answer candidate in the array with the highest Paragraph Precision@1 score.
  // Returns -1 if there are no non-zero scores amoung the candidates.
  def findGoldACIndex(question:Question, candidates:Array[AnswerCandidate]):Int = {
    val scorer = new Scorer()
    var maxScore:Double = 0.0
    var maxIdx:Int = -1

    for (i <- 0 until candidates.size) {
      for (ga <- question.goldAnswers) {
        val paraAt1 = scorer.paraOverlapF1(candidates(i), ga)
        if (paraAt1 > maxScore) {
          maxScore = paraAt1
          maxIdx = i
        }
      }
    }

    maxIdx
  }

  /*
   * Create detailed error analysis output
   */
  def createErrorAnalysis(models:Array[RankingModel],
                          questions:Array[Question],
                          queryAnnotations:Array[Document],
                          reranked:Array[Array[AnswerCandidate]],
                          filenamePrefix:String) {

    val correctQuestions = new ArrayBuffer[String]
    val incorrectQuestions = new ArrayBuffer[String]


//    val matrixInstrumented = new TranslationMatrixInstrumented(0.0)         //## TEMP
//    matrixInstrumented.importMatrixTEMP(props)                              //## TEMP

    // Open output files
    val pwCorrect = new PrintWriter(filenamePrefix + ".correct.txt")
    val pwIncorrect = new PrintWriter(filenamePrefix + ".incorrect.txt")

    val srs = new ScaleRange[String]
    srs.mins = Counter.loadFrom[String](new FileReader(new File("/lhome/bsharp/causal/yahoo/EA/test_scaleRange.mins")))
    srs.maxs = Counter.loadFrom[String](new FileReader(new File("/lhome/bsharp/causal/yahoo/EA/test_scaleRange.maxs")))

    // Regenerate features for gold and top answers, for each question
    for (i <- 0 until questions.size) {
      val question = questions(i)
      val candidates = reranked(i)
      val topIdx = findGoldACIndex(question, candidates)

//      matrixInstrumented.instrumentQuestion(question, queryAnnotations(i), candidates, topIdx)        //## TEMP

      if (topIdx == -1) {
        // Case 1: A gold answer does not exist amoung the answer candidates.  It is not possible for the reranker to
        // answer the question correctly.

      } else if (topIdx == 0) {
        // Case 2: The gold answer is in the top position -- the question was answered correctly
        pwCorrect.println ("")
        pwCorrect.println (" ============================================================================= ")
        pwCorrect.println (" ============================================================================= ")
        pwCorrect.println (" Question[" + i + "]: ")
        pwCorrect.println (question.toString())
        pwCorrect.println (" ============================================================================= ")

        for (model <- models) {
          val processedQuestions = model.mkProcessedQuestion(queryAnnotations(i))

          for (processedQuestion <- processedQuestions) {
            pwCorrect.println ("\n TOP/GOLD ANSWER CANDIDATE: \n")
            val (fGold, k1) = model.mkFeatures(candidates(0), processedQuestion, None, pwCorrect)
            val rescaledGold = ErrorAnalysisCausal.rescale(srs, fGold)
            val fGoldString = ErrorAnalysisCausal.includeWeightInfo(rescaledGold)
            pwCorrect.println ("")
            pwCorrect.println ("TOP/GOLD (CAND 0): " + candidates(0).getText)
            pwCorrect.println ("TOP/GOLD (CAND 0) FEATURES: \n" + fGoldString)
            pwCorrect.println ("")

            pwCorrect.println ("\n Next 3 Candidates: \n")
            for (j <- 1 until 4) {
              val (fCurr, _) = model.mkFeatures(candidates(j), processedQuestion, None, pwCorrect)
              val rescaledCurr = ErrorAnalysisCausal.rescale(srs, fCurr)
              val fCurrString = ErrorAnalysisCausal.includeWeightInfo(rescaledCurr)
              pwCorrect.println ("")
              pwCorrect.println (s"CAND $j: " + candidates(j).getText)
              pwCorrect.println (s"CAND $j FEATURES: \n" + fCurrString)
              pwCorrect.println ("")
            }

          }

        }

        correctQuestions += i.toString

      } else {
        // Case 3: The gold answer exists within the answer candidates, but is not in the top position
        pwIncorrect.println ("")
        pwIncorrect.println (" ============================================================================= ")
        pwIncorrect.println (" ============================================================================= ")
        pwIncorrect.println (" Question[" + i + "]: ")
        pwIncorrect.println (question.toString())
        pwIncorrect.println (" ============================================================================= ")

        for (model <- models) {
          val processedQuestions = model.mkProcessedQuestion(queryAnnotations(i))

          for (processedQuestion <- processedQuestions) {
            pwIncorrect.println ("\n TOP (INCORRECT) ANSWER CANDIDATE: \n")
            val (fTop, k1) = model.mkFeatures(candidates(0), processedQuestion, None, pwIncorrect)
            val rescaledTop = ErrorAnalysisCausal.rescale(srs, fTop)
            val fTopString = ErrorAnalysisCausal.includeWeightInfo(rescaledTop)
            pwIncorrect.println ("")
            pwIncorrect.println ("TOP (INCORRECT) (CAND 0): " + candidates(0).getText)
            pwIncorrect.println ("TOP (INCORRECT) CAND 0 FEATURES: \n" + fTopString)
            pwIncorrect.println ("")
            pwIncorrect.println ("--------------------------------------------------------------------------")
            pwIncorrect.println (s"\n GOLD ANSWER CANDIDATE (CAND $topIdx): \n")
            val (fGold, k2) = model.mkFeatures(candidates(topIdx), processedQuestion, None, pwIncorrect)
            val rescaledGold = ErrorAnalysisCausal.rescale(srs, fGold)
            val fGoldString = ErrorAnalysisCausal.includeWeightInfo(rescaledGold)
            pwIncorrect.println ("")
            pwIncorrect.println (s"GOLD CAND (CAND $topIdx): " + candidates(topIdx).getText)
            pwIncorrect.println (s"GOLD (CAND $topIdx) FEATURES: \n" + fGoldString)
            pwIncorrect.println ("")
            pwIncorrect.println ("(Repeat of TOP (INCORRECT) FEATURES:) \n" + fTopString)

            pwIncorrect.println ("\n Next 3 Candidates: \n")
            for (j <- 1 until 4) {
              val (fCurr, _) = model.mkFeatures(candidates(j), processedQuestion, None, pwCorrect)
              val rescaledCurr = ErrorAnalysisCausal.rescale(srs, fCurr)
              val fCurrString = ErrorAnalysisCausal.includeWeightInfo(rescaledCurr)
              pwIncorrect.println ("")
              pwIncorrect.println (s"CAND $j: " + candidates(j).getText)
              pwIncorrect.println (s"CAND $j FEATURES: \n" + fCurrString)
              pwIncorrect.println ("")
            }


          }
        }

        incorrectQuestions += i.toString

      }

      // Flush output
      pwCorrect.flush()
      pwIncorrect.flush()

    }

    // Print list of questions contained in each file at the end
    pwCorrect.println ("\n\n\n=========================================")
    pwCorrect.println ("List of correct questions (" + correctQuestions.size + " total): ")
    for (idx <- correctQuestions) pwCorrect.print (idx + ", ")
    pwCorrect.println ("")

    pwIncorrect.println ("\n\n\n=========================================")
    pwIncorrect.println ("List of incorrect questions (" + incorrectQuestions.size + " total): ")
    for (idx <- incorrectQuestions) pwIncorrect.print (idx + ", ")
    pwIncorrect.println ("")


    // Close output files
    pwCorrect.close()
    pwIncorrect.close()


//    matrixInstrumented.generateSummary ("instrumentedsummary.txt")        //## TEST
//    matrixInstrumented.save("instrumentedYA100kdev2.5k")

  }


  /*
   * Randomizes question order
   */
  def randomizeQuestionOrder(questionsIn:Array[Question]):Array[Question] = {
    val randomizeEnabled = StringUtils.getBool(props, "ranker.randomize_question_order", false)
    if (!randomizeEnabled) return questionsIn

    val questionsRand = new ArrayBuffer[Question]   // (note: Shuffle needs data in the form of an ArrayBuffer)
    questionsRand ++= questionsIn
    Random.shuffle(questionsRand).toArray           // Randomly shuffle the question presentation order.
  }

  /*
   * Generates a long, verbose filename detailing the simulation paramaters
   */
  def generateVerboseFilename(props:Properties, questionsTrainSize:String, questionsTestSize:String, svm_c:String):String = {
    var incrementalTrainingFile: String = props.getProperty("ranker.incremental_training_report", "/dev/null")

    val modelSelection = props.getProperty("ranker.model", "cusp")
    val answersFromIR = StringUtils.getInt(props, "ranker.answersfromIR", 20)
    val sentRangeStart = StringUtils.getInt(props, "discourse.sentrange_start", 0)
    val sentRangeEnd = StringUtils.getInt(props, "discourse.sentrange_end", 3)
    val sentRangeDesc:String = "-sr" + sentRangeStart + "to" + sentRangeEnd
    var classifierType:String = "SVM"   // Default in RankingClassifier constructor is SVM
    if (props.getProperty("classifierClass", "") == "JForestsRankingClassifier") classifierType = "BDT"
    if (props.getProperty("classifierClass", "") == "SVMRankingClassifier") classifierType = "SVM"
    if (props.getProperty("classifierClass", "") == "SVMKRankingClassifier") classifierType = "SVMK"
    var w2v_enabled:Boolean = StringUtils.getBool(props, "ranker.enable_word2vec_feature", false)
    val minScoreThreshW2V:Double = StringUtils.getDouble(props, "discourse.match_threshold_w2v", 0.75)

    val minScoreThreshCusp = props.getProperty("discourse.match_threshold_cusp")
    val minScoreThreshCuspW2V = props.getProperty("discourse.match_threshold_w2v_cusp")
    val minScoreThreshTree = props.getProperty("discourse.match_threshold_cusp")
    val minScoreThreshTreeW2V = props.getProperty("discourse.match_threshold_w2v_cusp")


    if (modelSelection == "ngram") {
      incrementalTrainingFile = incrementalTrainingFile +
        "-model_" + modelSelection +
        "-tr_" + props.getProperty("ranker.train_method", "UNDEFINED") +
//        "-disclist_" + props.getProperty("discourse.connectives_list_size", "UNDEFINED") +
        "-qseg_" + props.getProperty("discourse.question_processor", "UNDEFINED") +
        "-disc_thresh_cusp" + minScoreThreshCusp +
        "_w" + minScoreThreshCuspW2V +
        "_tree" + minScoreThreshTree +
        "_w" + minScoreThreshTreeW2V +
//        "-bvf_" + StringUtils.getBool(props, "discourse.binary_score", false) +
//        "-srm_" + StringUtils.getBool(props, "discourse.add_sentence_range_marker", true) +
        sentRangeDesc +
        "-intra_" + StringUtils.getBool(props, "discourse.intrasentence", false) +
//        "-deltaIR_" + StringUtils.getBool(props, "ranker.add_delta_ir_score", false) +
//        "-ngs" + props.getProperty("discourse.ngram_start_size") + "to" + props.getProperty("discourse.ngram_end_size") +
        "-candIR_" + answersFromIR +
//        "-metric_" + props.getProperty("ranker.scoring_metric", "UNDEFINED") +
//        "-tuningmetric_" + props.getProperty("ranker.tuning_metric", "para_p1") +
//        "-voting_" + props.getProperty("voting.method", "unspecified") +
        "-c_" + svm_c +
        "-cl_" + classifierType +
        "-qTrain" + questionsTrainSize.toString +
        "-qTest" + questionsTestSize.toString +
        ".txt"
    } else {
      incrementalTrainingFile = incrementalTrainingFile +
        "-model_" + modelSelection +
        "-tr_" + props.getProperty("ranker.train_method", "UNDEFINED") +
//        "-disclist_" + props.getProperty("discourse.connectives_list_size", "UNDEFINED") +
        "-qseg_" + props.getProperty("discourse.question_processor", "UNDEFINED") +
        "-w2v_" + w2v_enabled +
        "-disc_thresh_cusp" + minScoreThreshCusp +
        "_w" + minScoreThreshCuspW2V +
        "_tree" + minScoreThreshTree +
        "_w" + minScoreThreshTreeW2V +
//        "-bvf_" + StringUtils.getBool(props, "discourse.binary_score", false) +
//        "-srm_" + StringUtils.getBool(props, "discourse.add_sentence_range_marker", true) +
        sentRangeDesc +
        "-intra_" + StringUtils.getBool(props, "discourse.intrasentence", false) +
//        "-deltaIR_" + StringUtils.getBool(props, "ranker.add_delta_ir_score", false) +
//        "-candIR_" + answersFromIR +
//        "-metric_" + props.getProperty("ranker.scoring_metric", "UNDEFINED") +
//        "-tuningmetric_" + props.getProperty("ranker.tuning_metric", "para_p1") +
//        "-voting_" + props.getProperty("voting.method", "unspecified") +
        "-c_" + svm_c +
        "-cl_" + classifierType +
        "-qTrain" + questionsTrainSize.toString +
        "-qTest" + questionsTestSize.toString +
        ".txt"
    }

    return incrementalTrainingFile      // return verbose filename

  }


  //## MOVE TO ITS OWN UTILITY CLASS?
  def displayMemoryUsage(pw:PrintWriter) {
    val runtime = Runtime.getRuntime
    val mb = 1024 * 1024
    val total = runtime.totalMemory() / mb
    val free = runtime.freeMemory() / mb
    val max = runtime.maxMemory() / mb
    val used = total - free

    // Display memory usage
    logger.info(" #### HEAP MEMORY UTALIZATION STATISTICS ### ")
    logger.info(" #### Used: " + used + "MB")
    logger.info(" #### Free: " + free + "MB")
    logger.info(" #### Total: " + total + "MB")
    logger.info(" #### Max: " + max + "MB")

    pw.println(" #### HEAP MEMORY UTALIZATION STATISTICS ### ")
    pw.println(" #### Used: " + used + "MB")
    pw.println(" #### Free: " + free + "MB")
    pw.println(" #### Total: " + total + "MB")
    pw.println(" #### Max: " + max + "MB")
    pw.flush()

  }




}




/*
trait RankingModel {
  def mkFeatures(
                  answer:AnswerCandidate,
                  question:ProcessedQuestion,
                  externalFeatures:Option[Counter[String]]): Counter[String]

  def mkProcessedQuestion(question:Document):Array[ProcessedQuestion]

}

*/
