package edu.arizona.sista.qa.word2vec

import java.io._
import java.util.Properties

import edu.arizona.sista.learning.ScaleRange
import edu.arizona.sista.processors.Document
import edu.arizona.sista.processors.fastnlp.FastNLPProcessor
import edu.arizona.sista.qa.baselines.{RelationLookupModel, RelationLookup}
import edu.arizona.sista.qa.index.TermFilter
import edu.arizona.sista.qa.ranking.{ProcessedQuestionSegments, ProcessedQuestion, RankingModel}
import edu.arizona.sista.qa.retrieval.AnswerCandidate
import edu.arizona.sista.qa.segmenter.QuestionProcessor
import edu.arizona.sista.qa.translation.TransView
import edu.arizona.sista.struct.Counter
import edu.arizona.sista.utils.StringUtils
import org.slf4j.LoggerFactory
import Word2VecRelationModel.logger

import scala.collection.mutable.ArrayBuffer

/**
  * Created by bsharp on 4/25/16.
  */
class Word2VecRelationModel (props:Properties) extends RankingModel {
  //val existingFrequencyFile = StringUtils.getStringOption(props, "w2v.filtering.existing_frequency_file")
  //val numWordsToUse = StringUtils.getIntOption(props, "w2v.filtering.num_words")
  //val minCount = StringUtils.getIntOption(props, "w2v.filtering.min_count")
  //val existingWordSet: Option[Set[String]] = existingFrequencyFile.map(filename => FrequencyFile.parseFrequencyFile(filename, numWordsToUse, minCount))

  // store information needed to load a w2v model
  case class VectorsAndInfo(vectorsC2E: Word2VecRelation,
                            vectorsE2C: Word2VecRelation,
                            view: String,
                            name: String,
                            vocabulary: Option[Set[String]] = None)

  val qProcessor = new QuestionProcessor(props)
  // Cause -> Effect
//  val targetVectorFilename = props.getProperty("word2vec_relation.target_vectors", "")
//  val contextVectorFilename = props.getProperty("word2vec_relation.context_vectors", "")
//  val model = new Word2VecRelation(targetVectorFilename, contextVectorFilename, None)
//  // Effect --> Cause
  //val targetVectorFilename2 = props.getProperty("word2vec_relation.e2c.target_vectors", "")
  //val contextVectorFilename2 = props.getProperty("word2vec_relation.e2c.context_vectors", "")
  //val model2 = new Word2VecRelation(targetVectorFilename2, contextVectorFilename2, None)
  val enableBidir = StringUtils.getBool(props, "word2vec_relation.enable_bidir", default = false)

  val defaultView = "words_content"
  val viewName = props.getProperty("view.view", defaultView)

  lazy val processor = new FastNLPProcessor()
  val termFilter = new TermFilter
  val useAsBackoff = StringUtils.getBool(props, "word2vec_relation.use_as_backoff", default = false)
  lazy val relationLookup = new RelationLookup(props)

  // maximum number of matrices to use (used to determine number of properties to look at)
  val nMatrices = StringUtils.getInt(props, "word2vec_relation.n_matrices", 20)

  // read the properties corresponding to a given index model, and if the model's enabled, return info for it
  def vectorsForIndexFromProperties(index: Int): Option[VectorsAndInfo] = {
    val propertyPrefix = s"word2vec_relation.model${index}."
    val filenameTarget = props.getProperty(propertyPrefix + "vectors_target")
    val filenameContext = props.getProperty(propertyPrefix + "vectors_context")
    val filenameE2CTarget = props.getProperty(propertyPrefix + "vectors_e2c_target")
    val filenameE2CContext = props.getProperty(propertyPrefix + "vectors_e2c_context")
    val enabled = StringUtils.getBool(props, propertyPrefix + "enable", false)
    val view = props.getProperty(propertyPrefix + "override_view", defaultView)
    var name = props.getProperty(propertyPrefix + "name")

    // frequency files are a list of word and frequency pairs, one per line, sorted by frequency descending.
    // if this property is passed, filter the vocabulary of the word2vec matrix to use only the top num_words words
    // from the frequency file
//    val existingFrequencyFile = StringUtils.getStringOption(props, propertyPrefix +"filtering.existing_frequency_file")
//    val numWordsToUse = StringUtils.getIntOption(props, propertyPrefix + "filtering.num_words")
//    val minCount = StringUtils.getIntOption(props, propertyPrefix + "filtering.min_count")
//    lazy val existingWordSet: Option[Set[String]] = existingFrequencyFile.map(filename => FrequencyFile.parseFrequencyFile(filename, numWordsToUse, minCount))

    // construct a unique-ish name if one wasn't specified
//    if (enabled && name == null) {
//      name = new File(filename).getName + "_" + view
//    }
    if (! enabled)
      None
    else {
      Some(VectorsAndInfo(
        vectorsC2E = new Word2VecRelation(filenameTarget, filenameContext, wordsToUse = None),
        vectorsE2C = new Word2VecRelation(filenameE2CTarget, filenameE2CContext, wordsToUse = None),
         view = view, name = name))
    }
  }

  // read the properties and get a list of info for all enabled models
  val enabledModels: Seq[VectorsAndInfo] = for {
    index <- (1 to nMatrices)
    info <- vectorsForIndexFromProperties(index)
  } yield info


  def mkFeatures( answer:AnswerCandidate,
                  question:ProcessedQuestion,
                  externalFeatures:Option[Counter[String]],
                  errorPw:PrintWriter = null): (Counter[String], String) = {

    question match {
      case q:ProcessedQuestionSegments => (mkFeaturesCuspDispatch (answer, q), null)
      case _ => throw new RuntimeException ("Word2VecModel.mkFeatures(): question passed is not of type ProcessedQuestionSegments")
    }

  }

  def mkFeaturesCuspDispatch (answer:AnswerCandidate,
                              question:ProcessedQuestionSegments): Counter[String] = {
    return mkFeaturesWord2VecRelation(answer, question)
  }

  def mkProcessedQuestion(question:Document):Array[ProcessedQuestion] = {
    // default to QSEG if method is unknown
    return Array[ProcessedQuestion](qProcessor.mkProcessedQuestionOneArgument(question))
  }

  def mkFeaturesWord2VecRelation( answer:AnswerCandidate,
                          question:ProcessedQuestionSegments): Counter[String] = {

    val features = new Counter[String]()

    // Add basic IR feature
//    if (props.getProperty("ranker.train_method", "incremental").toLowerCase != "hierarchical") {
//      features.setCount("ir", answer.answerScore)
//    }

    // Step 1A: If we failed to properly segment the question, then return an empty feature list.  Default back to other methods (e.g. IR)
    if (question.questionType == "failed") return features

    // Step 2: Locate words in question and answer
    val q = if (viewName.contains("dep") || viewName.contains("trans")) {
      docAnnotations(question.annotation)
    } else {
      question.annotation
    }

    val a = if (viewName.contains("dep") || viewName.contains("trans")) {
      docAnnotations(answer.annotation)
    } else {
      answer.annotation
    }

    val QView = new TransView(q)
    val AView = new TransView(a)

    QView.makeView(viewName)
    AView.makeView(viewName)

    val questionTokens = QView.features.toArray
    val answerTokens = AView.features.toArray

    def checkRange(dbl: Double) = {
      if (dbl.isNaN || dbl == Double.MinValue || dbl == Double.MaxValue) 0.0 else dbl
    }

    // map view name to (Question tokens, Answer tokens), so that we can lookup tokens by model's view
    val featuresByView: Map[String, (Array[String], Array[String])] = enabledModels.map(_.view).toSet.map({
      (viewName:String) => {
        val QView = new TransView(q)
        val AView = new TransView(a)

        QView.makeView(viewName)
        AView.makeView(viewName)
        val questionTokens = QView.features.toArray
        val answerTokens = AView.features.toArray
        logger.debug ("* mkFeaturesWord2VecRelation: Started...")
        logger.debug (s"*     [ questionTokens = ${questionTokens.mkString(" ")} ]")
        logger.debug (s"*     [ answerFeatures   = ${answerTokens.mkString(" ")} ]")

        viewName -> (questionTokens, answerTokens)
      }
    }).toMap

    for (model <- enabledModels) {
      // Determine whether the question contains a cause or an effect (i.e. is the question the target or
      // context language?)
      val qText = q.sentences.map(_.getSentenceText())
      //val (mq, ma, dq, da) = selectMatrix(qText, model.vectors, Some(pw))
      val (mq, ma, dq, da) = selectMatrix(qText, model.vectorsC2E, None)
      val (ma2, mq2, da2, dq2) = selectMatrix(qText, model.vectorsE2C, None)  // for the E2C --> note the opposite direction

      //val vectors = model.vectorsC2E
      val (questionFeatures, answerFeatures) = featuresByView(model.view)
      // Step 3: Determine word2vec scores
      if (enableBidir) {
        val scoreTextSimilarity = checkRange(Word2VecRelation.bidirTextSimilarity(questionFeatures, answerFeatures, mq, ma, dq, da,
          mq2, ma2, dq2, da2))
        features.setCount(s"WORD2VEC_REL_BIDIR_${model.name}_TEXTSIM", scoreTextSimilarity)
      } else {
        var scoreTextSimilarity = checkRange(Word2VecRelation.textSimilarity(questionFeatures, answerFeatures, mq, ma, dq, da))
        if (useAsBackoff) {
          val map = RelationLookupModel.selectMap(questionTokens, relationLookup)
          val scoreMaxMatches = relationLookup.maxMatches(questionTokens, answerTokens, map)
          if (scoreMaxMatches > 100) scoreTextSimilarity = 2.0
        }
        features.setCount(s"WORD2VEC_REL_${model.name}_TEXTSIM", scoreTextSimilarity)
      }

      if (enableBidir) {
        val scoreAvgSimilarity = checkRange(Word2VecRelation.bidirAvgSimilarity(questionFeatures, answerFeatures, mq, ma, mq2, ma2))
        features.setCount(s"WORD2VEC_REL_BIDIR_${model.name}_AVGSIM", scoreAvgSimilarity)
      } else {
        var scoreAvgSimilarity = checkRange(Word2VecRelation.avgSimilarity(questionFeatures, answerFeatures, mq, ma))
        if (useAsBackoff) {
          val map = RelationLookupModel.selectMap(questionTokens, relationLookup)
          val scoreMaxMatches = relationLookup.maxMatches(questionTokens, answerTokens, map)
          if (scoreMaxMatches > 100) scoreAvgSimilarity = 2.0
        }
        features.setCount(s"WORD2VEC_REL_${model.name}_AVGSIM", scoreAvgSimilarity)
      }

      if (enableBidir) {
        val scoreMaxSimilarity = checkRange(Word2VecRelation.bidirMaxSimilarity(questionFeatures, answerFeatures, mq, ma, mq2, ma2))
        features.setCount(s"WORD2VEC_REL_BIDIR_${model.name}_MAXSIM", scoreMaxSimilarity)
      } else {
        var scoreMaxSimilarity = checkRange(Word2VecRelation.maxSimilarity(questionFeatures, answerFeatures, mq, ma))
        if (useAsBackoff) {
          val map = RelationLookupModel.selectMap(questionTokens, relationLookup)
          val scoreMaxMatches = relationLookup.maxMatches(questionTokens, answerTokens, map)
          if (scoreMaxMatches > 100) scoreMaxSimilarity = 2.0
        }
        features.setCount(s"WORD2VEC_REL_${model.name}_MAXSIM", scoreMaxSimilarity)
      }

      if (enableBidir) {
        val scoreMinSimilarity = checkRange(Word2VecRelation.bidirMinSimilarity(questionFeatures, answerFeatures, mq, ma, mq2, ma2))
        features.setCount(s"WORD2VEC_REL_BIDIR_${model.name}_MINSIM", scoreMinSimilarity)
      } else {
        val scoreMinSimilarity = checkRange(Word2VecRelation.minSimilarity(questionFeatures, answerFeatures, mq, ma))
        features.setCount(s"WORD2VEC_REL_${model.name}_MINSIM", scoreMinSimilarity)
      }


      // NB: if you add any features here, make sure to add their names to featureNames as well
    }


//    // Step 3: Determine word2vec scores
//    var scoreTextSimilarity = checkRange(model.textSimilarity(questionTokens, answerTokens, mq, ma, dq, da))
//    if (useAsBackoff) {
//      val map = RelationLookupModel.selectMap(questionTokens, relationLookup)
//      val scoreMaxMatches = relationLookup.maxMatches(questionTokens, answerTokens, map)
//      if (scoreMaxMatches > 100) scoreTextSimilarity = 1.0
//    }
//    if (enableBidir) {
//      val e2cTextSim = checkRange(model2.textSimilarity(questionTokens, answerTokens, mq2, ma2, dq2, da2))
//      scoreTextSimilarity = (0.5 * scoreTextSimilarity) + (0.5 * e2cTextSim)
//    }
//    features.setCount("WORD2VEC_REL_TEXTSIM", scoreTextSimilarity)
//
//    var scoreAvgSimilarity = checkRange(model.avgSimilarity(questionTokens, answerTokens, mq, ma))
//    if (useAsBackoff) {
//      val map = RelationLookupModel.selectMap(questionTokens, relationLookup)
//      val scoreMaxMatches = relationLookup.maxMatches(questionTokens, answerTokens, map)
//      if (scoreMaxMatches > 100) scoreAvgSimilarity = 1.0
//    }
//    if (enableBidir) {
//      val e2cAvgSim = checkRange(model2.avgSimilarity(questionTokens, answerTokens, mq2, ma2))
//      scoreAvgSimilarity = (0.5 * scoreAvgSimilarity) + (0.5 * e2cAvgSim)
//    }
//
//    features.setCount("WORD2VEC_REL_AVGSIM", scoreAvgSimilarity)
//
//    var scoreMaxSimilarity = checkRange(model.maxSimilarity(questionTokens, answerTokens, mq, ma))
//    if (useAsBackoff) {
//      val map = RelationLookupModel.selectMap(questionTokens, relationLookup)
//      val scoreMaxMatches = relationLookup.maxMatches(questionTokens, answerTokens, map)
//      if (scoreMaxMatches > 100) scoreMaxSimilarity = 1.0
//    }
//    if (enableBidir) {
//      val e2cMaxSim = checkRange(model2.maxSimilarity(questionTokens, answerTokens, mq2, ma2))
//      scoreMaxSimilarity = (0.5 * scoreMaxSimilarity) + (0.5 * e2cMaxSim)
//    }
//    features.setCount("WORD2VEC_REL_MAXSIM", scoreMaxSimilarity)
//
//    var scoreMinSimilarity = checkRange(model.minSimilarity(questionTokens, answerTokens, mq, ma))
//    if (enableBidir) {
//      val e2cMinSim = checkRange(model2.minSimilarity(questionTokens, answerTokens, mq2, ma2))
//      scoreMinSimilarity = (0.5 * scoreMinSimilarity) + (0.5 * e2cMinSim)
//    }
//    features.setCount("WORD2VEC_REL_MINSIM", scoreMinSimilarity)

    // return list of features
    features
  }

  /*
    def mkWordsFromQuestion(question:ProcessedQuestionSegments):Array[String] = {
      val out = new ArrayBuffer[String]
      mkWordsFromAnnotation( question.segments(0).doc )
    }
  */

  def mkWordsFromAnnotation(annotation:Document):Array[String] = {
    val out = new ArrayBuffer[String]
    for (sent <- annotation.sentences) {
      for (word <- sent.words) {
        out += word
      }
    }
    out.toArray
  }
  // Helper method for annotation
  def docToString(doc:Document):String = {
    // Transform a Document
    val textBuffer = new ArrayBuffer[String]
    for (sentence <- doc.sentences) {
      var text = sentence.words.mkString(" ")
      //TODO: Remove final spaces
      textBuffer += text
    }
    textBuffer.mkString(" ")
  }
  // Helper method for annotation
  def docAnnotations(doc:Document):Document = {
    val text = docToString(doc)
    val newDoc = processor.annotate(text)
    newDoc
  }

  // Based on the text of the question, select whether the question text is from the target or source vocabulary
  // Returns the (questionMatrix, answerMatrix, questionMatDims, answerMatDims)
  def selectMatrix(qText:Array[String],
                   model:Word2VecRelation, pw:Option[PrintWriter]): (Map[String, Array[Double]], Map[String, Array[Double]], Int, Int) = {
    // debug printing
    val qHead = qText.head

    // Word-based heuristics
    if (pw.isDefined) pw.get.println ("Question: " + qText.head)
    for (cRegex <- Word2VecRelationModel.causeRegexes) {
      if (qHead.toLowerCase.matches(cRegex)) {
        if (pw.isDefined) {
          pw.get.println ("\tQuestion = EFFECT")
          pw.get.flush()
        }
        return (model.matrixContext, model.matrixTarget, model.contextDimensions, model.targetDimensions)
      }
    }
    for (rRegex <- Word2VecRelationModel.resultRegexes) {
      if (qHead.toLowerCase.matches(rRegex)) {
        if (pw.isDefined) {
          pw.get.println ("\tQuestion = CAUSE")
          pw.get.flush()
        }
        return (model.matrixTarget, model.matrixContext, model.targetDimensions, model.contextDimensions)
      }
    }

    println (s"**WARNING: Question didn't match causal patterns: ${qText.mkString(" ")}")
    println (s"** Question head: $qHead")
    println ("**Defaulting to: Question=Cause, Answer=Effect")
    if (pw.isDefined) {
      pw.get.println ("\tQuestion = CAUSE (because of default)")
      pw.get.flush()
    }

    (model.matrixTarget, model.matrixContext, model.targetDimensions, model.contextDimensions)
  }

  def generateModelErrorAnalysisForQuestion(answerCandidate: AnswerCandidate, qSegs: ProcessedQuestionSegments, pw:PrintWriter) = {
    val q = qSegs.annotation
    val a = answerCandidate.annotation

    val QView = new TransView(q)
    val AView = new TransView(a)

    QView.makeView(viewName)
    AView.makeView(viewName)

    val questionTokens = QView.features.toArray
    val answerTokens = AView.features.toArray

    pw.println("QuestionTokens: " + questionTokens.mkString(", "))
    pw.println("AnswerCandidateTokens: " + answerTokens.mkString(", "))

    def checkRange(dbl: Double) = {
      if (dbl.isNaN || dbl == Double.MinValue || dbl == Double.MaxValue) 0.0 else dbl
    }

    // map view name to (Question tokens, Answer tokens), so that we can lookup tokens by model's view
    val featuresByView: Map[String, (Array[String], Array[String])] = enabledModels.map(_.view).toSet.map({
      (viewName:String) => {
        val QView = new TransView(q)
        val AView = new TransView(a)

        QView.makeView(viewName)
        AView.makeView(viewName)
        val questionTokens = QView.features.toArray
        val answerTokens = AView.features.toArray
        logger.debug ("* mkFeaturesWord2VecRelation: Started...")
        logger.debug (s"*     [ questionTokens = ${questionTokens.mkString(" ")} ]")
        logger.debug (s"*     [ answerFeatures   = ${answerTokens.mkString(" ")} ]")

        viewName -> (questionTokens, answerTokens)
      }
    }).toMap

    for (model <- enabledModels) {
      // Determine whether the question contains a cause or an effect (i.e. is the question the target or
      // context language?)
      val qText = q.sentences.map(_.getSentenceText())
      val (mq, ma, dq, da) = selectMatrix(qText, model.vectorsC2E, Some(pw))
      val (ma2, mq2, da2, dq2) = selectMatrix(qText, model.vectorsE2C, None)  // for the E2C --> note the opposite direction

      val (questionFeatures, answerFeatures) = featuresByView(model.view)

      // Step 3: Determine word2vec scores
      if (enableBidir) {
        val scoreTextSimilarity = checkRange(Word2VecRelation.bidirTextSimilarity(questionFeatures, answerFeatures, mq, ma, dq, da,
          mq2, ma2, dq2, da2))
        pw.println ("TEXT SIMILARITY: " + scoreTextSimilarity)
        //features.setCount(s"WORD2VEC_REL_BIDIR_${model.name}_TEXTSIM", scoreTextSimilarity)
      } else {
        var scoreTextSimilarity = checkRange(Word2VecRelation.textSimilarity(questionFeatures, answerFeatures, mq, ma, dq, da))
        //features.setCount(s"WORD2VEC_REL_${model.name}_TEXTSIM", scoreTextSimilarity)
      }

      if (enableBidir) {
        val scoreAvgSimilarity = checkRange(Word2VecRelation.bidirAvgSimilarity(questionFeatures, answerFeatures, mq, ma, mq2, ma2))
        //features.setCount(s"WORD2VEC_REL_BIDIR_${model.name}_AVGSIM", scoreAvgSimilarity)
        pw.println ("AVERAGE SIMILARITY: " + scoreAvgSimilarity)
      } else {
        var scoreAvgSimilarity = checkRange(Word2VecRelation.avgSimilarity(questionFeatures, answerFeatures, mq, ma))
        //features.setCount(s"WORD2VEC_REL_${model.name}_AVGSIM", scoreAvgSimilarity)
      }

      if (enableBidir) {
        val scoreMaxSimilarity = checkRange(Word2VecRelation.bidirMaxSimilarity(questionFeatures, answerFeatures, mq, ma, mq2, ma2))
        pw.println ("MAX SIMILARITY: " + scoreMaxSimilarity)
        //features.setCount(s"WORD2VEC_REL_BIDIR_${model.name}_MAXSIM", scoreMaxSimilarity)
      } else {
        var scoreMaxSimilarity = checkRange(Word2VecRelation.maxSimilarity(questionFeatures, answerFeatures, mq, ma))
        //features.setCount(s"WORD2VEC_REL_${model.name}_MAXSIM", scoreMaxSimilarity)
      }

      if (enableBidir) {
        val scoreMinSimilarity = checkRange(Word2VecRelation.bidirMinSimilarity(questionFeatures, answerFeatures, mq, ma, mq2, ma2))
        pw.println ("MIN SIMILARITY: " + scoreMinSimilarity)
      } else {
        val scoreMinSimilarity = checkRange(Word2VecRelation.minSimilarity(questionFeatures, answerFeatures, mq, ma))

      }

      val srs = new ScaleRange[String]
      srs.mins = Counter.loadFrom[String](new FileReader(new File("/lhome/bsharp/causal/yahoo/EA/test_scaleRange.mins")))
      srs.maxs = Counter.loadFrom[String](new FileReader(new File("/lhome/bsharp/causal/yahoo/EA/test_scaleRange.maxs")))
      //val srs = ScaleRange.loadFrom[String](new BufferedReader(new File("/lhome/bsharp/causal/yahoo/EA/test_scaleRange.txt")))
      val allSims = Word2VecRelation.bidirSimComp(questionFeatures, answerFeatures, mq, ma, mq2, ma2)
      pw.println("--------------------------------------------------------------------------")
      pw.println ("\t\tSIMILARITY COMPARISON:")
//      val aToksFormatted = answerFeatures.map(af => "%1$-10s".format(af)).mkString("  ")
//      pw.println ("%1$-10s".format("Q words") + "  " + aToksFormatted)
      for (qTokenId <- 0 until questionFeatures.length) {
        val qTok = questionFeatures(qTokenId)
        val qTokFormatted = "%1$-10s".format(qTok)
        pw.println ("  QWord: " + qTok)
        val aWordComparisonInfo = allSims(qTokenId)._2
        //val alignedSims = allSims(qTokenId).map(sim => "%1$-10s".format("%3.5f".format(sim))).mkString("  ")
        for {
          ansInfoTuple <- aWordComparisonInfo
          //ansInfoTuple <- qRow
        } pw.println ("\t\t" + includeWeightInfo(rescale(srs, ansInfoTuple)))
        //pw.println("\t" + allSims(qTokenId))
      }
      pw.println("--------------------------------------------------------------------------")
    }
  }

  def includeWeightInfo(tup: (String, String, Double)): String = {
    val out = new StringBuilder
    val (featureName, word, score) = tup
    val weights = new Counter[String]
    weights.setCount("MAXSIM", 0.82839)
    weights.setCount("MINSIM", -0.44499)
    weights.setCount("AVGSIM", -2.17742)
    val f = featureName.split("_").last
    val w = weights.getCount(f)
    out.append(s"$f\t")
    out.append(s"score ($word): ${score.formatted("%1.3f")}  *  ")
    out.append(s"weight: ${w.formatted("%1.3f")} = ${(score*w).formatted("%1.3f")}")

    out.toString()
  }

  def rescale(ranges:ScaleRange[String], tup: (String, String, Double)): (String, String, Double) = {
    val (featureName, word, rawScore) = tup
    var min:Double = 0.0
    var max:Double = 0.0
    if(ranges.contains(featureName)) {
      min = ranges.min(featureName)
      max = ranges.max(featureName)
    }
    val scaledScore = scale(rawScore, min, max, 0.0, 1.0)

    (featureName, word, scaledScore)
  }

  def scale(value:Double, min:Double, max:Double, lower:Double, upper:Double):Double = {
    if(min == max) return upper

    // the result will be a value in [lower, upper]
    lower + (upper - lower) * (value - min) / (max - min)
  }

}


object Word2VecRelationModel {
  val logger = LoggerFactory.getLogger(classOf[Word2VecRelationModel])

  //val causePhrases = Array("What cause", "What is the cause", "What are the cause", "What effects the", "What affects the")

  // Used to detect Causal pattern, in each, X is the EFFECT
  // Example: What can cause X?
  val causeRegex1 = "^[Ww]hat ([a-z]+ ){0,3}cause.+"
  // Example: What could affect the X
  val causeRegex2 = "^[Ww]hat ([a-z]+ ){0,1}[ea]ffects? the .+"
  // Example: What might result in X?
  val causeRegex3 = "^[Wh]hat ([a-z]+ ){0,3}results? in .+"
  // Combine
  val causeRegexes = Array(causeRegex1, causeRegex2, causeRegex3)

  // Used to detect Causal pattern, in each, Y is the CAUSE
  // Example: What is the result of Y?
  val resultRegex1 = "^[Wh]hat ([a-z]+ ){0,3}results? of .+"
  // Example: What effect does Y have on plants?
  val resultRegex2 = "^[Wh]hat ([a-z]+ ){0,3}[ea]ffects? .+"
  // Combine
  val resultRegexes = Array(resultRegex1, resultRegex2)

}
