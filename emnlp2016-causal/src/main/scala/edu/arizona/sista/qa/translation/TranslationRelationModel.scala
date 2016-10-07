package edu.arizona.sista.qa.translation

import java.io.PrintWriter
import java.util.Properties
import edu.arizona.sista.processors.Document
import edu.arizona.sista.processors.fastnlp.FastNLPProcessor
import edu.arizona.sista.qa.index.TermFilter
import edu.arizona.sista.qa.ranking.{ProcessedQuestionSegments, ProcessedQuestion, RankingModel}
import edu.arizona.sista.qa.retrieval.AnswerCandidate
import edu.arizona.sista.qa.segmenter.QuestionProcessor
import edu.arizona.sista.qa.word2vec.Word2VecRelationModel
import edu.arizona.sista.struct.Counter
import edu.arizona.sista.utils.StringUtils
import org.slf4j.LoggerFactory
import TranslationRelationModel.logger
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

/**
  * Created by bsharp on 5/6/16.
  */
class TranslationRelationModel(props:Properties) extends RankingModel {
  case class MatrixAndInfo(filenamePrefixS2D: String, view: String, name: String,
                           float: Boolean = false, selfProb:Double,
                           filenamePrefixD2S: String) {
    // lazy load in case features are cached
    lazy val matrixS2D = TranslationMatrix.smartLoadDouble(filenamePrefixS2D, selfProb)
    lazy val matrixD2S = TranslationMatrix.smartLoadDouble(filenamePrefixD2S, selfProb)
  }

  val termFilter = new TermFilter()
  val indexDir = props.getProperty("index")
  val qProcessor = new QuestionProcessor(props)
  lazy val processor = new FastNLPProcessor()

  // Smoothing parameter
  val lambda = StringUtils.getDouble(props, "translation.relation.lambda", 0.25)

  val nMatrices = StringUtils.getInt(props, "translation.relation.n_matrices", 20)

  // extract view type
  val defaultView = props.getProperty("view.view", "lemmas_content")

  //val selfProb = StringUtils.getDouble(props, "translation.self", 0.50)

  val enableDistanceFeatures = StringUtils.getBool(props, "translation.relation.enable_distance_features", true)
  val enableProbabilityFeatures = StringUtils.getBool(props, "translation.relation.enable_probability_features", true)

  val disableProb = StringUtils.getBool(props, "translation.relation.disable_probability", false)
  val disableComposite = StringUtils.getBool(props, "translation.relation.disable_composite", false)
  val disableMin = StringUtils.getBool(props, "translation.relation.disable_min", false)
  val disableMax = StringUtils.getBool(props, "translation.relation.disable_max", false)
  val disableAvg = StringUtils.getBool(props, "translation.relation.disable_avg", false)

  val sanitizeForW2V = StringUtils.getBool(props, "translation.relation.w2v_sanitization", true)

  // read the matrix specification for the indexed properties, if it's enabled return Some(matrix info)
  def matrixForIndexFromProperties(index: Int): Option[MatrixAndInfo] = {
    val propertyPrefix = s"translation.relation.model${index}."
    val filenamePrefixS2D = props.getProperty(propertyPrefix + "matrix_prefix_s2d")
    val filenamePrefixD2S = props.getProperty(propertyPrefix + "matrix_prefix_d2s")
    var name =  props.getProperty(propertyPrefix + "name" )
    val enabled = StringUtils.getBool(props, propertyPrefix + "enable", false)
    val view = props.getProperty(propertyPrefix + "override_view", defaultView)

    val float = StringUtils.getBool(props, propertyPrefix + "float", false)
    val selfProb = StringUtils.getDouble(props, propertyPrefix + "self_prob", 0.5)

    if (! enabled)
      None
    else {
      Some(MatrixAndInfo(filenamePrefixS2D = filenamePrefixS2D,
        view = view,
        name = name,
        float = float,
        selfProb = selfProb,
        filenamePrefixD2S = filenamePrefixD2S))
    }
  }

  val enabledModels = for {
    index <- (1 to nMatrices)
    info <- matrixForIndexFromProperties(index)
  } yield info

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
    val newDoc = processor.mkDocument(text)
    processor.tagPartsOfSpeech(newDoc)
    processor.lemmatize(newDoc)
    processor.parse(newDoc)
    newDoc
  }

  def mkFeatures( answer:AnswerCandidate,
                  question:ProcessedQuestion,
                  externalFeatures:Option[Counter[String]],
                  errorPw:PrintWriter = null): (Counter[String], String) = {

    question match {
      case q:ProcessedQuestionSegments => (mkFeaturesTranslation(answer, q, errorPw), null)
      case _ => throw new RuntimeException ("TranslationModelAdditive.mkFeatures(): question passed is not of type ProcessedQuestionSegments")
    }
  }

  def mkProcessedQuestion(question:Document):Array[ProcessedQuestion] = {
    // default to QSEG if method is unknown
    Array[ProcessedQuestion](qProcessor.mkProcessedQuestionOneArgument(question))
  }

  def mkFeaturesTranslation( answer:AnswerCandidate,
                             question:ProcessedQuestionSegments,
                             errorPw:PrintWriter): Counter[String] = {

    var errorOut:PrintWriter = errorPw
    //if (errorPw == null) errorOut = new PrintWriter(System.out)         // Uncomment to direct error output to stdout if no output is specified

    val features = new Counter[String]()

    // Step 1: IR Feature must be included
    if (props.getProperty("ranker.train_method", "incremental").toLowerCase != "hierarchical") {
      features.setCount("ir", answer.score)
    }

    // Step 1A: If we failed to properly segment the question, then return an empty feature list.  Default back to other methods (e.g. IR)
    if (question.questionType == "failed") return features

    // Step 1B: Transform Q and A into partially annotated Documents
    val q = question.segments(0).doc
    val a = answer.doc.annotation

    // map view name to (Question features, Answer features)
    val featuresByView: Map[String, (Array[String], Array[String])] = enabledModels.map(_.view).toSet.map({
      (viewName:String) => {
        val QView = new TransView(q, sanitizeForW2V = sanitizeForW2V)
        val AView = new TransView(a, sanitizeForW2V = sanitizeForW2V)

        QView.makeView(viewName)
        AView.makeView(viewName)
        val questionFeatures = QView.features.toArray
        val answerFeatures = AView.features.toArray
        logger.debug ("* mkFeaturesTranslation: Started...")
        logger.debug (s"*     [ questionFeatures = ${questionFeatures.mkString(" ")} ]")
        logger.debug (s"*     [ answerFeatures   = ${answerFeatures.mkString(" ")} ]")
        if (errorOut != null) {
          errorOut.println ("* mkFeaturesTranslation: Started...")
          errorOut.println (s"*     [ questionFeatures = ${questionFeatures.mkString(" ")} ]")
          errorOut.println (s"*     [ answerFeatures   = ${answerFeatures.mkString(" ")} ]")
        }
        viewName -> (questionFeatures, answerFeatures)
      }
    }).toMap

    for (model <- enabledModels) {
      // Since the relation is directed, select the correct matrix to use given the question text
      val matrix = selectMatrix(q.sentences.head.getSentenceText, model)
      val (questionFeatures, answerFeatures) = featuresByView(model.view)
      if (errorOut != null) errorOut.println (s"\n **********\n ${model.name}: ")
      // Determine translation score

      // Translation: 1-hop
      if (enableProbabilityFeatures) {
        if (! disableProb)
          features.setCount(s"TRANS_RELATION_${model.name}_probability", matrix.prob(questionFeatures, answerFeatures, lambda, errorOut))
      }
//      if (enableDistanceFeatures) {
//        val distances = matrix.distances(questionFeatures, answerFeatures).toSeq //cache
//        if (! disableAvg)
//          features.setCount(s"TRANS_RELATION_${model.name}_avgDistance", matrix.avgDistance(distances).getOrElse(0.0))
//        if (! disableMax)
//          features.setCount(s"TRANS_RELATION_${model.name}_maxDistance", matrix.maxDistance(distances).getOrElse(0.0))
//        if (! disableMin)
//          features.setCount(s"TRANS_RELATION_${model.name}_minDistance", matrix.minDistance(distances).getOrElse(0.0))
//        if (! disableComposite)
//          features.setCount(s"TRANS_RELATION_${model.name}_txtDistance", matrix.textDistance(questionFeatures, answerFeatures).getOrElse(0.0))
//
//        // NB: if you add any features here, add their names to featureNames as well
//      }
    }

    // Store translation feature score
    logger.debug ("* Translation feature scores: " + counterToString(features) )

    if (math.random < 0.0025) {
      println("gc starting")
      println(System.currentTimeMillis)
      System.gc
      println("gc done")
      println(System.currentTimeMillis)
    }

    // return list of features
    features
  }

  // used to determine which features are produced by this model for caching purposes. Must update this when new
  // features are added
  override lazy val featureNames = (for {
    model <- enabledModels
    names = (
       if (disableProb) List[String]() else List(s"TRANS_RELATION_${model.name}_probability")) ++
        (if (disableAvg) List[String]() else List(s"TRANS_RELATION_${model.name}_avgDistance")) ++
        (if (disableMax) List[String]() else List(s"TRANS_RELATION_${model.name}_maxDistance")) ++
        (if (disableMin) List[String]() else List(s"TRANS_RELATION_${model.name}_minDistance")) ++
        (if (disableComposite) List[String]() else List(s"TRANS_RELATION_${model.name}_txtDistance")
      )
  } yield names).flatten.toSet ++ Set("ir")

  def toLowerCaseArray(in:Array[String]):Array[String] = {
    val out = new Array[String](in.size)
    for (i <- 0 until in.size) {
      out(i) = in(i).toLowerCase
    }
    out
  }

  // TODO: move this functionality in the Counter class
  def counterToString(in:Counter[String]):String = {
    val os = new mutable.StringBuilder()
    os ++= "("
    for (key <- in.keySet) {
      val value = in.getCount(key)
      os ++= s"[$key = $value] "
    }
    os ++= ")"
    os.toString()
  }

  // Based on the text of the question, select whether the question text is from the target or source vocabulary
  // Returns the (questionMatrix, answerMatrix)
  def selectMatrix(qHead:String, model:MatrixAndInfo): TranslationMatrix[Double] = {


    // Word-based heuristics

    // See if the question text is the EFFECT
    for (cRegex <- Word2VecRelationModel.causeRegexes) {
      if (qHead.toLowerCase.matches(cRegex)) {
        return model.matrixS2D
      }
    }
    // See if the question text is the CAUSE
    for (rRegex <- Word2VecRelationModel.resultRegexes) {
      if (qHead.toLowerCase.matches(rRegex)) {
        return model.matrixD2S
      }
    }

    println (s"**WARNING: Question didn't match causal patterns: ${qHead.mkString(" ")}")
    println ("**Defaulting to: Question=Cause, Answer=Effect")

    model.matrixD2S
  }
}

object TranslationRelationModel {
  val logger = LoggerFactory.getLogger(classOf[TranslationRelationModel])
}
