package edu.arizona.sista.qa.word2vec

import edu.arizona.sista.utils.{StringUtils, FrequencyFile}
import java.util.Properties
import edu.arizona.sista.qa.ranking.{ProcessedQuestion, RankingModel, ProcessedQuestionSegments}
import edu.arizona.sista.qa.retrieval.AnswerCandidate
import edu.arizona.sista.struct.Counter
import org.slf4j.LoggerFactory
import edu.arizona.sista.qa.index.TermFilter
import edu.arizona.sista.qa.segmenter.QuestionProcessor
import edu.arizona.sista.processors.fastnlp.FastNLPProcessor
import edu.arizona.sista.qa.translation.TransView
import java.io.PrintWriter
import edu.arizona.sista.processors.Document
import scala.collection.mutable.ArrayBuffer
import java.io._
import scala.collection.immutable.IndexedSeq

/**
 * Created by dfried on 6/3/14.
 */
/**
 * Use one or more word2vec matrices to generate features for QA pairs
 * @param props
 */
class Word2VecMultisetModel (props:Properties) extends RankingModel {
  import Word2VecMultisetModel.logger

  // store information needed to load a w2v model
  case class VectorsAndInfo(vectors: Word2vec,
                            view: String,
                            name: String,
                            vocabulary: Option[Set[String]] = None)

  val termFilter = new TermFilter()
  val indexDir = props.getProperty("index")
  val qProcessor = new QuestionProcessor(props)
  lazy val processor = new FastNLPProcessor()

  // maximum number of matrices to use (used to determine number of properties to look at)
  val nMatrices = StringUtils.getInt(props, "word2vec.n_matrices", 20)

  // extract view type
  val defaultView = props.getProperty("view.view", "words")

  val disableComposite = StringUtils.getBool(props, "word2vec.disable_composite", false)
  val disableMin = StringUtils.getBool(props, "word2vec.disable_min", false)
  val disableMax = StringUtils.getBool(props, "word2vec.disable_max", false)
  val disableAvg = StringUtils.getBool(props, "word2vec.disable_avg", false)

  // read the properties corresponding to a given index model, and if the model's enabled, return info for it
  def vectorsForIndexFromProperties(index: Int): Option[VectorsAndInfo] = {
    val propertyPrefix = s"word2vec.model${index}."
    val filename = props.getProperty(propertyPrefix + "vectors")
    val enabled = StringUtils.getBool(props, propertyPrefix + "enable", false)
    val view = props.getProperty(propertyPrefix + "override_view", defaultView)
    var name = props.getProperty(propertyPrefix + "name")

    // frequency files are a list of word and frequency pairs, one per line, sorted by frequency descending.
    // if this property is passed, filter the vocabulary of the word2vec matrix to use only the top num_words words
    // from the frequency file
    val existingFrequencyFile = StringUtils.getStringOption(props, propertyPrefix +"filtering.existing_frequency_file")
    val numWordsToUse = StringUtils.getIntOption(props, propertyPrefix + "filtering.num_words")
    val minCount = StringUtils.getIntOption(props, propertyPrefix + "filtering.min_count")
    lazy val existingWordSet: Option[Set[String]] = existingFrequencyFile.map(filename => FrequencyFile.parseFrequencyFile(filename, numWordsToUse, minCount))

    // construct a unique-ish name if one wasn't specified
    if (enabled && name == null) {
      name = new File(filename).getName + "_" + view
    }
    if (! enabled)
      None
    else {
      Some(VectorsAndInfo(vectors = new Word2vec(filename, wordsToUse = existingWordSet), view = view, name = name))
    }
  }

  // read the properties and get a list of info for all enabled models
  val enabledModels: Seq[VectorsAndInfo] = for {
    index <- (1 to nMatrices)
    info <- vectorsForIndexFromProperties(index)
  } yield info

  /**
   * @return A tuple containing explicit features and a kernelized version of the datum, if any (null, if non existing)
   */
  def mkFeatures( answer:AnswerCandidate,
                  question:ProcessedQuestion,
                  externalFeatures:Option[Counter[String]],
                  errorPw:PrintWriter = null): (Counter[String], String) = {

    question match {
      case q:ProcessedQuestionSegments => (mkFeaturesWord2Vec(answer, q, errorPw), null)
      case _ => throw new RuntimeException ("Word2VecMultisetModel.mkFeatures(): question passed is not of type ProcessedQuestionSegments")
    }
  }

  def mkProcessedQuestion(question:Document):Array[ProcessedQuestion] = {
    // default to QSEG if method is unknown
    Array[ProcessedQuestion](qProcessor.mkProcessedQuestionOneArgument(question))
  }

  def mkFeaturesWord2Vec(answer:AnswerCandidate,
                         question:ProcessedQuestionSegments,
                         errorOut: PrintWriter): Counter[String] = {
    val features = new Counter[String]()

    // Step 1: IR Feature must be included
    if (props.getProperty("ranker.train_method", "incremental").toLowerCase != "hierarchical") {
      features.setCount("ir", answer.score)
    }

    // Step 1A: If we failed to properly segment the question, then return an empty feature list.  Default back to other methods (e.g. IR)
    if (question.questionType == "failed") return features

    // Step 1B: Transform Q and A into partially annotated Documents
    // re-annotate the question
    val q = question.segments(0).doc//docAnnotations(question.segments(0).doc)
    val a = answer.doc.annotation//docAnnotations(answer.doc.annotation)
    // Step 2: Locate words in question and answer


    // map view name to (Question tokens, Answer tokens), so that we can lookup tokens by model's view
    val featuresByView: Map[String, (Array[String], Array[String])] = enabledModels.map(_.view).toSet.map({
      (viewName:String) => {
        val QView = new TransView(q)
        val AView = new TransView(a)

        QView.makeView(viewName)
        AView.makeView(viewName)
        val questionTokens = QView.features.toArray
        val answerTokens = AView.features.toArray
        logger.debug ("* mkFeaturesWord2Vec: Started...")
        logger.debug (s"*     [ questionTokens = ${questionTokens.mkString(" ")} ]")
        logger.debug (s"*     [ answerFeatures   = ${answerTokens.mkString(" ")} ]")
        if (errorOut != null) {
          errorOut.println ("* mkFeaturesWord2Vec: Started...")
          errorOut.println (s"*     [ questionTokens = ${questionTokens.mkString(" ")} ]")
          errorOut.println (s"*     [ answerFeatures   = ${answerTokens.mkString(" ")} ]")
        }
        viewName -> (questionTokens, answerTokens)
      }
    }).toMap

    // define a wrapper function to replace numerical errors with 0
    // the numerical errors we are checking for here
    // should only arise if either the
    // question or the answer is length 0
    def checkRange(dbl: Double) = {
      if (dbl.isNaN || dbl == Double.MinValue || dbl == Double.MaxValue) 0.0 else dbl
    }

    // add features for all enabled models
    for (model <- enabledModels) {
      val vectors = model.vectors
      val (questionFeatures, answerFeatures) = featuresByView(model.view)
      // Step 3: Determine word2vec scores
      if (! disableComposite) {
        val scoreTextSimilarity = checkRange(vectors.textSimilarity(questionFeatures, answerFeatures))
        features.setCount(s"WORD2VEC_${model.name}_TEXTSIM", scoreTextSimilarity)
      }

      if (!disableAvg) {
        val scoreAvgSimilarity = checkRange(vectors.avgSimilarity(questionFeatures, answerFeatures))
        features.setCount(s"WORD2VEC_${model.name}_AVGSIM", scoreAvgSimilarity)
      }

      if (StringUtils.getBool(props, "word2vec.minmax_enabled", true)) {
        // multiple switches for backward compatibility
        if (!disableMax) {
          val scoreMaxSimilarity = checkRange(vectors.maxSimilarity(questionFeatures, answerFeatures))
          features.setCount(s"WORD2VEC_${model.name}_MAXSIM", scoreMaxSimilarity)
        }

        if (!disableMin) {
          val scoreMinSimilarity = checkRange(vectors.minSimilarity(questionFeatures, answerFeatures))
          features.setCount(s"WORD2VEC_${model.name}_MINSIM", scoreMinSimilarity)
        }
      }

      // NB: if you add any features here, make sure to add their names to featureNames as well
    }
    // return list of features
    features
  }

  // set of all features produced by this model
  // this is used by the FeatureCache to determine when we need to call this model, and when we already have the feats
  override lazy val featureNames = (for {
    model <- enabledModels
  } yield (
      (if (disableComposite) List[String]() else List(s"WORD2VEC_${model.name}_TEXTSIM")) ++
      (if (disableAvg) List[String]() else List(s"WORD2VEC_${model.name}_AVGSIM")) ++
      (if (disableMax) List[String]() else List(s"WORD2VEC_${model.name}_MAXSIM")) ++
      (if (disableMin) List[String]() else List(s"WORD2VEC_${model.name}_MINSIM"))
    )
    ).flatten.toSet ++ Set("ir")

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
}

object Word2VecMultisetModel {
  val logger = LoggerFactory.getLogger(classOf[Word2VecMultisetModel])
}
