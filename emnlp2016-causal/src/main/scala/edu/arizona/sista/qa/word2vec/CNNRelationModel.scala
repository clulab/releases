package edu.arizona.sista.qa.word2vec

import java.io.PrintWriter
import java.util.Properties

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
import CNNRelationModel.logger

import scala.collection.mutable.ArrayBuffer

/**
  * Created by bsharp on 5/9/16.
  */
class CNNRelationModel(props:Properties) extends RankingModel{

  // store information needed to load a w2v model
  case class CNNAndInfo(CNN: CNNRelation,
                        view:String,
                        name: String)

  val qProcessor = new QuestionProcessor(props)
  // Cause -> Effect
  //  val targetVectorFilename = props.getProperty("word2vec_relation.target_vectors", "")
  //  val contextVectorFilename = props.getProperty("word2vec_relation.context_vectors", "")
  //  val model = new Word2VecRelation(targetVectorFilename, contextVectorFilename, None)
  //  // Effect --> Cause
  //  val targetVectorFilename2 = props.getProperty("word2vec_relation.e2c.target_vectors", "")
  //  val contextVectorFilename2 = props.getProperty("word2vec_relation.e2c.context_vectors", "")
  //  val model2 = new Word2VecRelation(targetVectorFilename2, contextVectorFilename2, None)

  val defaultView = "lemmas_content"
  val viewName = props.getProperty("view.view", defaultView)

  lazy val processor = new FastNLPProcessor()
  val termFilter = new TermFilter

  // If using as backoff for the relation lookup model
  val useAsBackoff = StringUtils.getBool(props, "cnn_relation.use_as_backoff", default = false)
  lazy val relationLookup = new RelationLookup(props)

  // maximum number of matrices to use (used to determine number of properties to look at)
  val nCNNs = StringUtils.getInt(props, "cnn_relation.n_cnns", 20)

  // read the properties corresponding to a given index model, and if the model's enabled, return info for it
  def CNNForIndexFromProperties(index: Int): Option[CNNAndInfo] = {
    val propertyPrefix = s"cnn_relation.model${index}."
    val filenameInputTrain = props.getProperty(propertyPrefix + "keras_output_train")
    val filenameInputTest = props.getProperty(propertyPrefix + "keras_output_test")
    val filenameTrainInfo = props.getProperty(propertyPrefix + "candidateinfo_train")
    val filenameTestInfo = props.getProperty(propertyPrefix + "candidateinfo_test")
    val enabled = StringUtils.getBool(props, propertyPrefix + "enable", false)
    val view = props.getProperty(propertyPrefix + "override_view", defaultView)
    val name = props.getProperty(propertyPrefix + "name")

    if (! enabled)
      None
    else {
      Some(CNNAndInfo(
        CNN = new CNNRelation(filenameInputTrain, filenameInputTest, filenameTrainInfo, filenameTestInfo),
        view = view,
        name = name
      ))
    }
  }

  // read the properties and get a list of info for all enabled models
  val enabledModels: Seq[CNNAndInfo] = for {
    index <- (1 to nCNNs)
    info <- CNNForIndexFromProperties(index)
  } yield info


  def mkFeatures( answer:AnswerCandidate,
                  question:ProcessedQuestion,
                  externalFeatures:Option[Counter[String]],
                  errorPw:PrintWriter = null): (Counter[String], String) = {

    question match {
      case q:ProcessedQuestionSegments => (mkFeaturesCuspDispatch (answer, q), null)
      case _ => throw new RuntimeException ("CNNRelationModel.mkFeatures(): question passed is not of type ProcessedQuestionSegments")
    }

  }

  def mkFeaturesCuspDispatch (answer:AnswerCandidate,
                              question:ProcessedQuestionSegments): Counter[String] = {
    return mkFeaturesCNNRelation(answer, question)
  }

  def mkProcessedQuestion(question:Document):Array[ProcessedQuestion] = {
    // default to QSEG if method is unknown
    return Array[ProcessedQuestion](qProcessor.mkProcessedQuestionOneArgument(question))
  }

  def mkFeaturesCNNRelation( answer:AnswerCandidate,
                                  question:ProcessedQuestionSegments): Counter[String] = {

    val features = new Counter[String]()

    // Add basic IR feature
    if (props.getProperty("ranker.train_method", "incremental").toLowerCase != "hierarchical") {
      features.setCount("ir", answer.answerScore)
    }

    // Step 1A: If we failed to properly segment the question, then return an empty feature list.  Default back to other methods (e.g. IR)
    if (question.questionType == "failed") return features


    def checkRange(dbl: Double) = {
      if (dbl.isNaN || dbl == Double.MinValue || dbl == Double.MaxValue) 0.0 else dbl
    }

    for (model <- enabledModels) {

      // Determine CNN scores
      val currCandidateID = answer.doc.docid
      val score = checkRange(model.CNN.getScore(currCandidateID))
      features.setCount(s"CNN_REL_${model.name}_SCORE", score)
    }

    // return list of features
    features
  }


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


}


object CNNRelationModel {
  val logger = LoggerFactory.getLogger(classOf[CNNRelationModel])

  val causeRegexes = Word2VecRelationModel.causeRegexes
  val resultRegexes = Word2VecRelationModel.resultRegexes

}

