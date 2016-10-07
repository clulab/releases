package edu.arizona.sista.qa.discourse

import java.util.Properties
import edu.arizona.sista.qa.ranking.{ProcessedQuestionSegments, ProcessedQuestion, RankingModel}
import edu.arizona.sista.qa.retrieval.AnswerCandidate
import edu.arizona.sista.struct.Counter
import edu.arizona.sista.processors.Document
import collection.mutable.ArrayBuffer
import edu.arizona.sista.qa.segmenter.QuestionProcessor
import java.io.PrintWriter


/**
 * Meta-model that combines the DiscourseModelCusp and DiscourseModelTree features
 * User: peter
 * Date: 9/3/13
 */

class DiscourseModelCombined(props:Properties, qsegMode:String) extends RankingModel {
  var modelConnectives:RankingModel = null
  var modelParser:RankingModel = null
  if (qsegMode.toLowerCase == "ir") {
    modelConnectives = new DiscourseModelCusp(props)
    modelParser = new DiscourseModelTree(props)
  } else if (qsegMode.toLowerCase == "w2v") {
    modelConnectives = new DiscourseModelCuspQSEGW2V(props)
    modelParser = new DiscourseModelTreeQSEGW2V(props)
  } else if (qsegMode.toLowerCase == "backoff") {
    modelConnectives = new DiscourseModelCuspBackoff(props)
    modelParser = new DiscourseModelTreeBackoff(props)
  } else if (qsegMode.toLowerCase == "combo") {
    modelConnectives = new DiscourseModelCuspCombined(props)
    modelParser = new DiscourseModelTreeCombined(props)
  } else {
    throw new RuntimeException ("DiscourseModelCombined: qseg mode not recognized (" + qsegMode + "). Valid options are: ir, w2v, backoff.")
  }


  val qProcessor = new QuestionProcessor(props)

  def mkFeatures( answer:AnswerCandidate,
                  question:ProcessedQuestion,
                  externalFeatures:Option[Counter[String]],
                  errorPw:PrintWriter = null): (Counter[String], String) = {

    question match {
      case q:ProcessedQuestionSegments => (mkFeaturesCuspDispatch (answer, q), null)
      case _ => throw new RuntimeException ("DiscourseModelCombined.mkFeatures(): question passed is not of type ProcessedQuestionSegments")
    }

  }


  def mkFeaturesCuspDispatch (answer:AnswerCandidate,
                              question:ProcessedQuestionSegments): Counter[String] = {

    val featuresConnectives = modelConnectives.mkFeatures(answer, question, None)._1
    val featuresParser = modelParser.mkFeatures(answer, question, None)._1
    var combined = featuresConnectives + featuresParser

    return combined
  }


  def mkProcessedQuestion(question:Document):Array[ProcessedQuestion] = {
    // Changed to return an array of ProcessedQuestions.  In many cases this will just contain one processedQuestion, but when using the hybrid method it may contain two.
    if (props.getProperty("discourse.question_processor").toUpperCase == "ONESEG") return Array[ProcessedQuestion](qProcessor.mkProcessedQuestionOneArgument(question))
    if (props.getProperty("discourse.question_processor").toUpperCase == "SVO") return Array[ProcessedQuestion]( qProcessor.mkProcessedQuestionSVO(question) )
    if (props.getProperty("discourse.question_processor").toUpperCase == "HYBRIDSVO") {
      val processedQuestions = new ArrayBuffer[ProcessedQuestion]()
      processedQuestions.append (qProcessor.mkProcessedQuestionOneArgument(question))         // Add ONESEG by default
      processedQuestions.append (qProcessor.mkProcessedQuestionSVO(question))                 // Also use Subj/Verb/Obj/IndObj
      return processedQuestions.toArray
    }

    // default to QSEG if method is unknown
    return Array[ProcessedQuestion](qProcessor.mkProcessedQuestionOneArgument(question))
  }

}
