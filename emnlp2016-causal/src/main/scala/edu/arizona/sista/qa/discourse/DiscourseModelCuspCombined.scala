package edu.arizona.sista.qa.discourse

import java.util.Properties
import edu.arizona.sista.qa.ranking.{ProcessedQuestionSegments, ProcessedQuestion, RankingModel}
import edu.arizona.sista.qa.segmenter.QuestionProcessor
import edu.arizona.sista.qa.retrieval.AnswerCandidate
import edu.arizona.sista.struct.Counter
import edu.arizona.sista.processors.Document
import java.io.PrintWriter

/**
 * DMM_IR + DMM_LS
 * User: peter
 * Date: 1/3/14
 */

class DiscourseModelCuspCombined(props:Properties) extends RankingModel {
  val modelConnectivesIR = new DiscourseModelCusp(props)
  val modelConnectivesW2V = new DiscourseModelCuspQSEGW2V(props)

  val qProcessor = new QuestionProcessor(props)

  def mkFeatures( answer:AnswerCandidate,
                  question:ProcessedQuestion,
                  externalFeatures:Option[Counter[String]],
                  errorPw:PrintWriter = null): (Counter[String], String) = {

    question match {
      case q:ProcessedQuestionSegments => (mkFeaturesCuspDispatch (answer, q), null)
      case _ => throw new RuntimeException ("DiscourseModelTreeCombined.mkFeatures(): question passed is not of type ProcessedQuestionSegments")
    }
  }

  def mkFeaturesCuspDispatch (answer:AnswerCandidate,
                              question:ProcessedQuestionSegments): Counter[String] = {
    val featuresQSEGIR = modelConnectivesIR.mkFeatures(answer, question, None)._1
    val featuresQSEGW2V = modelConnectivesW2V.mkFeatures(answer, question, None)._1
    var combined = featuresQSEGIR + featuresQSEGW2V
    combined
  }


  def mkProcessedQuestion(question:Document):Array[ProcessedQuestion] = {
    // Changed to return an array of ProcessedQuestions.  In many cases this will just contain one processedQuestion, but when using the hybrid method it may contain two.
    if (props.getProperty("discourse.question_processor").toUpperCase == "ONESEG") return Array[ProcessedQuestion](qProcessor.mkProcessedQuestionOneArgument(question))
    // default to QSEG if method is unknown
    return Array[ProcessedQuestion](qProcessor.mkProcessedQuestionOneArgument(question))
  }
}
