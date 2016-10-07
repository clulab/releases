package edu.arizona.sista.qa.ranking

import java.util.Properties
import edu.arizona.sista.qa.retrieval.AnswerCandidate
import edu.arizona.sista.struct.Counter
import edu.arizona.sista.processors.Document
import edu.arizona.sista.qa.segmenter.QuestionProcessor
import java.io.PrintWriter


/**
 * Meta-model that combines several basic ranking models
 * User: dfried
 * Date: 5/12/14
 */

class MultiModel(props:Properties, models: Seq[RankingModel]) extends RankingModel {
  val qProcessor = new QuestionProcessor(props)

  def mkFeatures( answer:AnswerCandidate,
                  question:ProcessedQuestion,
                  externalFeatures:Option[Counter[String]],
                  errorPw:PrintWriter = null): (Counter[String], String) =
    question match {
      case q:ProcessedQuestionSegments => (mkFeatures(answer, q), null)
      case _ => throw new RuntimeException ("MultiModel.mkFeatures(): question passed is not of type ProcessedQuestionSegments")
    }

  def featureUnion(c1: Counter[String], c2: Counter[String]): Counter[String] = {
    val c = new Counter[String]
    for ((k, v) <- c1.toSeq) c.setCount(k, v)
    for ((k, v) <- c2.toSeq) {
      if (c.contains(k) && c.getCount(k) != v)
        throw new Exception(s"warning: non-identical values in featureValues for feature $k: ${v} vs ${c.getCount(k)}")
      c.setCount(k, v)
    }
    c
  }

  def mkFeatures (answer:AnswerCandidate,
                  question:ProcessedQuestionSegments): Counter[String] = {
    val multiFeats: Counter[String] = models.map(model => model.mkFeatures(answer, question, None)._1).reduce(featureUnion)
    println(s"multi model features: ${multiFeats.toShortString}")
    multiFeats
  }

  def mkProcessedQuestion(question:Document):Array[ProcessedQuestion] =
    Array[ProcessedQuestion](qProcessor.mkProcessedQuestionOneArgument(question))

  override lazy val featureNames: Set[String] = {
    models.map(_.featureNames).reduce(_ ++ _)
  }

}
