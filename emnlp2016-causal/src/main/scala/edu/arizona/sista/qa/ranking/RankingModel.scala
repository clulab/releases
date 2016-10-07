package edu.arizona.sista.qa.ranking

import edu.arizona.sista.qa.retrieval.AnswerCandidate
import edu.arizona.sista.struct.Counter
import edu.arizona.sista.processors.Document
import java.io.PrintWriter

/**
 * Trait for all ranking models
 * User: mihais
 * Date: 4/30/13
 */
trait RankingModel {

  /**
   * Construct the features for this datum
   * @param answer
   * @param question
   * @param externalFeatures
   * @return A tuple containing explicit features and a kernelized version of the datum, if any (null, if non existing)
   */
  def mkFeatures(
    answer:AnswerCandidate,
    question:ProcessedQuestion,
    externalFeatures:Option[Counter[String]],
    errorPw:PrintWriter = null): (Counter[String], String)

  def mkProcessedQuestion(question:Document):Array[ProcessedQuestion]

  def usesKernels:Boolean = false

  // implement this for any models whose features will be cached
  def featureNames: Set[String] = sys.error("featureNames not yet implemented for " + this.getClass.getName)

}
