package org.clulab.odinsynth.scorer

import ai.lum.odinson.Document
import org.clulab.odinsynth.Spec
import org.clulab.odinsynth.Query
import org.clulab.odinsynth.Searcher

class StaticWeightScorer(weights: AstCost) extends Scorer {
  override def score(
    sentences: Seq[Seq[String]],
    specs: Iterable[Spec],
    statesToScore: Iterable[Query],
    currentState: Query
  ): Iterable[Float] = {
    statesToScore.map { it => -it.cost(weights) }
  }
  val version: String = f"StaticWeightScorer(${weights})"
}
object StaticWeightScorer {
  def apply(weights: AstCost = AstCost.getStandardWeights()) = new StaticWeightScorer(weights)
}
