package org.clulab.odinsynth.scorer

import ai.lum.odinson.Document
import org.clulab.odinsynth.{Query, Spec, Searcher}
import org.clulab.odinsynth.ApiCaller

class DynamicWeightScorer private (val apiCaller: ApiCaller) extends Scorer {

  override def score(
    sentences: Seq[Seq[String]],
    specs: Iterable[Spec],
    statesToScore: Iterable[Query],
    currentState: Query
  ): Iterable[Float] = {
    apiCaller.getScores(sentences, specs.toSet, statesToScore.toSeq.map(_.pattern), currentState.pattern)
  }

  override def version: String = apiCaller.getVersion()
}
object DynamicWeightScorer {

  def apply(apiCaller: ApiCaller): DynamicWeightScorer = new DynamicWeightScorer(apiCaller)

  def apply(address: String = "http://localhost:8000"): DynamicWeightScorer = new DynamicWeightScorer(new ApiCaller(address))

}
