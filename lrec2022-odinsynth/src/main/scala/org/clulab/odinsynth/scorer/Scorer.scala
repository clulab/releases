package org.clulab.odinsynth.scorer

import ai.lum.odinson.Document
import org.clulab.odinsynth.Spec
import org.clulab.odinsynth.Query
import org.clulab.odinsynth.Searcher

trait Scorer {
  
  def score(
    sentences: Seq[Seq[String]], 
    specs: Iterable[Spec], 
    statesToScore: Iterable[Query], 
    currentState: Query
  ): Iterable[Float]

  def version: String
  
}
