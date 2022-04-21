package org.clulab.odinsynth

import org.scalatest._
import ai.lum.odinson._
import org.clulab.odinsynth.scorer.StaticWeightScorer

class TestHeuristics extends FlatSpec with Matchers {

  def mkFixture(jsonDocs: Seq[String], specs: Set[Spec]) = new {
    val docs = jsonDocs.map(Document.fromJson)
    val fieldNames = Set("word", "tag")
    val searcher = new Searcher(docs, specs, fieldNames, None, None, StaticWeightScorer())
  }

  def gummyBearsFixture = {
    val json = """{"id":"test","metadata":[],"sentences":[{"numTokens":5,"fields":[{"$type":"ai.lum.odinson.TokensField","name":"raw","tokens":["Becky","ate","gummy","bears","."],"store":true},{"$type":"ai.lum.odinson.TokensField","name":"word","tokens":["Becky","ate","gummy","bears","."]},{"$type":"ai.lum.odinson.TokensField","name":"tag","tokens":["NNP","VBD","JJ","NNS","."]},{"$type":"ai.lum.odinson.TokensField","name":"lemma","tokens":["becky","eat","gummy","bear","."]},{"$type":"ai.lum.odinson.TokensField","name":"entity","tokens":["I-PER","O","O","O","O"]},{"$type":"ai.lum.odinson.TokensField","name":"chunk","tokens":["B-NP","B-VP","B-NP","I-NP","O"]},{"$type":"ai.lum.odinson.GraphField","name":"dependencies","edges":[[1,0,"nsubj"],[1,3,"dobj"],[1,4,"punct"],[3,2,"amod"]],"roots":[1]}]}]}"""
    val specs = Set(Spec("test", 0, 0, 2))
    mkFixture(Seq(json), specs)
  }

  "over-approximation" should "accept feasible branch" in {
    val f = gummyBearsFixture
    val pattern = "□ [word=ate]"
    val query = Parser.parseBasicQuery(pattern)
    f.searcher.overApproximationCheck(query) should be (true)
  }

  it should "reject infeasible branch" in {
    val f = gummyBearsFixture
    val pattern = "[word=ate] □"
    val query = Parser.parseBasicQuery(pattern)
    f.searcher.overApproximationCheck(query) should be (false)
  }

  "redundancy" should "accept non-redundant query" in {
    val f = gummyBearsFixture
    val pattern = "[word=Becky] [word=ate]"
    val query = Parser.parseBasicQuery(pattern)
    f.searcher.redundancyCheck(query) should be (true)
  }

  it should "reject redundant query" in {
    val f = gummyBearsFixture
    val pattern = "[word=Becky | word=bears] [word=ate]"
    val query = Parser.parseBasicQuery(pattern)
    f.searcher.redundancyCheck(query) should be (false)
  }

}
