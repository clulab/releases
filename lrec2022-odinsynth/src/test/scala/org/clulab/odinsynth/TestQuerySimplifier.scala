package org.clulab.odinsynth

import org.scalatest._
import org.clulab.odinsynth.Parser.parseBasicQuery
import org.clulab.odinsynth.QuerySimplifier.simplifyQuery

class TestQuerySimplifier extends FlatSpec with Matchers {

  val q1 = "[word=was] [tag=VBN] [tag=IN] (?<arg> ([tag=NNP]? [tag=NNP]? [tag=NNP] [tag=NNP]?))"
  val q2 = "[word=was] [tag=VBN] [tag=IN] (?<arg> ([tag=NNP]? [tag=NNP]? [tag=NNP]?))"
  val q3 = "[word=was] [tag=VBN] [tag=IN] (?<arg> ([tag=NNP]? [tag=NNP]? [tag=NNP]? [tag=NNP]?))"
  val q4 = "[word=was] [tag=VBN] [tag=IN] (?<arg> ([tag=NNP]? [tag=NNP]?))"

  "simplifyQuery" should "should replace consecutive queries with same underlying constraint with a repeat query" in {
    assert(
      simplifyQuery(parseBasicQuery(q1), Some(100)) 
      ==
      parseBasicQuery("[word=was] [tag=VBN] [tag=IN] (?<arg> ([tag=NNP]{1,4}))")
    )
    assert(
      simplifyQuery(parseBasicQuery(q2), Some(2)) 
      ==
      parseBasicQuery("[word=was] [tag=VBN] [tag=IN] (?<arg> ([tag=NNP]*))")
    )
    assert(
      simplifyQuery(parseBasicQuery(q3), Some(4)) 
      ==
      parseBasicQuery("[word=was] [tag=VBN] [tag=IN] (?<arg> ([tag=NNP]{0,4}))")
    )
    assert(
      simplifyQuery(parseBasicQuery(q4), Some(1)) 
      ==
      parseBasicQuery("[word=was] [tag=VBN] [tag=IN] (?<arg> ([tag=NNP]*))")
    )
    assert(
      simplifyQuery(parseBasicQuery(q4), Some(100))
      ==
      parseBasicQuery("[word=was] [tag=VBN] [tag=IN] (?<arg> ([tag=NNP]{0,2}))")
    )
  }
}
