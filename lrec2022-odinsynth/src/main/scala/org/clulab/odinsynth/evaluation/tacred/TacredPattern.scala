package org.clulab.odinsynth.evaluation.tacred

/**
  * Holding a pattern for TACRED data
  *
  * @param pattern  : the pattern
  * @param relation : the relation associated with the pattern
  * @param direction: the directionality of this pattern (SubjObj or ObjSubj), which tells the way
  *                   the entities appeared in the data on which this pattern was generated
  * @param weight   : the associated weight
  */
final case class TacredPattern(pattern: String, relation: String, direction: PatternDirection, weight: Double)