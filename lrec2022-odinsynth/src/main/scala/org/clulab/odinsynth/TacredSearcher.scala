package org.clulab.odinsynth

import scala.collection.mutable.PriorityQueue
import ai.lum.odinson._
import java.io.PrintWriter
import org.clulab.odinsynth.scorer.Scorer
import org.clulab.odinsynth.scorer.DynamicWeightScorer
import scala.util.Random
import scala.collection.mutable

// FIXME Only for a single doc. Multiple docs can be considered by merging them in a single doc
// Handles trivial cases:
//    - nothing in between the entities
//        - we return a concatenation of the entity types
//    - only one word in between the entities
//        - we return an or with all the types; we favor the ones with the least number of ors
//          for example, if the vocabulary is like: {"tag": ["NN"], "word": ["car", "vehicle"]}
//          we use [tag=NN]
class TacredSearcher(
  docs: Seq[Document],
  specs: Set[Spec],
  fieldNames: Set[String],
  maxSteps: Option[Int] = Some(10000),
  writer: Option[PrintWriter],
  scorer: Scorer,
  withReward: Boolean = false,
  val firstType: Query,
  val secondType: Query,
) extends Searcher(docs, specs, fieldNames, maxSteps, writer, scorer, withReward) {

  override def findFirst(): Option[SynthesizedRule] = { 
    if (specs.forall { s => s.start == s.end }) {
      Some(SynthesizedRule(ConcatQuery(Vector(firstType, secondType)),0,0))
    } else if (specs.forall { s => s.start + 1 == s.end }) {
      val (name, values) = vocabularies.toList.sortBy(_._2.size).head
      val orConstraint = OrConstraint(values.map { v => FieldConstraint(StringMatcher(name),StringMatcher(v)) }.toVector)
      Some(SynthesizedRule(ConcatQuery(Vector(firstType, TokenQuery(orConstraint), secondType)),0,0))
    } else {
      super.findFirst().map { q => 
        SynthesizedRule(ConcatQuery(Vector(firstType, q.rule, secondType)),q.nSteps,q.currentSteps)
      }
    }
  }

  /**
    * Get multiple solutions
    * 
    * The number of solutions is specified in @param maxSol
    * It doesn't guarantee that it will find the specified number of solutions, but
    * that it will try to find at most that number of solutions and return all of them
    *
    * @param maxSol
    * @return
    */
  def getAll(maxSol: Int): Seq[SynthesizedRule] = {
    if (specs.forall { s => s.start == s.end }) {
      Seq(SynthesizedRule(ConcatQuery(Vector(firstType, secondType)),0,0))
    } else if (specs.forall { s => s.start + 1 == s.end }) {
      val orConstraints = vocabularies.toList
                            .sortBy(_._2.size)
                            .map { case (name, values) => OrConstraint(values.map { v => FieldConstraint(StringMatcher(name),StringMatcher(v)) }.toVector) }
                            .take(maxSol)

      orConstraints.map { orConstraint => SynthesizedRule(ConcatQuery(Vector(firstType, TokenQuery(orConstraint), secondType)),0,0) }

    } else {
      var maxSolutions = maxSol
      val iterable = searcher.findAll()
      val iterator = iterable.iterator
      val solutions = mutable.ListBuffer.empty[SynthesizedRule]
      while (iterator.hasNext && maxSolutions > 0) {
        maxSolutions -= 1
        val next = iterator.next()
        if (next != null) {
          solutions.append(SynthesizedRule(ConcatQuery(Vector(firstType, next.rule, secondType)),next.nSteps,next.currentSteps))
        }
      }
      solutions.toSeq
    }
  }

}
