package org.clulab.odinsynth

import scala.collection.mutable.PriorityQueue
import ai.lum.odinson._
import java.io.PrintWriter
import org.clulab.odinsynth.scorer.Scorer
import org.clulab.odinsynth.scorer.DynamicWeightScorer
import scala.util.Random
import scala.collection.mutable
import org.clulab.processors.fastnlp.FastNLPProcessor


/**
  * Particular to this type of searcher is that not all highlighted units
  * can have the same constraint. For example, for the highlight "X was born in Y",
  * we might we might want something like: "[ne=PER] [tag=VBD] [lemma=born] [tag=IN] [ne=LOC]"
  * instead of "[word=X] [tag=VBD] [lemma=born] [tag=IN] [word=Y]"
  * That is, not all highlighted words allow for the same vocabulary
  *
  * @param originalDoc -> the document; Not a sequence anymore
  * @param specs       -> the specs, but as a sequence of set. This way we separate specs that need
  *                       masking from those that don't. That is, 
  * @param masked      -> a sequence of booleans, specifying whether the corresponding spec from specs (same index)
  *                       needs masking or not
  * @param fieldNames  -> same as in a traditional searcher
  * @param maxSteps    -> same as in a traditional searcher
  * @param writer      -> same as in a traditional searcher
  * @param scorer      -> same as in a traditional searcher
  * @param withReward  -> same as in a traditional searcher
  * @param firstType   -> same as in a traditional searcher
  * @param secondType  -> same as in a traditional searcher
  */
class MaskedSearcher(
  originalDoc: Document,
  specs: Seq[Set[Spec]],
  masked: Seq[Boolean],
  fieldNames: Set[String],
  maxSteps: Option[Int] = Some(10000),
  writer: Option[PrintWriter],
  scorer: Scorer,
  withReward: Boolean = false,
) { //extends Searcher(docs, specs, fieldNames, maxSteps, writer, scorer, withReward) {

  private val fullSearcher = new Searcher(originalDoc, specs.flatten.groupBy(_.sentId).values.map { it => Spec(it.head.docId, it.head.sentId, it.minBy(_.start).start, it.maxBy(_.end).end) }.toSet, fieldNames)
  private val searchers = {
    val slicedDocs = specs.map { spec => 
      originalDoc.copy(
        sentences = originalDoc.sentences.zip(spec).map { case (sentence, spec) => sentence.copy(numTokens = spec.end - spec.start, fields=sentence.fields.collect { case tf@TokensField(name, tokens) => tf.copy(tokens=tokens.slice(spec.start, spec.end)) }) }
      )
    }
    // slicedDocs.foreach(println)
    specs.zip(masked).zip(slicedDocs).map { //case (spec, mask) =>
      case ((spec, true), doc)  => new Searcher(Seq(doc), spec.map(it => it.copy(start=0, end=it.end-it.start)), fieldNames.diff(Set("word", "lemma")), maxSteps, writer, scorer, withReward)
      case ((spec, false), doc) => new Searcher(Seq(doc), spec.map(it => it.copy(start=0, end=it.end-it.start)), fieldNames, maxSteps, writer, scorer, withReward)
      // new Searcher(docs, spec, fieldNames, maxSteps, writer, scorer, withReward)
    }
  }


  def findFirst(): Option[SynthesizedRule] = {
    val iterable = findAll()
    val iterator = iterable.iterator
    if (iterator.hasNext) {
      val next = iterator.next()
      if (next != null) {
        return Some(next)
      }
    }
    return None
  }

  def findAll(): Iterable[SynthesizedRule] = {
    val solutions = searchers.map { it => 
      // By-pass the scorer if there is only one thing highlighted
      if (it.specs.forall { s => s.start + 1 == s.end }) {
        it.vocabularies.toList.sortBy(_._2.size).map { case (name, values) =>
          val orConstraint = OrConstraint(values.map { v => FieldConstraint(StringMatcher(name),StringMatcher(v)) }.toVector)
          val result: Query = ConcatQuery(Vector(TokenQuery(orConstraint)))
          SynthesizedRule(result, 0, 0)
        }.toIterable
      } else {
        it.findAll() 
      }
    }
    if (solutions.forall(_.nonEmpty)) {
      val solutionAndMasks  = solutions.zip(masked).zipWithIndex.map { case ((rules, b), i) => (rules.iterator, b, i) }
      val solutionsForMasks = solutionAndMasks.filter(_._2).map { case (iterator, masked, idx) => 
        val next = iterator.next()
        (idx, next.copy(rule = NamedCaptureQuery(Parser.parseBasicQuery(next.rule.pattern)))) 
      }
      .toMap
      // val currentSolution  = solutionAndMasks
      // val nSteps           = solutions.map(_.nSteps).sum
      // val currentSteps     = solutions.map(_.currentSteps).sum
      new Iterable[SynthesizedRule] {
        def iterator: Iterator[SynthesizedRule] = new Iterator[SynthesizedRule] {
          def hasNext: Boolean = {
            solutionAndMasks.filterNot(_._2).forall(_._1.nonEmpty)
          }
          def next(): SynthesizedRule = {
            solutionAndMasks.map {
              case (rules, true, position)  => {
                val resultingRule = Parser.parseBasicQuery(QuerySimplifier.simplifyQuery(solutionsForMasks(position).rule).pattern).asInstanceOf[NamedCaptureQuery]
                val modifiedQuery = resultingRule.query match {
                case it@MatchAllQuery                     => ???
                case it@HoleQuery                         => ???
                case it@TokenQuery(constraint)            => RepeatQuery(it, 1, None)
                case it@ConcatQuery(queries)              => it
                case it@OrQuery(queries)                  => it
                case it@RepeatQuery(query, min, max)      => it
                case it@NamedCaptureQuery(query, argName) => it
                }

                solutionsForMasks(position).copy(rule = resultingRule.copy(query=modifiedQuery))
              }
              case (rules, false, position) => rules.next()
            }.let { result =>
              if(result.exists(_ == null)) {
                null
              } else {
                val nSteps       = result.map { rule => rule.nSteps }.sum
                val currentSteps = result.map { rule => rule.currentSteps }.sum
                val finalRules = SynthesizedRule(ConcatQuery(result.map { rule => rule.rule }.toVector), nSteps, currentSteps)
                if(fullSearcher.isSolution(finalRules.rule)) {
                  finalRules
                } else {
                  if (hasNext) {
                    next()
                  } else {
                    null
                  }
                }
              }
            }
          }
        }
      }
    } else {
      Iterable.empty
    }
  }
}

// An example of MaskedSearcher that is as self-contained as possible
object MaskedSearcherExample extends App {
  import scala.util.Try
  import scala.util.Failure
  import org.clulab.odinsynth.EnhancedType
  import org.clulab.processors.fastnlp.FastNLPProcessor
  import edu.cmu.dynet.Initialize
  import org.clulab.odinsynth.scorer.DynamicWeightScorer
  import org.clulab.odinsynth.evaluation.DocumentFromSentences
  import org.clulab.dynet.Utils.initializeDyNet


  val q = Seq(
    Seq("Joanna", "was", "born", "in", "Phoenix", "."),
    Seq("John", "was", "born", "in", "Tucson", "."),
  )
  val p = {
    initializeDyNet()
    new FastNLPProcessor
  }

  val d = DocumentFromSentences.documentFromSentencesKeepCase(q, p)
  println(d.sentences)
  println(d.sentences.size)
  println("-"*100)
  println("a")
  println("b")
  
  val specs = Seq(
    // Splitting the spec into 3 sets:
    Set(Spec(d.id, 0, 0, 1), Spec(d.id, 1, 0, 1)), // 1: the first entity
    Set(Spec(d.id, 0, 1, 4), Spec(d.id, 1, 1, 4)), // 2: the words in-between
    Set(Spec(d.id, 0, 4, 5), Spec(d.id, 1, 4, 5)), // 3: the second entity
    // the first and second entity will be masked, which means that we will not use the full vocabulary when generating them
  )
  val masked = Seq(true, false, true)
  val scorer = DynamicWeightScorer("http://localhost:8002")
  val searcher = new MaskedSearcher(d, specs, masked, Set("word", "lemma", "tag"), Some(1000), None, scorer, false)
  println(searcher.findAll().map(_.rule.pattern).take(3).toList)
}
