package org.clulab.odinsynth

import scala.collection.mutable.PriorityQueue
import ai.lum.odinson._
import java.io.PrintWriter
import org.clulab.odinsynth.scorer.Scorer
import org.clulab.odinsynth.scorer.DynamicWeightScorer
import scala.util.Random


// 
case class SynthesizedRule (rule: Query, nSteps: Int, currentSteps: Int) 

// FIXME Only for a single doc. Multiple docs can be considered by merging them in a single doc
class Searcher(
    val docs: Seq[Document],
    val specs: Set[Spec],
    val fieldNames: Set[String],
    val maxSteps: Option[Int] = Some(10000),
    val writer: Option[PrintWriter],
    val scorer: Scorer,
    val withReward: Boolean = false
) {

  assert(specs.map(_.docId).intersect(docs.map(_.id).toSet).nonEmpty, message = f"None of the specifications have their corresponding docId in the docs: specs: {${specs.map(_.docId).toSeq.mkString(", ")}}; docs: {${docs.map(_.id).toSeq.mkString(", ")}}")

  private var currentSteps = 0

  private val random = new Random(1)

  def numberOfSteps(): Int = currentSteps

  def this(
      doc: Document,
      specs: Set[Spec],
      fieldNames: Set[String],
      maxSteps: Option[Int],
      writer: Option[PrintWriter],
      scorer: Scorer,
      withReward: Boolean
  ) = {
    this(Seq(doc), specs, fieldNames, maxSteps, writer, scorer, withReward)
  }

  def this(doc: Document, specs: Set[Spec], fieldNames: Set[String]) =
    this(doc, specs, fieldNames, None, None, DynamicWeightScorer(), false)

  // make extractor engine
  val extractorEngine = ExtractorEngine.inMemory(docs)

  // make vocabularies
  val vocabMaker = new VocabularyMaker(fieldNames)
  val vocabularies = vocabMaker.apply(docs, Seq(specs))

  val emptyQueryArray = new Array[Query](0)

  // Use a subset of the sentences when there are multiple sentences
  // with the same words highlighted
  val sentencesWithHighlight = {
    val specSentenceId = specs.map(_.sentId).toSeq.sorted
    val sentences =
      specSentenceId.map(docs.head.sentences).flatMap { sentence =>
        sentence.fields
          .collect { case f: TokensField => f }
          .filter(_.name == "word")
          .map(_.tokens)
      }

    val qwe = sentences
      .zip(specs.toSeq.sortBy(_.sentId))
      .groupBy { case (sentence, spec) =>
        sentence.slice(spec.start, spec.end).mkString(" ")
      }
      .mapValues(_.map(_._1))

    val result = sentences
      .zip(specs.toSeq.sortBy(_.sentId))
      .groupBy { case (sentence, spec) =>
        sentence.slice(spec.start, spec.end).mkString(" ")
      }
      .mapValues(_.map(_._1))
      .mapValues { s => random.shuffle(s).take(2) }
      .values
      .flatten
      .toSeq

    result
  }

  val (subsetSentences, subsetSpecs, subsetDocs, subsetEe) = {
    val specSentenceId = specs.map(_.sentId).toSeq.sorted
    val sentences =
      specSentenceId.map(docs.head.sentences).flatMap { sentence =>
        sentence.fields
          .collect { case f: TokensField => f }
          .filter(_.name == "word")
          .map(_.tokens)
      }

    if (sentences.size > 10) {
      val result = sentences
        .zip(specs.toSeq.sortBy(_.sentId))
        .groupBy { case (sentence, spec) =>
          sentence.slice(spec.start, spec.end).mkString(" ")
        } //.mapValues(it => it.map(_._1))
        .mapValues { s => random.shuffle(s).take(2) }
        .values
        .flatten
        .toSeq

      val unzipped = result.unzip
      val subsetDoc = Document(
        docs.head.id,
        docs.head.metadata,
        unzipped._2.map(_.sentId).map(docs.head.sentences)
      )
      val subsetEe = ExtractorEngine.inMemory(subsetDoc)
      (
        unzipped._1,
        unzipped._2.zipWithIndex.map { case (spec, idx) =>
          spec.copy(sentId = idx)
        }.toSet,
        subsetDoc,
        subsetEe
      )
    } else {
      (
        sentences,
        specs,
        docs.head,
        extractorEngine
      )
    }
  }

  /** search for a query that satisfies the specification */
  def findFirst(): Option[SynthesizedRule] = {
    findFirst(HoleQuery)
  }

  /** search for a query that satisfies the specification,
    *  starting the node represented by the given pattern
    */
  def findFirst(pattern: String): Option[SynthesizedRule] = {
    findFirst(Parser.parseBasicQuery(pattern))
  }

  /** search for a query that satisfies the specification,
    *  starting from a given node in the search tree
    */
  def findFirst(start: Query): Option[SynthesizedRule] = {
    val iterable = findAll(start)
    val iterator = iterable.iterator
    if (iterator.hasNext) {
      val next = iterator.next()
      if (next != null) {
        return Some(next)
      }
    }
    return None
  }

  /** returns an iterable over all the queries
    *  that satisfy the specification
    */
  def findAll(): Iterable[SynthesizedRule] = {
    findAll(HoleQuery)
  }

  /** returns an iterable over all the queries that satisfy
    *  the specification and that are reachable from the node
    *  represented by the given pattern
    */
  def findAll(pattern: String): Iterable[SynthesizedRule] = {
    findAll(Parser.parseBasicQuery(pattern))
  }

  // necessary to call the python backend
  // and get scores from pytorch
  // val apiCaller = new ApiCaller("http://localhost:8000/score")

  /** returns an iterable over all the queries that satisfy
    *  the specification and that are reachable from the given
    *  node in the search tree
    */
  val searcher = this
  def findAll(start: Query): Iterable[SynthesizedRule] = new Iterable[SynthesizedRule] {
    val t = System.nanoTime()
    def iterator = new Iterator[SynthesizedRule] {
      implicit private val ord = Ordering.by(priority)
      // private val workset = SortedSet( (start, 0.0f) )
      private val workset = PriorityQueue((start, 0.0f))
      private var state: Query = _
      private var stateAvailable: Boolean = false
      private var visited: Set[AstNode] = Set.empty
      // count number of states
      var nStates = 0

      /** returns true if there is a state available, false otherwise */
      def hasNext: Boolean = {
        // if we already know then return true
        if (stateAvailable) return true
        // if we don't know we have to search
        while (workset.nonEmpty) {
          if (((System.nanoTime - t) / 1e9d) >= 60 * 100) {
            return false
          }
          // // Since we have a maximum number of dequeues, we can save space by limiting the number of elements in the workset
          // if (maxSteps.isDefined && workset.size >= maxSteps.get * 5) {
          //   val temp = (0 until maxSteps.get).map(it => workset.dequeue()) // Store top max steps entries
          //   workset.clear() // Remove everything
          //   temp.foreach { it => workset += it } // Add again the top max steps entries
          // }
          
          // remove cheapest state from workset
          state = workset.dequeue()._1
          writer.foreach { it =>
            it.println(f"Explore\t${state.pattern}\t$currentSteps")
          }
          // println(f"Explore\t${state.pattern}\t$currentSteps")
          // println(state.pattern)
          if (!visited.contains(state)) {
            currentSteps += 1
            if (maxSteps.isDefined && currentSteps > maxSteps.get) {
              stateAvailable = false
              return false
            }

            if (isSolution(state)) {
              // we found a solution!
              // take note for the next time hasNext is called
              stateAvailable = true
              visited += state
              return true
            } else {
              // find then states that follow the current state
              // and add them to the workset
              val tmpNextStates = nextStates(state)
              // get scores
              // FIXME: will we ever use two documents here?
              val scores = if (withReward) {
                val rewards = tmpNextStates.map(searcher.reward)
                scorer
                  .score(subsetSentences, subsetSpecs, tmpNextStates, state)
                  .zip(rewards)
                  .map { case (score, reward) => reward / 2 + score }
              } else {
                scorer.score(subsetSentences, subsetSpecs, tmpNextStates, state)
              }

              nStates += tmpNextStates.length
              // val scores = tmpNextStates.map(-_.cost)
              // call API fake
              val nextStatesPlusScore = tmpNextStates.zip(scores)
              //
              // val nextStatesPlusScore = tmpNextStates.map(f=> (f, 0.0f))
              //
              workset ++= nextStatesPlusScore
            }
          }
        }
        // there is nothing left
        false
      }

      /** returns a state if hasNext is true, returns null otherwise */
      def next: SynthesizedRule = {
        if (hasNext) {
          println(s"number of states (nStates) checked: <${nStates}>")
          println(s"number of states (currentSteps) checked: <${currentSteps}>")
          stateAvailable = false
          SynthesizedRule(state, nStates, currentSteps)
        } else {
          null
        }
      }
    }
  }

  /** checks if a given query is a solution,
    *  that is, it is a valid odinson query
    *  and it satisfies the specification
    */
  def isSolution(query: Query): Boolean = {
    if (query.isValidQuery) {
      val results =
        executeQuery(query, disableMatchSelector = false, extractorEngine)
      // FIXME Right now the specs are only using empty captures; Might change in the future
      // FIXME Investigate redundancyCheck and its curious behavior.
      // For rule (?<arg> ([tag=NNP]+)) [word=Mayor] it fails when applied on the sentence "University President Robert Caret and former San Jose Mayor Susan Hammer first discussed the possibility , an elegant new eight-story library , for both city and campus use , is being built in this city of almost one million people ."
      results.map(_.copy(captures = Set.empty)) == specs && underApproximationCheck(query)// && redundancyCheck(query)
    } else {
      false
    }
  }

  /** executes a query and returns the results */
  def executeQuery(
      query: Query,
      disableMatchSelector: Boolean,
      ee: ExtractorEngine
  ): Set[Spec] = {
    // make a rule object
    val rule = Rule(
      name = "Searcher_executeQuery",
      label = None,
      ruletype = "basic",
      priority = "1",
      pattern = query.pattern
    )
    // convert rule object into an extractor
    val extractors = ee.ruleReader.mkExtractors(Seq(rule))
    // use extractor to find matches
    val mentions = ee
      .extractMentions(extractors, disableMatchSelector = disableMatchSelector)
      .toSeq
    // convert mentions to Spec objects
    Spec.fromOdinsonMentions(mentions)
  }

  /** returns all the queries that are reachable
    *  from the given state in the search tree
    */
  def nextStates(state: Query): Array[Query] = {
    state
      .nextNodes(vocabularies)
      .collect { case q: Query =>
        q
      }
      .flatMap(heuristicsCheck)
  }

  def heuristicsCheck(state: Query): Option[Query] = {
    if (state.isValidQuery) {
      Some(state)
    } else if (!overApproximationCheck(state)) {
      // println("\tREJECTED BY OVER-APPROXIMATION")
      None
    } else if (!underApproximationCheck(state)) {
      // println("\tREJECTED BY UNDER-APPROXIMATION")
      None
    } else if (!redundancyCheck(state)) {
      // println("\tREJECTED BY REDUNDANCY")
      //emptyQueryArray
      None
    } else {
      Some(state)
    }
  }

  /** checks if the provided query (or any of its descendents)
    *  can match all positive examples in the specification
    */
  def overApproximationCheck(state: Query): Boolean = {
    state.checkOverApproximation(subsetEe, subsetSpecs.toSet)
  }

  /** checks if the provided query (or any of its descendents)
    *  can reject all negative examples in the specification
    */
  def underApproximationCheck(state: Query): Boolean = {
    state.checkUnderApproximation(subsetEe, subsetSpecs.toSet)
  }

  /** checks if all parts of the given query are really required
    *  to satisfy the specification
    */
  def redundancyCheck(state: Query): Boolean = {
    !state.unroll.hasRedundancy(subsetEe, subsetSpecs.toSet)
    // !redundantChunks(state) && !redundantClauses(state)
  }

  /** unrolls pattern, splits it in chunks, and checks if any chunk is redundant */
  def redundantChunks(state: Query): Boolean = {
    state.unroll.split.distinct.filter(_.isValidQuery).exists { q =>
      val results = executeQuery(q, disableMatchSelector = false, subsetEe)
      results.forall { r =>
        specs.forall { s =>
          r.docId != s.docId || r.sentId != s.sentId || !(r.interval subset s.interval)
        }
      }
    }
  }

  /** checks for ORs with redundant clauses */
  def redundantClauses(state: AstNode): Boolean = {
    state match {
      case ConcatQuery(clauses) =>
        clauses.exists(redundantClauses)
      case OrQuery(clauses) =>
        val qs = clauses.filter(_.isValidQuery)
        qs.length != qs.distinct.length || subsumedClause(qs) || qs.exists(
          redundantClauses
        )
      case RepeatQuery(q, min, max) =>
        redundantClauses(q)
      case TokenQuery(constraint) =>
        redundantClauses(constraint)
      case OrConstraint(clauses) =>
        val qs = clauses.filter(_.isValidQuery)
        qs.length != qs.distinct.length || subsumedClause(qs.map(TokenQuery))
      case AndConstraint(clauses) =>
        val qs = clauses.filter(_.isValidQuery)
        qs.length != qs.distinct.length || subsumedClause(qs.map(TokenQuery))
      case _ => false
    }
  }

  def subsumedClause(queries: Vector[Query]): Boolean = {
    val results =
      queries.map(q => executeQuery(q, disableMatchSelector = false, subsetEe))
    var i = 0
    var j = 0
    while (i < results.length) {
      j = i + 1
      while (j < results.length) {
        if (
          (results(i) union results(j)).size == math.max(
            results(i).size,
            results(j).size
          )
        ) {
          return true
        }
        j += 1
      }
      i += 1
    }
    false
  }

  /** returns the node's priority (bigger is better) */
  def priority(node: Pair[Query, Float]): Float = node._2
  // def priority(node: Pair[Query, Float]): Float = -node._1.cost

  var results: Set[Spec] = _
  /* returns the percentage of the spec that matches with the state */
  def reward(node: Query): Float = {
    // get a valid query
    if (node.isValidQuery) {
      results = executeQuery(node, disableMatchSelector = false, subsetEe)
    } else if (node.getValidQuery.isDefined) {
      results = executeQuery(
        node.getValidQuery.get,
        disableMatchSelector = false,
        subsetEe
      )
    } else {
      return -0.0f
    }

    // make sets to operate over
    def sumSets(sets: Set[Spec]) = {
      sets
        .map(s => (s.start until s.end).toSet)
        .reduce((s1, s2) => s1 ++ s2)
    }

    val resultsBySentId =
      results.groupBy(f => f.sentId).map(f => f._1 -> sumSets(f._2))

    val specsBySentId =
      specs.groupBy(f => f.sentId).map(f => f._1 -> sumSets(f._2))

    var goodMatch = 0.0f
    var badMatch = 0.0f
    // FIXME: assuming we will only have a single document for now
    val doc = docs.head

    for (sId <- 0 to doc.sentences.size) {
      // initialize sets as empty sets
      // if sentence_id in spec
      if (specs.exists(s => s.sentId == sId)) {
        //println(s"sentence ${sId} is in the spec")
        // when both exists
        if (results.exists(r => r.sentId == sId)) {
          // calculate how much was correct
          // what was unmatched on the spec
          val unmatchedSet = specsBySentId(sId) -- resultsBySentId(sId)
          // check how many good matches
          goodMatch += (specsBySentId(sId).size - unmatchedSet.size)
          // what was matched but should not
          val matchedSet = resultsBySentId(sId) -- specsBySentId(sId)
          badMatch += matchedSet.size
          //println(s"matched but should not ${matchedSet}, should be matched but did not ${unmatchedSet}")
          // calculate how much was wrong
        } else {
          // when the result does not exist
          // everything is wrong?
          badMatch += specsBySentId(sId).size
        }
      } else {
        // when there are no specs
        if (results.exists(r => r.sentId == sId)) {
          badMatch += resultsBySentId(sId).size
        }
        // everything is wrong again?
      }
    }
    //print(s"badmatch: ${badMatch} goodmatch: ${goodMatch}")
    goodMatch - badMatch
    //-badMatch
  }

  def rewardAsF1(node: Query): Double = {
    if (node.isValidQuery) {
      results = executeQuery(node, disableMatchSelector = false, subsetEe)
    } else if (node.getValidQuery.isDefined) {
      results = executeQuery(
        node.getValidQuery.get,
        disableMatchSelector = false,
        subsetEe
      )
    } else {
      return 0.0f
    }

    val resultsBySentId =
      results.groupBy(f => f.sentId).map(f => f._1 -> f._2.flatMap(it => (it.start until it.end).toSet))

    val specsBySentId =
      specs.groupBy(f => f.sentId).map(f => f._1 -> f._2.flatMap(it => (it.start until it.end).toSet))
    
    // println(resultsBySentId)
    // println(specsBySentId)

    var tp = resultsBySentId.map { case (sentId, set) => specsBySentId(sentId).intersect(set).size }.sum.toDouble
    var fp = resultsBySentId.map { case (sentId, set) => set.diff(specsBySentId(sentId)).size }.sum.toDouble
    var fn = resultsBySentId.map { case (sentId, set) => specsBySentId(sentId).diff(set).size }.sum.toDouble

    val p = if (tp + fp == 0) {
      0
    } else {
      tp / (tp + fp)
    }    
    val r = if (tp + fn == 0) {
      0
    } else {
      tp / (tp + fn)
    }


    if (p + r == 0) {
      return 0.0
    } else {
      return (2 * p * r) / (p+r)
    }
  }

}
