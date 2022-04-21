package org.clulab.odinsynth.evaluation.fstacred

import java.io.File
import org.clulab.odinsynth.evaluation.ClusterDataset
import org.clulab.odinsynth.evaluation.tacred.TacredRuleGeneration
import java.io.PrintWriter
import org.clulab.processors.fastnlp.FastNLPProcessor
import org.clulab.dynet.Utils.initializeDyNet
import org.clulab.odinsynth.evaluation.tacred.TacredConfig
import ai.lum.common.ConfigFactory
import org.clulab.odinsynth.scorer.DynamicWeightScorer
import org.clulab.odinsynth.Parser
import org.clulab.odinsynth.evaluation.DocumentFromSentences
import org.clulab.odinsynth.TacredSearcher
import upickle.default.read
import org.clulab.odinsynth.using
import org.clulab.odinsynth.EnhancedType
import org.clulab.odinsynth.EnhancedColl
import scala.io.Source
import scala.util.Try
import org.clulab.odinsynth.evaluation.tacred.SolutionStatus
import org.clulab.odinsynth.evaluation.tacred.SolutionSucceeded
import org.clulab.odinsynth.evaluation.tacred.SolutionFailed
import org.clulab.odinsynth.evaluation.tacred.PatternDirection
import org.clulab.odinsynth.evaluation.tacred.SubjObjDirection
import org.clulab.odinsynth.evaluation.tacred.ObjSubjDirection
import org.clulab.odinsynth.evaluation.tacred.TacredSolution
import scala.collection.parallel.ForkJoinTaskSupport
import java.util.concurrent.ForkJoinPool
import java.util.concurrent.atomic.AtomicInteger

/**
  * Generate rules from the data processed by python/tacred_data_prep.py 
  * 
  */
object FewShotTacredGenerateRules extends App {
  // What api endpoints to use
  val apiLocations = Seq(
      // "http://localhost:8000",
      // "http://localhost:8001",
      "http://localhost:8002",
      "http://localhost:8003",
      "http://localhost:8004",
      "http://localhost:8005",
      "http://localhost:8006",
      "http://localhost:8007",
      "http://localhost:8008",
      "http://localhost:8009",
      "http://localhost:8010",
      "http://localhost:8011",
      "http://localhost:8012",
      "http://localhost:8013",
      "http://localhost:8014",
      "http://localhost:8015",
  )

  // Parallelization
  val taskSupport = new ForkJoinTaskSupport(new ForkJoinPool(apiLocations.size))
  val clusterPath = "/data/nlp/corpora/fs-tacred/few-shot-dev/dev-processed/clusters_0_unique/"
  val cache       = "/data/nlp/corpora/fs-tacred/odinsynth_cache"
  val p = {
    initializeDyNet()
    new FastNLPProcessor
  }

  run()

  def run() = {
    val allClusters = new File(clusterPath).listFiles()
                          .map { file => (file.getAbsolutePath(), ClusterDataset(file.getAbsolutePath())) }

    
    val scorers        = apiLocations.map(location => DynamicWeightScorer(location))
    val size           = (allClusters.size / scorers.size).toInt + 1
    // Map from a cluster from a path like ../df_ep6797/3.per_age/cluster_r0_1 to ../cluster_uniques/cluster_1
    val clusterMapping = using(Source.fromFile(f"/data/nlp/corpora/fs-tacred/few-shot-dev/dev-processed/clusters_0_unique_identical_clusters_paths")) { it => 
      val string = it.mkString
      val json = ujson.read(string)//.asInstanceOf[Map[String, String]]
      json.obj.mapValues(_.str)
    }.toMap
    val aidx = new AtomicInteger(1)
    val pw = new PrintWriter(new File("/data/nlp/corpora/fs-tacred/few-shot-dev/dev-processed/clusters_0_unique_all_solutions.tsv"))
    pw.println(TacredSolution.getHeaderWithoutIndex())
    
    // Map from a unique cluster to its optional solution (can be None if it is not found)
    val results = allClusters.sliding(size, size).toSeq.zip(scorers)
                             .par.let { it => it.tasksupport = taskSupport; it }.map { case (clusters, scorer) =>
      clusters.map { case (path, cluster) =>
        val firstType  = Parser.parseBasicQuery(cluster.getFirstObjects.map(_.toLowerCase()).map(it => f"word=$it").mkString(start="[", sep=" | ", end = "]"))
        val secondType = Parser.parseBasicQuery(cluster.getSecondObjects.map(_.toLowerCase()).map(it => f"word=$it").mkString(start="[", sep=" | ", end = "]"))
        val imposeQueryStructure = Some((firstType, secondType))

        val data             = cluster.getData    
        val clusterSentences = data._1.map(_.toSeq).toSeq
        val doc       = synchronized { DocumentFromSentences.documentFromSentencesAndCache(clusterSentences, p, cache, false) } // ProcessorsUtils.convertDocument(p.annotateFromTokens(sentences.map(_.map(_.toLowerCase()))))
        val specs     = data._2
                            .map { it => it.copy(docId = doc.id) }
                            .toSet// ++ (0 until negs.size).map { idx => Spec(doc.id, idx + maxSent, -1, -1, Set()) }   // Write the proper doc id
        val fieldNames = Set("word", "tag", "lemma")

        val searcher = new TacredSearcher(Seq(doc), specs, fieldNames, Some(1000), None, scorer, withReward = false, firstType, secondType)


        val maxSentenceLength = specs.maxBy(it => it.end - it.start).let { it => it.end - it.start }
        val minSentenceLength = specs.minBy(it => it.end - it.start).let { it => it.end - it.start }
        val directionality: PatternDirection = {
          val startEndValues = cluster.pld.map { it => (it("subj_start").toInt, it("subj_end").toInt, it("obj_start").toInt, it("obj_end").toInt) }
          val isSubjObjDirection = startEndValues.forall { case (subjStart, subjEnd, objStart, objEnd) => subjEnd <= objStart }
          val isObjSubjDirection = startEndValues.forall { case (subjStart, subjEnd, objStart, objEnd) => objEnd <= subjStart }
          assert(isSubjObjDirection || isObjSubjDirection, "Either subj <..> obj or obj <..> subj for ALL entries in the cluster. Cannot have a rule that matches on both directions")
          if (isSubjObjDirection) {
            SubjObjDirection
          } else {
            ObjSubjDirection
          }
        }

        
        val q = Try(searcher.findFirst())

        val solutionStatus: SolutionStatus = if (q.isSuccess) {
          SolutionSucceeded
        } else {
          SolutionFailed
        }
            val solution = if (q.isSuccess && q.get.isDefined) {
              TacredSolution(
                            numberOfSentences = specs.size,
                            maxHighlightedLength = maxSentenceLength,
                            minHighlightedLength = minSentenceLength,
                            clusterPath = path,
                            numberOfSteps = searcher.numberOfSteps(),
                            solution = Some(q.get.get.rule.pattern),
                            wrongMatches = None,
                            trials = 0,
                            direction = directionality,
                            relation = cluster.getRelation,
                            scorerVersion = scorer.version,
                            solutionStatus = solutionStatus)
            } else {
              TacredSolution(
                            numberOfSentences = specs.size,
                            maxHighlightedLength = maxSentenceLength,
                            minHighlightedLength = minSentenceLength,
                            clusterPath = path,
                            numberOfSteps = searcher.numberOfSteps(),
                            solution = None,
                            wrongMatches = None,
                            trials = 0,
                            direction = directionality,
                            relation = cluster.getRelation,
                            scorerVersion = scorer.version,
                            solutionStatus = solutionStatus)
            }
        synchronized { pw.println(solution.getString()) }
        println(f"(${aidx.getAndIncrement()}/${allClusters.size}) ${solution.clusterPath} - ${solution.solution.isDefined} (${scorer.apiCaller.scoreEndpoint})")
        solution

      }
    }.toIndexedSeq.flatten
    pw.close()
    using(new PrintWriter(new File("/data/nlp/corpora/fs-tacred/few-shot-dev/dev-processed/clusters_0_unique_all_solutions_copy.tsv"))) { pw =>
      pw.println(TacredSolution.getHeaderWithoutIndex())
      results.foreach { s =>
        pw.println(s.getString())
      }
    }


  }

}
