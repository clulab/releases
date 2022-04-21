package org.clulab.odinsynth.evaluation.tacred

import scala.io.StdIn.readLine
import scala.util.control.Breaks._
import ai.lum.odinson._
import org.clulab.processors.clu.CluProcessor
import _root_.org.clulab.processors.{Document => CluDocument}
import org.clulab.processors.fastnlp.FastNLPProcessor
import scala.collection.parallel.ForkJoinTaskSupport
import java.util.concurrent.ForkJoinPool

import ai.lum.odinson.extra.ProcessorsUtils
import scala.collection.mutable
import edu.cmu.dynet.Initialize
import java.io.File
import java.io.PrintWriter
import java.io.FileWriter
import scala.util.Random
import org.clulab.odinsynth.Searcher
import org.clulab.odinsynth.ApiCaller
import org.clulab.odinsynth.Spec
import org.clulab.odinsynth.Query
import org.clulab.odinsynth.using
import org.clulab.odinsynth.scorer.DynamicWeightScorer
import scala.io.Source
import java.io.BufferedWriter
import org.clulab.odinsynth.scorer.Scorer
import org.clulab.odinsynth.scorer.StaticWeightScorer
import org.clulab.odinsynth.evaluation.PandasLikeDataset
import org.clulab.odinsynth.evaluation.DocumentFromSentences
import com.typesafe.scalalogging.LazyLogging
import java.nio.file.Files
import java.nio.file.Paths
import scala.util.Try
import java.{util => ju}
import org.clulab.odinsynth.TacredSearcher
import org.clulab.odinsynth.HoleQuery
import org.clulab.odinsynth.evaluation.ClusterDataset
import org.clulab.odinsynth.Parser
import org.clulab.dynet.Utils.initializeDyNet

/**
 * 
 * Reads each cluster and attempts to generate a rule
 * Run with:
 *          sbt -J-Xmx32g -Djava.io.tmpdir=~/sbttmp/ 'runMain org.clulab.odinsynth.evaluation.tacred.TacredRuleGeneration static_rule_generation.conf' -Dodinsynth.evaluation.tacred.skipClusters=0 -Dodinsynth.evaluation.tacred.takeClusters=10 -Dodinsynth.evaluation.tacred.ruleBasepath="odinsynth_tacred_generated_rules/static/s1"
 */
object TacredRuleGeneration extends App with LazyLogging {

  override def main(args: Array[String]): Unit = {
    
    val tacredConfig = TacredConfig.from(args.head)
    val r = new Random(1)
    val taskSupport = new ForkJoinTaskSupport(new ForkJoinPool(2))
    val scorer = tacredConfig.scorer

    val p = {
      initializeDyNet()
      new FastNLPProcessor
    }
    
    println(new File(tacredConfig.clustersPath).listFiles().filter(_.isDirectory()).toSeq
                        .sortBy(_.getAbsolutePath().split("/").last.split("\\.").head.toInt))
    println("-"*100)
    val relationNames = Seq(
      // ("no_relation", f"${tacredConfig.clustersPath}/0.no_relation_subset"),
      ("org:alternate_names", f"${tacredConfig.clustersPath}/1.org_alternate_names"),
      ("org:city_of_headquarters", f"${tacredConfig.clustersPath}/2.org_city_of_headquarters"),
      ("org:country_of_headquarters", f"${tacredConfig.clustersPath}/3.org_country_of_headquarters"),
      ("org:dissolved", f"${tacredConfig.clustersPath}/4.org_dissolved"),
      ("org:founded", f"${tacredConfig.clustersPath}/5.org_founded"),
      ("org:founded_by", f"${tacredConfig.clustersPath}/6.org_founded_by"),
      ("org:member_of", f"${tacredConfig.clustersPath}/7.org_member_of"),
      ("org:members", f"${tacredConfig.clustersPath}/8.org_members"),
      ("org:number_of_employees/members", f"${tacredConfig.clustersPath}/9.org_number_of_employees_members"),
      ("org:parents", f"${tacredConfig.clustersPath}/10.org_parents"),
      ("org:political/religious_affiliation", f"${tacredConfig.clustersPath}/11.org_political_religious_affiliation"),
      ("org:shareholders", f"${tacredConfig.clustersPath}/12.org_shareholders"),
      ("org:stateorprovince_of_headquarters", f"${tacredConfig.clustersPath}/13.org_stateorprovince_of_headquarters"),
      ("org:subsidiaries", f"${tacredConfig.clustersPath}/14.org_subsidiaries"),
      ("org:top_members/employees", f"${tacredConfig.clustersPath}/15.org_top_members_employees"),
      ("org:website", f"${tacredConfig.clustersPath}/16.org_website"),
      ("per:age", f"${tacredConfig.clustersPath}/17.per_age"),
      ("per:alternate_names", f"${tacredConfig.clustersPath}/18.per_alternate_names"),
      ("per:cause_of_death", f"${tacredConfig.clustersPath}/19.per_cause_of_death"),
      ("per:charges", f"${tacredConfig.clustersPath}/20.per_charges"),
      ("per:children", f"${tacredConfig.clustersPath}/21.per_children"),
      ("per:cities_of_residence", f"${tacredConfig.clustersPath}/22.per_cities_of_residence"),
      ("per:city_of_birth", f"${tacredConfig.clustersPath}/23.per_city_of_birth"),
      ("per:city_of_death", f"${tacredConfig.clustersPath}/24.per_city_of_death"),
      ("per:countries_of_residence", f"${tacredConfig.clustersPath}/25.per_countries_of_residence"),
      ("per:country_of_birth", f"${tacredConfig.clustersPath}/26.per_country_of_birth"),
      ("per:country_of_death", f"${tacredConfig.clustersPath}/27.per_country_of_death"),
      ("per:date_of_birth", f"${tacredConfig.clustersPath}/28.per_date_of_birth"),
      ("per:date_of_death", f"${tacredConfig.clustersPath}/29.per_date_of_death"),
      ("per:employee_of", f"${tacredConfig.clustersPath}/30.per_employee_of"),
      ("per:origin", f"${tacredConfig.clustersPath}/31.per_origin"),
      ("per:other_family", f"${tacredConfig.clustersPath}/32.per_other_family"),
      ("per:parents", f"${tacredConfig.clustersPath}/33.per_parents"),
      ("per:religion", f"${tacredConfig.clustersPath}/34.per_religion"),
      ("per:schools_attended", f"${tacredConfig.clustersPath}/35.per_schools_attended"),
      ("per:siblings", f"${tacredConfig.clustersPath}/36.per_siblings"),
      ("per:spouse", f"${tacredConfig.clustersPath}/37.per_spouse"),
      ("per:stateorprovince_of_birth", f"${tacredConfig.clustersPath}/38.per_stateorprovince_of_birth"),
      ("per:stateorprovince_of_death", f"${tacredConfig.clustersPath}/39.per_stateorprovince_of_death"),
      ("per:stateorprovinces_of_residence", f"${tacredConfig.clustersPath}/40.per_stateorprovinces_of_residence"),
      ("per:title", f"${tacredConfig.clustersPath}/41.per_title"),
    ).map(it => (it._1, new File(it._2)))
    
    // Here we save the results
    val basepath = tacredConfig.ruleBasepath
    val maxSteps = tacredConfig.maxSteps
    
    logger.info("Read the clusters")
    // val done = new File("/home/rvacareanu/projects/results/odinsynth_tacred_generated_rules/bert_models/4_512/aggregated/").listFiles().map(_.getAbsolutePath).toList
                                                // .map { it => it.split("/").last }
                                                // .map { it => it.replace("tacred_results_", "") }
                                                // .map { it => it.replace("___", "/") }

    // (relation, relation_path, cluster_path, pandas_dataset)
    val clusters = relationNames.map { it => (it._1, it._2, it._2.listFiles().filter(_.isFile()).sorted) } // Seq[(String, File, Array[File])]
                                .map { it => (it._1, it._2, it._3.map { f => (f, ClusterDataset(f.getAbsolutePath())) }) } // Seq[(String, File, Array[(File, PandasLikeDataset)])]
                                .flatMap { it => it._3.map { pld => (it._1, it._2, pld._1, pld._2) } } // Seq[(String, File, File, PandasLikeDataset)]
                                .let { it => // Sharding-like implementation
                                  it
                                  // .filter { c => !done.exists { it => c._3.getAbsolutePath().contains(it) } }
                                  .drop(tacredConfig.skipClusters.getOrElse(0))       // Drop a specified number of clusters
                                  .take(tacredConfig.takeClusters.getOrElse(it.size)) // Take a specified number of clusters or all
                                } 
                                // .filter { it => it._3.getAbsolutePath.contains("2.org_city_of_headquarters/cluster_r0_1_2_4_2_2") ||
                                // it._3.getAbsolutePath.contains("40.per_stateorprovinces_of_residence/cluster_r0_1_1_5_7_1") ||
                                // it._3.getAbsolutePath.contains("1.org_alternate_names/cluster_r0_1_1_2_4_3") ||
                                // it._3.getAbsolutePath.contains("2.org_city_of_headquarters/cluster_r1_1_2_1_1") ||
                                // it._3.getAbsolutePath.contains("2.org_city_of_headquarters/cluster_r1_1_2_1_1") ||
                                // it._3.getAbsolutePath.contains("18.per_alternate_names/cluster_r1_1_2_3_3_2") ||
                                // it._3.getAbsolutePath.contains("3.org_country_of_headquarters/cluster_r1_1_2_1_1") ||
                                // it._3.getAbsolutePath.contains("22.per_cities_of_residence/cluster_r1_2_2_6_2_1") ||
                                // it._3.getAbsolutePath.contains("30.per_employee_of/cluster_r0_1_3_5_5_4_6") ||
                                // it._3.getAbsolutePath.contains("25.per_countries_of_residence/cluster_r0_1_1_1_1_5_2") ||
                                // it._3.getAbsolutePath.contains("41.per_title/cluster_r1_1_4_5_6_1_4") ||
                                // it._3.getAbsolutePath.contains("2.org_city_of_headquarters/cluster_r0_1_1_1_3_5") ||
                                // it._3.getAbsolutePath.contains("7.org_member_of/cluster_r1_1_1_2_1_7") ||
                                // it._3.getAbsolutePath.contains("20.per_charges/cluster_r0_1_1_1_4_1") ||
                                // it._3.getAbsolutePath.contains("24.per_city_of_death/cluster_r0_1_2_1_3_5") ||
                                // it._3.getAbsolutePath.contains("31.per_origin/cluster_r1_1_2_2_3_2") ||
                                // it._3.getAbsolutePath.contains("41.per_title/cluster_r0_1_2_5_9_6") ||
                                // it._3.getAbsolutePath.contains("36.per_siblings/cluster_r1_1_1_1_8") ||
                                // it._3.getAbsolutePath.contains("10.org_parents/cluster_r1_1_1_2_4_2") ||
                                // it._3.getAbsolutePath.contains("39.per_stateorprovince_of_death/cluster_r0_1_1_2_1_2_5") ||
                                // it._3.getAbsolutePath.contains("3.org_country_of_headquarters/cluster_r1_1_2_2_6_2") ||
                                // it._3.getAbsolutePath.contains("13.org_stateorprovince_of_headquarters/cluster_r0_1_1_1_1") ||
                                // it._3.getAbsolutePath.contains("14.org_subsidiaries/cluster_r1_1_1_1_3_1") ||
                                // it._3.getAbsolutePath.contains("14.org_subsidiaries/cluster_r1_1_2_2_4_4") ||
                                // it._3.getAbsolutePath.contains("30.per_employee_of/cluster_r1_1_1_3") ||
                                // it._3.getAbsolutePath.contains("30.per_employee_of/cluster_r1_1_2_2_4_1") ||
                                // it._3.getAbsolutePath.contains("32.per_other_family/cluster_r0_1_1_1") ||
                                // it._3.getAbsolutePath.contains("32.per_other_family/cluster_r1_1_1_1") ||
                                // it._3.getAbsolutePath.contains("40.per_stateorprovinces_of_residence/cluster_r1_1_2_2_7_3") ||
                                // it._3.getAbsolutePath.contains("41.per_title/cluster_r1_1_5_2_3") }
                                
    println(clusters.size)

    logger.info("Add an associated ExtractorEngine to each cluster, to be used for checking for matches that should not happen")
    // Make an EE for each cluster. Will be used to get negative examples
    val clustersWithEe = clusters.map { case (relation, relationPath, clusterPath, cluster) =>
      val sentences = cluster.getData._1.map(_.toSeq).toSeq
      val doc       = DocumentFromSentences.documentFromSentencesAndCache(sentences, p, tacredConfig.odinsonDocumentCache) //ProcessorsUtils.convertDocument(p.annotateFromTokens(sentences.map(_.map(_.toLowerCase()))))
      (relation, relationPath, clusterPath, cluster, ExtractorEngine.inMemory(doc))
    }
    println(f"Clusters size ${clusters.size} ${clusters.take(2).map(_._3.getAbsolutePath())}")
    println(f"Try to solve each cluster (time: ${ju.Calendar.getInstance().getTime()})")
    val result = clusters.map { case (relation, relationPath, clusterPath, cluster) =>
      val allSolutions = mutable.ListBuffer.empty[TacredSolution]
      var tried = 0
      // println("-"*100)
      logger.info(f"Evaluation for ${clusterPath.getAbsolutePath()} ($relation)")
      println(f"Evaluation for ${clusterPath.getAbsolutePath()} ($relation)")
      val outputFilePath = f"$basepath/tacred_results_" + relationPath.getAbsolutePath().split("/").last + "___" + clusterPath.getAbsolutePath().split("/").last.split("\\.").head
      // val whereToCheck   = f"/home/rvacareanu/projects/results/odinsynth_tacred_generated_rules/dynamic_merged_all/tacred_results_" + relationPath.getAbsolutePath().split("/").last + "___" + clusterPath.getAbsolutePath().split("/").last.split("\\.").head
      // var finished = Files.exists(Paths.get(outputFilePath))
      // println(f"Check $outputFilePath")
      val finished = if (Files.exists(Paths.get(outputFilePath))) {
        // Check if it was finished
        using(Source.fromFile(outputFilePath)) { it =>
          val lines = it.getLines().toSeq
          if(lines.size < 1 || lines.init.last != "The final solution is:") {
            false
          } else if(!TacredSolution.checkValidString(lines.last)) {
            false
          } else if(TacredSolution.fromString(lines.last).solutionStatus == SolutionFailed) {
            false
          } else {
            true
          }
        }
      } else {
        false
      }
      if (finished) {
        println(f"Already done for $outputFilePath (${Files.exists(Paths.get(outputFilePath))})")
        extractSolutionsFromFile(outputFilePath)
      } else {
        println(f"Write to $outputFilePath ($outputFilePath ${Files.exists(Paths.get(outputFilePath))}) (${Files.exists(Paths.get(outputFilePath))})")
        val outputFile = new PrintWriter(new File(outputFilePath))
        val wronglyMatched   = mutable.ListBuffer.empty[Seq[String]] // We will hold here the sentences (current and previous) that were wrongly matched

        val firstType  = Parser.parseBasicQuery(cluster.getFirstObjects.map(_.toLowerCase()).map(it => f"word=$it").mkString(start="[", sep=" | ", end = "]"))
        val secondType = Parser.parseBasicQuery(cluster.getSecondObjects.map(_.toLowerCase()).map(it => f"word=$it").mkString(start="[", sep=" | ", end = "]"))
        val imposeQueryStructure = Some((firstType, secondType))
        // wronglyMatched ++= getRandomNegativeSentences(clusters, relation, 5)
        var currentSolution  = searchForSpec(cluster, wronglyMatched.distinct.toSeq, relation, outputFile, clusterPath, maxSteps, tried, scorer, p, tacredConfig, imposeQueryStructure)
        var solution = currentSolution
        var solutionTotalWrongMatches = Int.MaxValue

        // Keep redoing it as long as: 
        //      - we find a solution AND
        //      - the solution we find matches things outside its relation scope AND
        //      - we tried for less than 5 times
        // When we don't find a solution anymore, we return the best so far (fewest misses)
        // When we reached a solution without any misses, stop
        // When we tried for >= 5 times, stop
        while(currentSolution.solution.isDefined && solutionTotalWrongMatches > 0 && tried < 5) {
          tried += 1
          // println(f"Inside while. Currently, there are a total of ${wronglyMatched.size} sentences to be used as negative examples. The negative examples are: ")
          // wronglyMatched.foreach(println)
          // println(f"The pattern is ${currentSolution.solution.get._2}")

          // Calculate which are the sentences that the currentSolution wrongly matches
          val currentWronglyMatched = clustersWithEe.filter { it => it._1 != relation }.flatMap { case (_, _, cp, c, ee) =>
            val results = ee.query(ee.compiler.compile(currentSolution.solution.get))

            val indices = results.scoreDocs.flatMap(_.matches).map(it => (it.start, it.end))
            val docs = results.scoreDocs.map(it => (it.doc, it.matches)).flatMap { it => it._2.map(m => (it._1, m.start, m.end)) }.toList // (sentence, match_start, match_end)
              
            val clusterData = c.getData()
            val specs   = clusterData._2.map(it => (it.start, it.end))
            val wronglyMatchedSentences = indices.flatMap { m =>
              if (specs.contains(m)) {
                Some(clusterData._1(clusterData._2(specs.indexOf(m)).sentId))
              } else {
                None
              }
            }
            
            wronglyMatchedSentences
          }
          
          allSolutions.append(currentSolution.copy(wrongMatches = Some(currentWronglyMatched.size)))
          // println(f"With ${currentWronglyMatched.size} wrong matches. ($solutionTotalWrongMatches)")
          
          // Add parts of the wrongly matched sentences. Nothing is added if nothing wrong or nothing new was matched
          wronglyMatched ++= r.shuffle(currentWronglyMatched.diff(wronglyMatched)).take(cluster.length() * 5)

          if (currentWronglyMatched.size == 0 || currentWronglyMatched.size < solutionTotalWrongMatches) { // If we matched 0 wrong sentences
            // or we wrongly matched fewer than with the previous solution, update
            
            // println(f"Update the solution, as ${currentSolution.solution.get._2} got ${currentWronglyMatched.size}, while ${solution.solution.get._2} got ${solutionTotalWrongMatches}")
            solution = currentSolution.copy(wrongMatches = Some(currentWronglyMatched.size))
            solutionTotalWrongMatches = currentWronglyMatched.size
          }
          
          if(currentWronglyMatched.size > 0) {
            // println("Seach again")
            // Search again
            currentSolution = searchForSpec(cluster, wronglyMatched.distinct.toSeq, relation, outputFile, clusterPath, maxSteps, tried, scorer, p, tacredConfig, imposeQueryStructure)
          }

        }


        outputFile.println("We found the following solutions:")
        allSolutions.foreach(it => outputFile.println(it.getString()))
        outputFile.println("The final solution is:")
        // println("The final solution is:")
        outputFile.println(solution.getString())
        // println(solution)
        // println(solution.getString())
        // println(solution.wrongMatches)
        // println("-"*100)
        // println("\n")
        outputFile.close()
        (solution, allSolutions)
      }    
    }
    println(f"Finished trying to solve each cluster (time: ${ju.Calendar.getInstance().getTime()})")
    val (sol, allSol) = result.toList.let { it =>
      (it.map(_._1), it.flatMap(_._2))
    }

    using(new PrintWriter(new File(f"${basepath}/all_solutions.tsv"))) { pw =>
      pw.println(TacredSolution.getHeaderWithoutIndex())
      sol.foreach { s =>
        pw.println(s.getString())
      }
    }
    using(new PrintWriter(new File(f"${basepath}/all_solutions_all_trials.tsv"))) { pw =>
      pw.println(TacredSolution.getHeaderWithoutIndex())
      allSol.foreach { s =>
        pw.println(s.getString())
      }
    }
    using(new PrintWriter(new File(f"${basepath}/version.txt"))) { pw =>
      pw.println(scorer.version)
      pw.println(tacredConfig.toString())
    }


  }

  /**
   * 
    * Searches for a solution. Returns a TacredSolution, potentially with Some(pattern) (if it was found), or
    * None (otherwise) + the TacredSolution associated data
    * 
    * @param cluster          - the cluster for which to search for a rule (built with @see python/tacred_data_prep.py)
    * @param negativeExamples - the resulting pattern should not match anything from this sequence of sentences (Seq[Seq[String]])
    * @param relation         - the type of relation (String)
    * @param outputFile       - a PrintWriter used for logging
    * @param clusterPath      - path to this cluster (@see cluster)
    * @param steps            - the maximum number of steps to search for a rule. None means there is no limit
    * @param trialCount       - how many times was this method called for this particular cluster. 
    *                           Used when creating the solution (@see org.clulab.odinsynth.tacred.TacredSolution) as a parameter
    *                           to keep track of the "version" of this rule
    * @param s                - a scorer (@see org.clulab.odinsynth.scorer.Scorer)
    * @return a @see org.clulab.odinsynth.tacred.TacredSolution, potentially containing a rule (if succeeded, otherwise None). It also contains
    *         various additional information about this cluster (such as: number of sentences, number of highlighted tokens, etc)
    */
  def searchForSpec(cluster: ClusterDataset, 
                    negativeExamples: Seq[Seq[String]], 
                    relation: String, 
                    outputFile: PrintWriter, 
                    clusterPath: File,
                    steps: Option[Int],
                    trialCount: Int,
                    s: Scorer,
                    p: FastNLPProcessor,
                    tacredConfig: TacredConfig,
                    firstSecondType: Option[(Query, Query)],
                    ): TacredSolution = {

    
    val data             = cluster.getData    
    val clusterSentences = data._1.map(_.toSeq).toSeq

    val maxSentId = data._2.maxBy(_.sentId).sentId
    val sentences = clusterSentences ++ negativeExamples
    // Do not save to cache, as we are randomly adding negative examples. Therefore, for random seeds we will likely get a lot of misses
    val doc       = synchronized { DocumentFromSentences.documentFromSentencesAndCache(sentences, p, tacredConfig.odinsonDocumentCache, false) } // ProcessorsUtils.convertDocument(p.annotateFromTokens(sentences.map(_.map(_.toLowerCase()))))
    val specs     = data._2
                        .map { it => it.copy(docId = doc.id) }
                        .toSet// ++ (0 until negs.size).map { idx => Spec(doc.id, idx + maxSent, -1, -1, Set()) }   // Write the proper doc id
    
    // Which fields to use
    val fieldNames = Set("word", "tag", "lemma")
    val searcher = if (firstSecondType.isDefined) {
      new TacredSearcher(Seq(doc), specs, fieldNames, steps, Some(outputFile), s, withReward = tacredConfig.useReward, firstSecondType.get._1, firstSecondType.get._2)
    } else {
      new Searcher(doc, specs, fieldNames, steps, Some(outputFile), s, withReward = tacredConfig.useReward)
    }
    // val searcher = new TacredSearcher(Seq(doc), specs, fieldNames, steps, Some(outputFile), s, withReward = tacredConfig.useReward, HoleQuery, HoleQuery)

    // Used in the solution, as metadata
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
    

    // Handle the potentially failing operation
    // Empirically observed that some patterns are causing an Unimplemented error from Odinson
    val q = Try(searcher.findFirst())

    val solutionStatus: SolutionStatus = if (q.isSuccess) {
      outputFile.println("SolutionStatus succeeded")
      SolutionSucceeded
    } else {
      println(f"SolutionStatus failed for ${clusterPath.getAbsolutePath()}")
      outputFile.println("SolutionStatus failed")
      SolutionFailed
    }

    
    // outputFile.println(f"Trying: ${clusterPath} (${specs.size}) with (${maxSentenceLength} ${minSentenceLength}). ")
    sentences.foreach(outputFile.println)
    specs.foreach(outputFile.println)
    
    // println(f"Trying: ${clusterPath} (${specs.size}) with (${maxSentenceLength} ${minSentenceLength}). ")

    val solution = if (q.isSuccess && q.get.isDefined) {

      TacredSolution(
                    numberOfSentences = specs.size,
                    maxHighlightedLength = maxSentenceLength,
                    minHighlightedLength = minSentenceLength,
                    clusterPath = clusterPath.getAbsolutePath(),
                    numberOfSteps = searcher.numberOfSteps(),
                    solution = Some(q.get.get.rule.pattern),
                    wrongMatches = None,
                    trials = trialCount,
                    direction = directionality,
                    relation = relation,
                    scorerVersion = s.version,
                    solutionStatus = solutionStatus)
    } else {
      TacredSolution(
                    numberOfSentences = specs.size,
                    maxHighlightedLength = maxSentenceLength,
                    minHighlightedLength = minSentenceLength,
                    clusterPath = clusterPath.getAbsolutePath(),
                    numberOfSteps = searcher.numberOfSteps(),
                    solution = None,
                    wrongMatches = None,
                    trials = trialCount,
                    direction = directionality,
                    relation = relation,
                    scorerVersion = s.version,
                    solutionStatus = solutionStatus)
    }

    solution
  }

  def getRandomNegativeSentences(clusters: Traversable[(String, File, File, PandasLikeDataset)], currentRelation: String, howMany: Int, r: Random): Seq[Seq[String]] = {
    // clusters.foreach { it => assert(it.lines.size != 0) }
    val shuffled = r.shuffle(
      clusters.filter { it => it._1 != currentRelation } // Keep only clusters that are of different relation
              .map(_._4) // Map to PandasLikeDataset
              .flatMap { it => 
                val data = it.getData()
                data._1 // Get the sentences
              }.toSeq 
      )

    shuffled.take(howMany).toSeq
  }

  def extractSolutionsFromFile(path: String): (TacredSolution, Seq[TacredSolution]) = {
    using(Source.fromFile(path)) { it =>
      val solutionLines = it.getLines().dropWhile(it => it != "We found the following solutions:").toArray.tail
      val allSolutions = solutionLines.takeWhile(it => it != "The final solution is:").map(it => TacredSolution.fromString(it))
      val solution = TacredSolution.fromString(solutionLines.last)
      
      (solution, allSolutions)
    }
  }

}
