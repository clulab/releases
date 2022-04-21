package org.clulab.odinsynth.evaluation.tacred

import scala.io.StdIn.readLine
import scala.util.control.Breaks._
import ai.lum.odinson._
import org.clulab.processors.clu.CluProcessor
import _root_.org.clulab.processors.{Document => CluDocument}
import org.clulab.processors.fastnlp.FastNLPProcessor
import scala.collection.parallel.ForkJoinTaskSupport
import java.util.concurrent.ForkJoinPool

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
import org.clulab.odinsynth.EnhancedType

/**
  * Collect the rules from individual files. Does not try to generate new rules
  * 
  * sbt -J-Xmx32g 'runMain org.clulab.odinsynth.evaluation.tacred.TacredRuleGeneration dynamic_rule_generation.conf' -Dodinsynth.evaluation.tacred.clusterPath='/home/rvacareanu/projects/odinsynth_data/tacred_clusters' -Dodinsynth.evaluation.tacred.skipClusters=0 -Dodinsynth.evaluation.tacred.takeClusters=20 -Dodinsynth.evaluation.tacred.ruleBasepath=/home/rvacareanu/projects/results/odinsynth_tacred_generated_rules/bert_models/normal/bert_base/aggregated//s0 -Dodinsynth.evaluation.tacred.endpoint='http://localhost:8000' -Dodinsynth.useReward=false
  * 
  */
object TacredSolutionCollector extends App {
  val tacredConfig = TacredConfig.from(args.head)
  
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
  // println(basepath)
  // println((new File(basepath)).listFiles().size)
  // println((new File(basepath)).listFiles().take(5).toList.map(_.getAbsolutePath()))
  // System.exit(1)
  val result = (new File(basepath)).listFiles().sortBy(_.getAbsolutePath()).filterNot { file => file.getAbsolutePath().contains("version.txt") }.filter { file =>
    using(Source.fromFile(file)) { it =>
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
  }.let { it => println(it.size); it}.map { file => TacredRuleGeneration.extractSolutionsFromFile(file.getAbsolutePath) }
  
  println(f"Results collected ${result.size} ${result.flatMap(_._2).size}")

  val (sol, allSol) = result.toList.let { it =>
    (it.map(_._1), it.flatMap(_._2))
  }
  println(f"Found a total of ${sol.filter(_.solution.isDefined).size}")

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

}