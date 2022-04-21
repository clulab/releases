package org.clulab.odinsynth.evaluation.tacred

import org.clulab.odinsynth.scorer.Scorer
import org.clulab.odinsynth.scorer.DynamicWeightScorer
import org.clulab.odinsynth.scorer.StaticWeightScorer
import com.typesafe.config.Config
import com.typesafe.config.ConfigFactory
import collection.JavaConversions._

/**
 * 
 * Uses a configuration file to initialize i
 * Holds information particular to the TACRED dataset, such as (but not limited to):
 *          - relations
 *          - some paths
 * 
 * Mainly used by @see TacredEvaluation or @see TacredRuleGeneration
 * 
 */
class TacredConfig(config: Config) {
  // Sequence of (String, Int), where String is the relation type and Int is the index
  val allRelations = Seq(
      ("no_relation", 0), 
      ("org:alternate_names", 1), 
      ("org:city_of_headquarters", 2), 
      ("org:country_of_headquarters", 3), 
      ("org:dissolved", 4), 
      ("org:founded", 5), 
      ("org:founded_by", 6), 
      ("org:member_of", 7), 
      ("org:members", 8), 
      ("org:number_of_employees/members", 9), 
      ("org:parents", 10), 
      ("org:political/religious_affiliation", 11), 
      ("org:shareholders", 12), 
      ("org:stateorprovince_of_headquarters", 13), 
      ("org:subsidiaries", 14), 
      ("org:top_members/employees", 15), 
      ("org:website", 16), 
      ("per:age", 17), 
      ("per:alternate_names", 18), 
      ("per:cause_of_death", 19), 
      ("per:charges", 20), 
      ("per:children", 21), 
      ("per:cities_of_residence", 22), 
      ("per:city_of_birth", 23), 
      ("per:city_of_death", 24), 
      ("per:countries_of_residence", 25), 
      ("per:country_of_birth", 26), 
      ("per:country_of_death", 27), 
      ("per:date_of_birth", 28), 
      ("per:date_of_death", 29), 
      ("per:employee_of", 30), 
      ("per:origin", 31), 
      ("per:other_family", 32), 
      ("per:parents", 33), 
      ("per:religion", 34), 
      ("per:schools_attended", 35), 
      ("per:siblings", 36), 
      ("per:spouse", 37), 
      ("per:stateorprovince_of_birth", 38), 
      ("per:stateorprovince_of_death", 39), 
      ("per:stateorprovinces_of_residence", 40), 
      ("per:title", 41)
  )

  def trainProcessed: String = config.getString("odinsynth.evaluation.tacred.trainProcessed")
  def devProcessed: String   = config.getString("odinsynth.evaluation.tacred.devProcessed")

  // The generated rules are saved in this folder
  def ruleBasepath: String = config.getString("odinsynth.evaluation.tacred.ruleBasepath")

  // Where are the clusters for every relation
  // The clusters are created from the original data using tacred_data_prep.py
  def clustersPath = config.getString("odinsynth.evaluation.tacred.clusterPath")

  // Where to cache the documents
  val odinsonDocumentCache = config.getString("odinsynth.evaluation.tacred.odinsonDocumentCache")

  // Which type of scorer to use
  def scorer: Scorer = {
    val scorerType = config.getString("odinsynth.evaluation.tacred.scorer")
    scorerType match {
      case "StaticWeightScorer"  => StaticWeightScorer()
      case "DynamicWeightScorer" => DynamicWeightScorer(this.apiEndpoint)
      case _                     => throw new IllegalArgumentException("The type should be one of {\"StaticWeightScorer\", \"DynamicWeightScorer\"}")
    }
  }

  def maxSteps: Option[Int] = {
    if(config.hasPath("odinsynth.evaluation.tacred.steps")) {
      Some(config.getInt("odinsynth.evaluation.tacred.steps"))
    } else {
      None
    }
  }

  def relationsToUse: Seq[Int] = {
    if(config.hasPath("odinsynth.evaluation.tacred.relationsToUse")) {
      config.getIntList("odinsynth.evaluation.tacred.relationsToUse").toSeq.map(_.toInt)
    } else {
      // Use all
      (0 until 42)
    }
  }

  def patternsPath: String = {
    if(config.hasPath("odinsynth.evaluation.tacred.relationsPath")) {
      config.getString("odinsynth.evaluation.tacred.relationsPath")
    } else {
      val exceptionMessage = "The path to the patterns to use for evaluations is missing. " +
        "Please set \"odinsynth.evaluation.tacred.relationsPath\" in the config or as a command line argument"
      throw new IllegalArgumentException(exceptionMessage)
    }
  }

  // The path to the endpoint to use in the case of dynamic weight scorer
  def apiEndpoint: String = {
    if(config.hasPath("odinsynth.evaluation.tacred.endpoint")) {
      val endpoint = config.getString("odinsynth.evaluation.tacred.endpoint")
      if (endpoint.last == '/') {
        endpoint.init
      } else {
        endpoint
      }
    } else {
      val exceptionMessage = "Missing endpoint"
      throw new IllegalArgumentException(exceptionMessage)
    }
  }


  // Sharding details

  // How many clusters to skip
  def skipClusters: Option[Int] = {
    if(config.hasPath("odinsynth.evaluation.tacred.skipClusters")) {
      Some(config.getInt("odinsynth.evaluation.tacred.skipClusters"))
    } else {
      None
    }  
  }

  // How many clusters to take
  def takeClusters: Option[Int] = {
    if(config.hasPath("odinsynth.evaluation.tacred.takeClusters")) {
      Some(config.getInt("odinsynth.evaluation.tacred.takeClusters"))
    } else {
      None
    }  
  }

  def useReward: Boolean = {
    if(config.hasPath("odinsynth.useReward")) {
      config.getBoolean("odinsynth.useReward")
    } else {
      false
    }
  }

  def weightingScheme: WeightingScheme = {
    if(config.hasPath("odinsynth.evaluation.tacred.weightingScheme")) {
      config.getString("odinsynth.evaluation.tacred.weightingScheme") match {
        case "equal"       => EqualWeight
        case "specsize"    => SpecSizeWeight
        case "logspecsize" => LogSpecSizeWeight
        case unknown       => throw new IllegalArgumentException(f"This weighting scheme ($unknown) is unknown")
      }
    } else {
      val exceptionMessage = "Missing weighting scheme"
      throw new IllegalArgumentException(exceptionMessage)
    }
  }

  def distinctRules: Boolean = {
    if(config.hasPath("odinsynth.evaluation.tacred.distinctRules")) {
      config.getBoolean("odinsynth.evaluation.tacred.distinctRules")
    } else {
      false
    }
  }

  override def toString(): String = f"useReward=${useReward}, clustersPath=${clustersPath}, maxSteps=${maxSteps}"

}
object TacredConfig {
  def from(resource: String): TacredConfig = new TacredConfig(ConfigFactory.load(resource))
}
