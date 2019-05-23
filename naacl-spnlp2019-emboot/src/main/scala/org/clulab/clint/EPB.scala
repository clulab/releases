package org.clulab.clint

import java.io.{BufferedWriter, File, FileWriter}

import com.typesafe.config.ConfigFactory
import com.typesafe.scalalogging.LazyLogging
import ai.lum.common.ConfigUtils._
import ai.lum.common.StringUtils._
//import com.rockymadden.stringmetric.similarity.LevenshteinMetric
import org.clulab.embeddings.word2vec.Word2Vec
import org.clulab.learning.{LogisticRegressionClassifier, RVFDataset, RVFDatum}
import org.clulab.struct.Counter
import spray.json._
import spray.json.DefaultJsonProtocol._
import scala.collection.mutable
import scala.collection.parallel.ParSeq

import scala.collection.mutable
import scala.collection.mutable.{ArrayBuffer, HashMap, HashSet}
import scala.io._

/**
  * Created by ajaynagesh on 9/25/17.
  */

object EPB extends App with LazyLogging{
  // ----------------------------------------------------------------------------------
  // Parameter Initialization
  // ----------------------------------------------------------------------------------
  val config = ConfigFactory.load()

  val indexDir = config[File]("clint.index-dir") // Directory which contains the index of entities and patterns
  val seedsFile:String = config[String]("clint.seedsFile") // Reading seeds from a file whose location is given by the config param: clint.seedsFile
  val numEpochs = config[Int]("clint.numEpochs")
  val w2vVectorsFile = config[String]("clint.w2vVectors")
  val numPatternsToSelect = config[Int]("clint.numPatternsToSelect")
  val outputFile = config[String]("clint.outputFile")
  val promoteGlobal = config[Boolean]("clint.promoteGlobal")
  val classifierTopNpredictions = if(promoteGlobal) 0 else config[Int]("clint.classifierTopNpredictions")
  val promoteGlobalPercent = if(promoteGlobal) {
    config[Double]("clint.promoteGlobalPercent")
  } else {
    -1.0
  }
  val goldLabelsFile:File =  config[File]("clint.goldLabelsFile")
  val featuresString:String = config[String]("clint.features")
  //val pmiTypeforPatPromotion:String = config[String]("clint.pmi")
  // ----------------------------------------------------------------------------------

  // ----------------------------------------------------------------------------------
  // Loading the indices
  // ----------------------------------------------------------------------------------
  logger.info("loading data")
  val wordLexicon = LexiconBuilder.loadLexicon(new File(indexDir, "word.lexicon"))
  val entityLexicon = LexiconBuilder.loadLexicon(new File(indexDir, "entity.lexicon.filtered"))
  val patternLexicon = LexiconBuilder.loadLexicon(new File(indexDir, "patterns.lexicon.filtered"))
  val entityToPatterns = Index.loadFrom(new File(indexDir, "entityToPatterns.index.filtered"))
  val patternToEntities = Index.loadFrom(new File(indexDir, "patternToEntities.index.filtered"))
  val entityCounts = Counts.loadFrom(new File(indexDir, "entity.counts.filtered"))
  val patternCounts = Counts.loadFrom(new File(indexDir, "patterns.counts.filtered"))
  val entityPatternCount = Counts2.loadFrom(new File(indexDir, "entityId.patternId.counts.filtered"))
  val totalEntityCount = entityCounts.counts.values.sum
  val totalPatternCount = patternCounts.counts.values.sum
  val totalEntityPatternCount = entityPatternCount.counts.values.sum
  // ----------------------------------------------------------------------------------

  // ----------------------------------------------------------------------------------
  // MISC Initialization
  // ----------------------------------------------------------------------------------
  val featuresForEntClassifier = featuresString.split(",")
  val outputWriter = new BufferedWriter(new FileWriter(new File(outputFile)))
  val w2v = new Word2Vec(w2vVectorsFile, None)
  val sep = "-" * 70
  val goldLabels = Source.fromFile(goldLabelsFile).getLines.map {l =>
    val tmp = l.split("\t")
    val k = tmp(0)
    val v = tmp.tail
    k -> v }.toMap

  // For debug ------
  val embootdata = Source.fromFile(indexDir.toString + "/training_data_with_labels_emboot.txt").getLines.toArray.map{ x =>
    val fields = x.split("\t")
    (fields(0),fields(1))
  }
  // entity -> [(lbl,freq) ...]
  val entity_labels = embootdata.map(x => (x,1)).groupBy(_._1).map(x => (x._1,x._2.size)).toArray.map(x => (x._1._1,x._1._2,x._2)).groupBy(_._2).map(x => (x._1,x._2.map(y => (y._1,y._3))  ))
  // ----------------------------------------------------------------------------------
  // FEATURE NAMES:
  // ----------------------------------------------------------------------------------
  val ed_feat_name = "ed"
  val ed_global_feat_name = "ed-global"
  val pmi_feat_name = "pmi"
  val pmi_global_feat_name = "pmi-global"
  val embed_w2v_feat_name = "embed-w2v"
  val embed_w2v_global_feat_name = "embed-w2v-global"
  val semantic_drift_w2v_feat_name = "drift-w2v"
  // ----------------------------------------------------------------------------------
  var activeSemDrift = false
  // ----------------------------------------------------------------------------------
  // PRINT PARAMETERS:
  println(s"INDEX DIR: ${indexDir}")
  println(s"SEEDS FILE: ${seedsFile}")
  println(s"NUMBER OF EPOCHS: ${numEpochs}")
  println(s"GIGAWORD VECTORS FILE: ${w2vVectorsFile}")
  println(s"BOOTSTRAPPING OUTPUT IN: ${outputFile}")
  println(s"NUMBER OF PATTERNS SELECTED IN EPOCH: ${numPatternsToSelect}")
  println(s"FEATURES USED IN THE ENTITY CLASSIFIER: [${featuresForEntClassifier.mkString(", ")}]")
  if(promoteGlobal){
    logger.info("GLOBAL PROMOTION OF ENTITIES")

  }
  else {
    logger.info("PROMOTION OF ENTITIES PER CATEGORY")
    println(s"NUMBER OF ENTITIES PROMOTED: ${classifierTopNpredictions}")
  }
  //println(s"PMI TYPE USED FOR PATTERN PROMOTION ${pmiTypeforPatPromotion}")
  // ----------------------------------------------------------------------------------

  // ----------------------------------------------------------------------------------
  // Seed Initialization
  // ----------------------------------------------------------------------------------

  // maps from category name to (entity or pattern) ids
  val promotedEntities = HashMap.empty[String, HashSet[Int]]
  val promotedPatterns = HashMap.empty[String, HashSet[Int]]

  val promotedEntitiesWithEpoch = HashMap.empty[String, HashSet[(Int,Int,Double)]] // (entity_id, epoch_id, score)
  val num_ents_close_to_seeds = 10
  val num_ents_close_to_most_recent = 20

  logger.info("promoting seeds")
  val seeds:Map[String,Seq[String]] = Source.fromFile(seedsFile).getLines().mkString("\n").parseJson.convertTo[Map[String,Seq[String]]]
  val categories = seeds.keys.toArray   // an array with all category names
  for {
    cat <- categories
    ent <- seeds(cat)
    id <- entityLexicon.get(ent)
  } {
    promotedEntities.getOrElseUpdate(cat, HashSet.empty[Int]) += id
    promotedEntitiesWithEpoch.getOrElseUpdate(cat, HashSet.empty[(Int,Int,Double)]) += ((id,0,0.0))
  }


  for (cat <- categories) promotedPatterns(cat) = HashSet.empty[Int]

  printSeeds(outputWriter)
  // ----------------------------------------------------------------------------------

  // Set of Candidate Entities (for CoNLL we know the candidate entities from the dataset) (else we might have to run some heuristics such as all NPs to find the set of candidate entities)
  val candidateEntities = entityLexicon.inverseLexicon.values.toSet

  var numEntitiesRemaining = candidateEntities.size

  // BOOTSTRAPPING
  // ----------------------------------------------------------------------------------
  var epoch = 1
  //  for (epoch <- 1 to numEpochs) {
  while(numEntitiesRemaining > 0) {

    //1. Find candidate entities
    //   NOTE: Use all the entities present in the training dataset (minus the entities in the pool) as candidate entities
    // ----------------------------------------------------------------------------------
    val newCandidateEntities: Set[Int] =  candidateEntities -- promotedEntities.toSet.unzip._2.flatten
    numEntitiesRemaining = newCandidateEntities.size
    // ----------------------------------------------------------------------------------

    logger.info(s"${newCandidateEntities.size} candidate entities present in epoch ${epoch}")

    for (cat <- categories) yield {
      //2. Find candidate patterns
      //   Find TopK patterns (using one of the many scoring metrics)
      // ----------------------------------------------------------------------------------
      val candidatePatterns = getPatterns(promotedEntities(cat).toSet) // find the patterns corresponding to all the entities in the pool, using the index
      val preselectedPatterns = preselectPatterns(cat, candidatePatterns) //pre-select patterns only if they co-occur with more than one entity in the pool
      val patternsWithScores = scorePatterns(cat, preselectedPatterns) //Compute PMI based scores for each of the pre-selected patterns (also drop patterns which are overlapping and that are already present in the pool)
      val selectedPatterns = patternsWithScores.sortBy(- _._1).map(_._2).take(numPatternsToSelect) // sort in decreasing order of scores and select the top patterns
      promotedPatterns(cat) ++= selectedPatterns // Add the newly selected patterns to the pool

      logger.info(s"No of candidate patterns for ${cat}: ${candidatePatterns.size}")
      logger.info(s"No of pre-selected patterns for ${cat} (used): ${preselectedPatterns.size}")
      logger.info(s"No of patterns for ${cat} after removing overlapping patterns: ${patternsWithScores.size}")
      logger.info(s"${selectedPatterns.size} new patterns found in epoch ${epoch} for ${cat}")
      logger.info(sep)
      logger.info(s"\nPatterns-${epoch}-${cat}:${selectedPatterns.map ( p => patternLexicon(p).splitOnWhitespace.map {
        case "@" => "@"
        case n => wordLexicon(n.toInt)
      }.mkString(" ")).mkString(s"\nPatterns-${epoch}-${cat}:")}")
      // ----------------------------------------------------------------------------------

    }

    //NOTE: This feature is active only when `num_ents_in_pool` >= |m|+|n|  (m,n - m closest ents to seeds, n most recent ents)
    if (promotedEntities.values.toArray.map(_.size).sum >= (num_ents_close_to_most_recent + num_ents_close_to_seeds)*promotedEntities.size){
      //      println("Sem-drift features not active here")
      activeSemDrift = true
    }

    //3.  Train a multi-class classifier for all the categories.
    //    Training examples -- all the entities in the pool for every category
    // ----------------------------------------------------------------------------------

    // Create a dataset and populate with set of features
    val dataset = createTrainDataset()
    logger.info("Created the dataset")

    // Call the training API of k-class Logistic Regression classifier
    val classifier = new LogisticRegressionClassifier[String, String](bias = false)
    classifier.train(dataset)
    logger.info("Training complete!")

    val classifierWeights = classifier.getWeights(false)
    for ((label, weights) <- classifierWeights){
      logger.info(s"Classifier Train : ${label}:" )
      logger.info(s"Top 10 features:" )
      logger.info(s"${weights.sorted(true).take(10).map(x => x._1 + ":" + x._2.toString).mkString("\n")}")
      logger.info(s"Bottom 10 features:")
      logger.info(s"${weights.sorted(false).take(10).map(x => x._1 + ":" + x._2.toString).mkString("\n")}")
      logger.info(sep)
    }

    // ----------------------------------------------------------------------------------

    //4. Pick the top entities based on the output of the entity classifier
    // ----------------------------------------------------------------------------------
    // [(entity_id, predicted_label, conf_score_pred_label, all_conf_scores) (...) ...]
    val predictions = predictEntityCategories(newCandidateEntities.toArray, classifier)

    // NOTE: Choosing the top scored classifications irrespective of their absolute scores

    // To keep track of newly promoted entities
    val newlyPromotedEntities =  HashMap.empty[String, HashSet[Int]]
    for (cat <- categories) {
      newlyPromotedEntities.getOrElseUpdate(cat, HashSet.empty[Int])
    }

    if(promoteGlobal == false) { //Promote top `classifierTopNpredictions` entities in each predicted category
      val topPredictionsByCategory = predictions
        .groupBy(_._2) // groupBy the predicted category
        .map{ predByCat =>
        val pred = predByCat._1
        val topPreds = predByCat._2.sortBy(- _._3) // sort entries in decreasing order of prediction scores
          .take(classifierTopNpredictions) // select the top `classifierTopNpredictions` predictions
        (pred, topPreds)
      }

      for ((cat, sortedPreds) <- topPredictionsByCategory){
        for((entity, pred, conf, scores) <- sortedPreds){
          newlyPromotedEntities(cat).add(entity)
          promotedEntities(cat).add(entity)
          promotedEntitiesWithEpoch(cat).add((entity,epoch,conf))

          // Debug the predictions
          val datum = createDatum(entity, "~",true)
          val featsWithValues = datum.features.toArray.map(f => (f, datum.getFeatureCount(f))).sortBy(- _._2).take(10).map(x => x._1+":"+x._2)
          logger.info(s"Classifier Test : Entity ${entityLexicon(entity)}, prediction: ${pred}:, conf: ${conf}, scores: ${scores}" )
          logger.info(s"Top active features with their values: ${featsWithValues.mkString(", ")}" )
          //semDriftFeatures.incrementCount("avg_"+category+"_drift_w2v", avgSemDriftScore)
          //semDriftFeatures.incrementCount("max_"+category+"_drift_w2v", maxSemDriftScore)

          val driftFeaturesAvg = categories.map { c =>
            val f = "avg_" + c + "_drift_w2v"
            val fv = datum.getFeatureCount(f)
            (f,fv)
          }

          val driftFeaturesMax = categories.map { c =>
            val f = "max_" + c + "_drift_w2v"
            val fv = datum.getFeatureCount(f)
            (f,fv)
          }

          val driftFeatures = if( featuresForEntClassifier.contains(semantic_drift_w2v_feat_name) ) (driftFeaturesMax ++ driftFeaturesAvg).sortBy(- _._2).mkString(", ") else "NONE"
          val entityStr = entityLexicon(entity)
          val entityLbls = "[" + entity_labels(entityStr).mkString(", ") + "]"
          //          logger.info(s"DRIFT:\t${entityStr},\tGold Labels: $entityLbls,\tPrediction: ${pred},\tDrift features: $driftFeatures $driftFeaturesMax,\tEpochID: $epoch")
          println(s"DRIFT:\t${epoch}\t${entityStr}\t${entityLbls}\t${pred}\t${driftFeatures}")
          logger.info(sep)
        }
      }
    }
    else { // promote top `promoteGlobalPercent` of entities irrespective of their prediction category
      val numEntitiesToPromote =  math.round(predictions.size * promoteGlobalPercent).toInt
      val topPredictions = predictions.sortBy(- _._3) // sort predictions in decreasing order of confidence scores
        .take(numEntitiesToPromote) // select the top `promoteGlobalPercent` of the entities

      for ((entity, cat, conf, score) <- topPredictions){
        newlyPromotedEntities(cat).add(entity)
        promotedEntities(cat).add(entity)
        promotedEntitiesWithEpoch(cat).add((entity,epoch,conf))
      }
    }

    // Write the output file
    outputWriter.write(s"epoch ${epoch}\n")
    for(cat <- categories) {
      val entityIds = newlyPromotedEntities(cat)
      val entities = entityIds.map { e => entityLexicon(e) }
      outputWriter.write(s"${cat}\t${entities.mkString("\t")}\n")
    }
    outputWriter.write("\n")
    outputWriter.flush()

    logger.info(s"Number of predictions : ${predictions.size}")
    logger.info(s"Predictions histogram : ${predictions.groupBy(_._2).map(i => (i._1, i._2.size) ).mkString(" :: ")}")
    logger.info(s"Newly promoted entities histogram :  ${newlyPromotedEntities.map( i => (i._1, i._2.size)).mkString(" :: ")}")
    logger.info(s"Newly promoted entities --> ${newlyPromotedEntities.map( i => (i._1,  i._2.map( entityLexicon(_) ).mkString(", ") )).mkString(" : " )}")
    logger.info(sep)

    // ----------------------------------------------------------------------------------

    val stats = computeAccuracy
    val perLabelAccuracy = categories.zip(stats._4)
    logger.info("Accuracy")
    logger.info(sep)
    logger.info(s"${epoch}\t${stats._1}\t${stats._2}\t${stats._3}\t${perLabelAccuracy.mkString(" :: ")}")
    //bwAcc.write(s"${epoch}\t${stats._1}\t${stats._2}\t${stats._3}\n")
    logger.info(sep)
    logger.info(sep)
    epoch += 1

  }
  // END BOOTSTRAPPING
  // ----------------------------------------------------------------------------------

  outputWriter.close()

  /***
    * Methods to create the dataset for the classifier
    */

  // ----------------------------------------------------------------------------------
  // FEATURE FUNCTIONS
  // ----------------------------------------------------------------------------------

  def editDistPosFeature(candEntity: Int, category: String) : (Counter[String],Int) = {

    val feature = new Counter[String]

    val score = promotedEntities(category)
      .filterNot(_ == candEntity) // exclude the current entity (if present) in the promoted entities. Avoid filterNot as it creates a new copy (use forloop instead)
      .map { posEntId =>
      val candEntStr = entityLexicon(candEntity) // get the string
    val posEntStr = entityLexicon(posEntId)    // get the string
    val normalisedEditDist = levenshtein(candEntStr, posEntStr).toFloat / posEntStr.length  // [edit distance]/| posEntStr|
      if (normalisedEditDist < 0.2) // if score is less than threshold of 2
        1
      else
        0
    }.max

    feature.incrementCount("EDPos_"+category, score)
    (feature, score)
  }

  def editDistNegFeature(candEntity: Int, category: String) : (Counter[String],Int) = {

    val feature = new Counter[String]

    val negEntitiesSet = promotedEntities.values.flatten.toSet -- promotedEntities(category).toSet // Set of all entities that do not belong to category

    val score = negEntitiesSet
      .filterNot(_ == candEntity) // Not necessary. But keeping to be consistent.
      .map { negEntId =>
      val candEntStr = entityLexicon(candEntity) // get the string
    val negEntStr = entityLexicon(negEntId)    // get the string
    val normalisedEditDist = levenshtein(candEntStr, negEntStr).toFloat / negEntStr.length  // [edit distance]/| negEntStr|
      if (normalisedEditDist < 0.2) // if score is less than threshold of 2
        1
      else
        0
    }.max

    feature.incrementCount("EDNeg_"+category, score)
    (feature, score)
  }

  def patternPMIFeatures(entityID: Int, category: String) : (Counter[String], Double, Double) = {
    val promotedPatternForCategory = promotedPatterns(category).toSeq
    val pmiFeatures = new Counter[String]

    val pmiScores = for(patternId <- promotedPatternForCategory) yield {
      val ep_cnt = entityPatternCount.counts.getOrElse((entityID, patternId), 0)
      val e_cnt =  entityCounts.counts.getOrElse(entityID, 0)
      val p_cnt = patternCounts.counts.getOrElse(patternId, 0)

      val p_ep = ep_cnt.toDouble / totalEntityPatternCount
      val p_e = e_cnt.toDouble / totalEntityCount
      val p_p = p_cnt.toDouble / totalPatternCount

      val pmi = if(p_ep == 0) 0 else math.log( p_ep / (p_e * p_p) )

      val pattern = patternLexicon(patternId).splitOnWhitespace.map {
        case "@" => "@"
        case n => wordLexicon(n.toInt)
      }.mkString(" ")

      pmiFeatures.incrementCount(pattern+"_"+category+"_pmi", pmi) // TODO: Just for debugging purposes only. To remove this and use only pattern id.
      //      pmiFeatures.incrementCount(patternId+"_"+category+"_pmi", pmi)

      pmi
    }

    (pmiFeatures, pmiScores.max, pmiScores.sum/pmiScores.size)
  }

  def computeW2Vfeatures(candEntityId: Int, category: String) : (Counter[String], Double, Double) = {
    val embeddingFeatures = new Counter[String]

    val candEntity = entityLexicon(candEntityId)
    val entityPool = promotedEntities(category)
      .filterNot(_ == candEntityId) // Remove `candEntityId` if already present in the pool
      .map( entityLexicon(_) ).toSeq

    val sanitisedCandEntity = candEntity.split(" +").map( Word2Vec.sanitizeWord(_) )

    val simScores = for(entity <- entityPool) yield {
      val sanitisedEntity =  entity.split(" +").map( Word2Vec.sanitizeWord(_) )
      val sim = w2v.sanitizedAvgSimilarity(sanitisedCandEntity, sanitisedEntity)._1
      sim
    }

    val avgScore = simScores.sum / simScores.size
    val maxScore = simScores.max
    val minScore = simScores.min

    embeddingFeatures.incrementCount("avg_"+category+"_w2v", avgScore)
    embeddingFeatures.incrementCount("min_"+category+"_w2v", minScore)
    embeddingFeatures.incrementCount("max_"+category+"_w2v", maxScore)
    (embeddingFeatures, maxScore, avgScore)
  }

  def sigmoid(x: Double) = 1 / (1 + math.exp(- x))

  def computeSemDriftFeatures(candEntityId: Int, category: String) : (Counter[String], Double, Double) = {
    val semDriftFeatures = new Counter[String]

    if (activeSemDrift == false){
      return (semDriftFeatures, -1, -1)
    }

    val candEntity = entityLexicon(candEntityId)
    val entityPoolSorted = promotedEntitiesWithEpoch(category)
      .filterNot(_._1 == candEntityId) // Remove `candEntityId` if already present in the pool
      .map( x => (entityLexicon(x._1),x._2, x._3) ).toSeq.sortBy(x =>  (x._2, - x._3) ) // Sort by first by epoch and then in decreasing order of confidence score

    val sanitisedCandEntity = candEntity.split(" +").map( Word2Vec.sanitizeWord(_) )

    val entitiesCloseToSeeds = entityPoolSorted.take(num_ents_close_to_seeds).map(_._1)
    val entitiesMostRecent = entityPoolSorted.takeRight(num_ents_close_to_most_recent).map(_._1)

    val simScoresCloseToSeeds = for(entity <- entitiesCloseToSeeds) yield {
      val sanitisedEntity =  entity.split(" +").map( Word2Vec.sanitizeWord(_) )
      val sim = w2v.sanitizedAvgSimilarity(sanitisedCandEntity, sanitisedEntity)._1
      sim
    }
    val avgScoreCloseToSeeds = simScoresCloseToSeeds.sum / simScoresCloseToSeeds.size
    val maxScoreCloseToSeeds = simScoresCloseToSeeds.max

    val simScoresMostRecent = for(entity <- entitiesMostRecent) yield {
      val sanitisedEntity =  entity.split(" +").map( Word2Vec.sanitizeWord(_) )
      val sim = w2v.sanitizedAvgSimilarity(sanitisedCandEntity, sanitisedEntity)._1
      sim
    }
    val avgScoreMostRecent = simScoresMostRecent.sum / simScoresMostRecent.size
    val maxScoreMostRecent = simScoresMostRecent.max

    val simScores = for(entity <- entityPoolSorted.unzip3._1) yield {
      val sanitisedEntity =  entity.split(" +").map( Word2Vec.sanitizeWord(_) )
      val sim = w2v.sanitizedAvgSimilarity(sanitisedCandEntity, sanitisedEntity)._1
      sim
    }
    val avgScore = simScores.sum / simScores.size
    val maxScore = simScores.max

    val avgSemDriftScore =  sigmoid(avgScoreCloseToSeeds) /  sigmoid(avgScoreMostRecent) // compute the sigmoid instead of just the ratio
    val maxSemDriftScore =  sigmoid(maxScoreCloseToSeeds) /  sigmoid(maxScoreMostRecent) // compute the sigmoid instead of just the ratio

    semDriftFeatures.incrementCount("avg_"+category+"_drift_w2v", avgSemDriftScore * avgScore) //replacing drift with [ drift * similarity ]
    semDriftFeatures.incrementCount("max_"+category+"_drift_w2v", maxSemDriftScore * maxScore) //replacing drift with [ drift * similarity ]
    //    println(s"Entities in the pool : ${entityPoolSorted.mkString(", ")}")
    //    println(s"Sem-Drift Features [$candEntity : ${semDriftFeatures.toSeq.mkString(", ")}]")
    (semDriftFeatures, avgScoreCloseToSeeds, avgScoreMostRecent)
  }
  // ----------------------------------------------------------------------------------

  // ----------------------------------------------------------------------------------
  // CREATE FEATURES FOR ONE DATAPOINT
  // ----------------------------------------------------------------------------------
  def createDatum(entity: Int, category: String, isTrain: Boolean):  RVFDatum[String, String] = {
    val features = new Counter[String]

    // Note: Generate features for all the categories (not just the `category` that the `entity` belongs to )

    val edPosOverall = new mutable.HashMap[String, Double]
    val edNegOverall = new mutable.HashMap[String, Double]
    val pmiMaxOverall = new mutable.HashMap[String, Double]
    val pmiAvgOverall = new mutable.HashMap[String, Double]
    val w2vMaxOverall = new mutable.HashMap[String, Double]
    val w2vAvgOverall = new mutable.HashMap[String, Double]

    for (c <- categories) {
      val (edPosFeat,edPosScore) = editDistPosFeature(entity, c)
      val (edNegFeat,edNegScore) = editDistNegFeature(entity, c)
      val (pmiFeat,maxPmiScore,avgPmiScore) = patternPMIFeatures(entity, c)
      val (w2vFeat, maxW2vScore,avgW2vScore)= computeW2Vfeatures(entity, c)
      val (semDriftFeat,avgScoreClosetoSeeds,avgScoreMostRecent) = computeSemDriftFeatures(entity, c)
      //      if(isTrain)
      //        println(s"CREATE-DATUM: ${entityLexicon(entity)}\t${c}\t${semDriftFeat}\t${avgScoreClosetoSeeds}\t${avgScoreMostRecent}")

      // Feature selection
      // -------------------------------------------------------------
      if(featuresForEntClassifier.contains(ed_feat_name)) {
        features += edPosFeat
        features += edNegFeat
      }
      if(featuresForEntClassifier.contains(pmi_feat_name))  features += pmiFeat

      if(featuresForEntClassifier.contains(embed_w2v_feat_name)) features += w2vFeat

      if(featuresForEntClassifier.contains(semantic_drift_w2v_feat_name)) features += semDriftFeat
      // -------------------------------------------------------------

      edPosOverall  += (c -> edPosScore)
      edNegOverall  += (c -> edNegScore)
      pmiMaxOverall += (c -> maxPmiScore)
      pmiAvgOverall += (c -> avgPmiScore)
      w2vMaxOverall += (c -> maxW2vScore)
      w2vAvgOverall += (c -> avgW2vScore)

    }

    // Feature selection
    // -------------------------------------------------------------
    // NOTE: Add the global features irrespective of whether feat or its global couterpart is set
    var tmp = ("dummy", -1.0) //TODO: Hacky code. Need to correct this
    if(featuresForEntClassifier.contains(ed_feat_name) || featuresForEntClassifier.contains(ed_global_feat_name)){
      // Add the overall features (Max/Avg across all categories)
      tmp = edPosOverall.maxBy(_._2)
      features.incrementCount("EDPosMax_"+tmp._1, tmp._2)
      tmp = edNegOverall.maxBy(_._2)
      features.incrementCount("EDNegMax_"+tmp._1, tmp._2)
    }

    if(featuresForEntClassifier.contains(pmi_feat_name) || featuresForEntClassifier.contains(pmi_global_feat_name)){
      tmp = pmiMaxOverall.maxBy(_._2)
      features.incrementCount("PMI_Max_"+tmp._1, tmp._2)

      tmp = pmiAvgOverall.maxBy(_._2)
      features.incrementCount("PMI_Avg_"+tmp._1, tmp._2)
    }

    if(featuresForEntClassifier.contains(embed_w2v_feat_name) || featuresForEntClassifier.contains(embed_w2v_global_feat_name)){
      tmp = w2vMaxOverall.maxBy(_._2)
      features.incrementCount("W2V_Max_"+tmp._1, tmp._2)

      tmp = w2vAvgOverall.maxBy(_._2)
      features.incrementCount("W2V_Avg_"+tmp._1, tmp._2)
    }

    val datum =	new RVFDatum[String, String](category, features)
    return datum

  }
  // ----------------------------------------------------------------------------------

  // ----------------------------------------------------------------------------------
  // Training Dataset creation
  // Input: PromotedEntities in every category, PromotedPatterns in every category (from global variables)
  // Output: RVFDataset
  // ----------------------------------------------------------------------------------
  def createTrainDataset(): RVFDataset[String, String] = {

    val dataset = new RVFDataset[String, String]()

    for {
      cat <- categories
      ent <- promotedEntities(cat)
    }{
      val datum = createDatum(ent, cat, true)
      //      println(s"TRAIN: ${entityLexicon(ent)} DRIFT: ${datum.features.filter(_.endsWith("_drift_w2v")).map(f=> (f,datum.getFeatureCount(f))).toArray.sortBy(- _._2).mkString(" , ")}")
      dataset += datum
    }

    dataset
  }
  // ----------------------------------------------------------------------------------

  // ----------------------------------------------------------------------------------
  // Prediction method
  // Input: candidate entities, trained classifier
  // Output: [(entity_id, predicted_label, conf_score_pred_label, all_conf_scores) (...) ...] for every candidate entity
  // ----------------------------------------------------------------------------------
  def predictEntityCategories(entities: Array[Int], classifier : LogisticRegressionClassifier[String, String]) = {

    val predictions = for (entity <- entities) yield {
      val datum = createDatum(entity, "~", false)
      val predLabel = classifier.classOf(datum)
      val scores = classifier.scoresOf(datum)
      val confidence = scores.getCount(predLabel)
      (entity, predLabel, confidence, scores)
    }

    predictions
  }
  // ----------------------------------------------------------------------------------

  def computeAccuracy() : (Double, Double, Int, Array[Double]) = {
    val stats = for(cat <- categories) yield {
      val entitiesPredicted = promotedEntities(cat).map { e => entityLexicon(e) }
      val entitiesInGold = goldLabels(cat).toSet // convert to set to avoid duplicates
      val correctEntities = entitiesInGold.intersect(entitiesPredicted)

      val precision = correctEntities.size.toDouble / entitiesPredicted.size
      val recall = correctEntities.size.toDouble / entitiesInGold.size
      (precision, recall, entitiesPredicted.size)
    }

    val (precision, recall, size) = stats.unzip3
    val avgPrecision = precision.sum / precision.size
    val avgRecall = recall.sum / recall.size
    val sumSize = size.sum
    (avgPrecision, avgRecall, sumSize, precision)

  }

  /****
    *
    *
    * Methods to compute measures reading the indices and computing the various measures
    *
    */

  // number of times pattern `p` matches an entity with category `cat`
  def positiveEntityCounts(p: Int, cat: String): Double = {
    patternToEntities(p)
      .filter(promotedEntities(cat).contains) // only the ones in the pool
      .map(entityPatternCount(_, p)) // count them
      .sum // total count
  }

  def countEntityMentions(cat: String): Double = {
    promotedEntities(cat).map(entityCounts(_)).sum
  }

  def PMIpattern(p: Int, cat: String): Double = {
    val numerator = positiveEntityCounts(p, cat) // number of times pattern `p` matches an entity with category `cat`
    val denom = countEntityMentions(cat) * patternCounts(p) // total number of occurrences of an entity in `cat` * |p|
    math.log( numerator / denom  )
  }

  def modifiedPMIpattern(p: Int, cat: String): Double = {
    val numerator = positiveEntityCounts(p, cat) // number of times pattern `p` matches an entity with category `cat`
    val denom = countEntityMentions(cat) // total number of occurrences of an entity in `cat`
    math.log( numerator / denom  )
  }

  def modifiedPMIpatternLogFreqPattern(p: Int, cat: String): Double = {
    val numerator = positiveEntityCounts(p, cat) // number of times pattern `p` matches an entity with category `cat`
    val denom = countEntityMentions(cat) // total number of occurrences of an entity in `cat`
    math.log( numerator / denom  ) * math.log(patternCounts(p))
  }

  def PMIlogFreqPattern(p: Int, cat: String): Double = {
    val numerator = positiveEntityCounts(p, cat) // number of times pattern `p` matches an entity with category `cat`
    val denom = countEntityMentions(cat) * patternCounts(p) // total number of occurrences of an entity in `cat` * |p|
    math.log( numerator / denom  ) * math.log(patternCounts(p))
  }

  // Compute the scores for all patterns; Use PMI to score a pattern and a category
  def scorePatterns(category: String, patterns: Set[Int]): Seq[(Double,Int)] = {
    val scoredPatterns = for (p <- patterns) yield {
      //      if(pmiTypeforPatPromotion == "modpmi")
      //        (modifiedPMIpattern(p,category), p)
      //      else if(pmiTypeforPatPromotion == "modpmilogfreq")
      //        (modifiedPMIpatternLogFreqPattern(p,category), p)
      //      else //if(pmiTypeforPatPromotion == "pmi")
      (PMIpattern(p, category), p)
      //      else
      //      (PMIlogFreqPattern(p, category), p)
    }
    val pats = scoredPatterns.toSeq.filterNot(p => promotedPatterns(category).contains(p._2)) // remove patterns already present in the pattern pool
    takeNonOverlapping(pats, category) // remove patterns which are overlapping with already promoted patterns in `category` or patterns in `pats`
  }

  def containsOrContained(x: String, y: String): Boolean = {
    x.indexOfSlice(y) >= 0 || y.indexOfSlice(x) >= 0
  }

  def takeNonOverlapping(xs: Seq[(Double,Int)], category: String): Seq[(Double,Int)] = {
    val ids = new ArrayBuffer[(Double,Int)]
    val pats = new ArrayBuffer[String]
    var i = 0
    while (i < xs.size) {
      val x = xs(i)
      val p = patternLexicon(x._2)
      val similarPatternSeen = pats.exists(containsOrContained(_, p)) || promotedPatterns(category).map(patternLexicon.apply).exists(containsOrContained(_, p))
      if (!similarPatternSeen) {
        ids += x
        pats += p
      }
      i += 1
    }
    ids
  }

  def getPatterns(entities: Set[Int]): Set[Int] = {
    entities.flatMap(e => entityToPatterns(e))
  }

  def preselectPatterns(category: String, patterns: Set[Int]): Set[Int] = {
    val results = for (p <- patterns) yield {
      val candidates = patternToEntities(p)
      val entities = promotedEntities(category)
      val matches = candidates intersect entities
      val accuracy = matches.size.toDouble / candidates.size.toDouble
      if (matches.size <= 1) None // NOTE: Changing check from '==' I think the expected behavior is, do not select patterns if there are less than or equal to 1 overlaps
      // else if (accuracy < 0.5) None
      else Some(p)
    }
    results.flatten
  }

  def printSeeds(outputWriter: BufferedWriter): Unit = {
    println("Seeds")
    outputWriter.write("Epoch 0\n")
    for (cat <- categories) {
      println(s"$cat entities:")
      for (e <- seeds(cat)) {
        println(e)
      }
      outputWriter.write(s"${cat}\t${seeds(cat).mkString("\t")}\n")
      println()
    }
    outputWriter.write("\n")
    println("=" * 70)
    println()
  }
  def levenshtein(s1: String, s2: String): Int = {
    val memoizedCosts = mutable.Map[(Int, Int), Int]()

    def lev: ((Int, Int)) => Int = {
      case (k1, k2) =>
        memoizedCosts.getOrElseUpdate((k1, k2), (k1, k2) match {
          case (i, 0) => i
          case (0, j) => j
          case (i, j) =>
            ParSeq(1 + lev((i - 1, j)),
              1 + lev((i, j - 1)),
              lev((i - 1, j - 1))
                + (if (s1(i - 1) != s2(j - 1)) 1 else 0)).min
        })
    }

    lev((s1.length, s2.length))
  }

  /***
    *
    * Data Strcutures to load the indices
    *
    */
  //===================================
  class Index(val index: Map[Int, Set[Int]]) {
    def apply(id: Int): Set[Int] = index.getOrElse(id, Set.empty[Int])
  }

  object Index {
    def loadFrom(file: File): Index = {
      val source = Source.fromFile(file)
      val entries = (for (line <- source.getLines) yield {
        val Array(entity, patterns) = line.split("\t")
        entity.toInt -> patterns.splitOnWhitespace.map(_.toInt).toSet
      }).toList
      source.close()
      new Index(entries.toMap)
    }
  }

  class Counts(val counts: Map[Int, Int]) {
    def apply(id: Int): Double = counts.getOrElse(id, 0).toDouble
  }

  object Counts {
    def loadFrom(file: File): Counts = {
      val source = Source.fromFile(file)
      val entries = (for (line <- source.getLines) yield {
        val Array(id, count) = line.splitOnWhitespace
        id.toInt -> count.toInt
      }).toList
      source.close()
      new Counts(entries.toMap)
    }
  }

  class Counts2(val counts: Map[(Int, Int), Int]) {
    def apply(id1: Int, id2: Int): Double = counts.getOrElse((id1, id2), 0).toDouble
  }

  object Counts2 {
    def loadFrom(file: File): Counts2 = {
      val source = Source.fromFile(file)
      val entries = (for (line <- source.getLines) yield {
        val Array(id1, id2, count) = line.splitOnWhitespace
        (id1.toInt, id2.toInt) -> count.toInt
      }).toList
      source.close()
      new Counts2(entries.toMap)
    }
  }

  def readMap(file: File): Map[String, String] = {
    val source = Source.fromFile(file)
    val items = for (line <- source.getLines()) yield {
      val Array(k, v) = line.split("\t")
      (k, v)
    }
    val map = items.toMap
    source.close()
    map
  }

}

