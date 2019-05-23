//package org.clulab.clint
//
//import java.io._
//
//import scala.io.Source
//import scala.collection.mutable.{ArrayBuffer, HashMap, HashSet}
//import com.typesafe.scalalogging.LazyLogging
//import com.typesafe.config.ConfigFactory
//import ai.lum.common.ConfigUtils._
//import ai.lum.common.StringUtils._
//import org.clulab.learning.LogisticRegressionClassifier
//import org.clulab.learning.RVFDataset
//// Causes dumb errors (AZ)
//import com.rockymadden.stringmetric.similarity.LevenshteinMetric
//import org.clulab.learning.RVFDatum
//import org.clulab.struct.Counter
//import org.clulab.embeddings.word2vec.Word2Vec
//import org.apache.commons.lang3.StringEscapeUtils
//import spray.json._
//import spray.json.DefaultJsonProtocol._
//
//object BootstrapGupta extends App with LazyLogging {
//  val sep = "-" * 70
//
//  val config = ConfigFactory.load()
//  val indexDir = config[File]("clint.index-dir")
//
////// Seeds from Marco's latest embedding based algorithm
////   val seeds = Map (
////        "PER" -> Seq("Clinton", "Dole", "Arafat", "Yeltsin", "Lebed", "Dutroux", "Wasim Akram", "Mushtaq Ahmed", "Waqar Younis", "Mother Teresa"),
////        "LOC" -> Seq("U.S.", "Germany", "Britain", "Australia", "France", "England", "Spain", "Italy", "China", "Russia"),
////        "ORG" -> Seq("Reuters", "U.N.", "PUK", "OSCE", "NATO", "EU", "Honda", "European Union", "Ajax", "KDP"),
////        "MISC" -> Seq ("Russian", "German", "British", "French", "Dutch", "Israeli", "GMT", "Iraqi", "European", "English")
////    )
//  // Reading seeds from a file whose location is given by the config param: clint.seedsFile
//  val seedsFile:String = config[String]("clint.seedsFile")
//  val seeds:Map[String,Seq[String]] = Source.fromFile(seedsFile).getLines().mkString("\n").parseJson.convertTo[Map[String,Seq[String]]]
//
//  // Seeds for ScienceIE task
////  val seeds = Map (
////    "Process" -> Seq("oxidation", "isogeometric analysis", "chemical reaction", "SNG", "TIs", "AFM", "simulation", "BEM", "SW", "solar radiation" ),
////    "Material" -> Seq("water", "oxide", "simulations", "liquid",  "surface", "solution", "experimental data", "algorithm", "carbon", "Newman – Penrose constants"  ),
////    "Task" -> Seq("D - flat directions", "microscopy", "energy – momentum scheme", "GFRFs", "radiative spacetimes", "non - autonomous systems", "polymer crystallization", "excitation energies", "Design semantics", "AAMM" )
////  )
//
//  val numEpochs = config[Int]("clint.numEpochs")
//
//  // an array with all category names
//  val categories = seeds.keys.toArray
//  val numCategories = categories.size
//
//  // maps from category name to (entity or pattern) ids
//  val promotedEntities = HashMap.empty[String, HashSet[Int]]
//  val promotedPatterns = HashMap.empty[String, HashSet[Int]]
//
//  val newlyPromotedPatterns = HashMap.empty[String, HashSet[Int]]
//
//  logger.info("loading data")
//  val wordLexicon = LexiconBuilder.loadLexicon(new File(indexDir, "word.lexicon"))
//  val entityLexicon = LexiconBuilder.loadLexicon(new File(indexDir, "entity.lexicon")) //no need to alter TODO: Check?
//  //val normEntities = readMap(new File(indexDir, "entity.normalized")) //no need to alter TODO: Check?
//  val patternLexicon = LexiconBuilder.loadLexicon(new File(indexDir, "entityPatterns.lexicon")) //no need to alter TODO: Check?
//  val entityToPatterns = Index.loadFrom(new File(indexDir, "entityPatternsFiltered.invertedIndex")) //"entityPatterns.invertedIndex")) //no need to alter TODO: Check?
//  val patternToEntities = Index.loadFrom(new File(indexDir, "entityPatterns.filtered.forwardIndex")) //"entityPatterns.forwardIndex")) //no need to alter TODO: Check?
//  val entityCounts = Counts.loadFrom(new File(indexDir, "entityThresholded.counts")) //"entity.counts")) //use Filtered version
//  val patternCounts = Counts.loadFrom(new File(indexDir, "entityPatternsThresholded.counts")) //"entityPatterns.counts")) //use Filtered version
//  val entityPatternCount = Counts2.loadFrom(new File(indexDir, "entityId.entityPatternId.filtered.counts"))//"entityId.entityPatternId.counts")) //use Filtered version
//  val totalEntityCount = entityCounts.counts.values.sum //use Filtered version
//  val totalPatternCount = patternCounts.counts.values.sum //use Filtered version
//  val totalEntityPatternCount = entityPatternCount.counts.values.sum //use Filtered version
//  val goldLabelsFile:File =  config[File]("clint.goldLabelsFile")
//  val goldLabels = Source.fromFile(goldLabelsFile).getLines.map {
//  l =>
//    val tmp = l.split("\t")
//    val k = tmp(0)
//    val v = tmp.tail
//    k -> v }
//  .toMap
//
//  //  val patternCountThreshold = config[Int]("clint.patternCountThreshold") //  val patternCountThreshold = 5
//
////  val patternCountsFiltered = patternCounts.counts.filter(_._2 >= patternCountThreshold)
////  val patternLexiconFiltered = patternLexicon.lexicon diff patternCounts.counts.filter(_._2 < patternCountThreshold).map(_._1).toArray.map(patternLexicon(_))
////  val totalPatternCountFiltered = patternCountsFiltered.values.sum
//
////  otherPatternCounts.map( p =>  patternToEntities( p._1 )).flatten.map( (_,1) ).groupBy(_._1).map(x => (x._1, entityLexicon(x._1), entityCounts.counts(x._1), x._2.size, entityCounts.counts(x._1) - x._2.size ) )
////  val otherPatternCounts = patternCounts.counts.filter(_._2 < patternCountThreshold)
////
////  var entCountsToSubtract = new HashMap[Int, Int]()
////  for(p <- otherPatternCounts.map(_._1) ) {
////    val ents = patternToEntities(p)
////    for (e <- ents) {
////      entCountsToSubtract.put(e, 1)
////    }
////  }
//
//  ///*****************************************************
//
////  var entityCountsFiltered = new HashMap[Int, Int]()
////  entityCounts.counts.foreach { ec =>
////    val e = ec._1
////    val cnt = ec._2
////    val pats = entityToPatterns(e)
////    if(pats.size == 0){
////      logger.info(s"Filtering entityId ${e} (${entityLexicon(e)}) since it does not contain any patterns associated")
////      entityCountsFiltered.put(e, 0)
////    }
////    else
////      entityCountsFiltered.put(e, cnt)
//////    val cntToSub = entCountsToSubtract.getOrElse(e, 0)
//////    entityCountsFiltered.put(e, cnt-cntToSub)
////  }
//////
////  val totalEntityCountFiltered = entityCountsFiltered.values.sum
//
//  //// *******************************************************
//
////  val entityPatternCountFiltered = (entityPatternCount.counts.map {ep_cnt =>
////    val ent = ep_cnt._1._1
////    val pat = ep_cnt._1._2
////    val cnt = ep_cnt._2
////
////    if( entityCountsFiltered.contains(ent) && entityCountsFiltered.get(ent) != 0 && patternCountsFiltered.contains(pat) )
////      Some(ep_cnt)
////    else
////      None
////  }).flatten.toMap
////
////  val totalEntityPatternCountFiltered = entityPatternCountFiltered.values.sum //use Filtered version
//
//
////  val xx = (for (k <- goldLabels.keys) yield {
////    val v = goldLabels(k)
////    val t = for(i <- v) yield (i,k)
////    t
////  }).flatten.toMap
////
////  val yy = (for (lbl <- goldLabels.keys) yield {
////    val arr = goldLabels(lbl)
////    arr.map(x  => (x,1)).groupBy(_._1).map(x => (x._2.size, x._1, lbl) )
////  }).flatten
//
//  val w2vVectorsFile = config[String]("clint.w2vVectors")
//  val w2v = new Word2Vec(w2vVectorsFile, None) //"/Users/ajaynagesh/Research/code/research/clint/data/vectors.txt", None)
//  val numPatternsToSelect = config[Int]("clint.numPatternsToSelect") //5
//
//  val useComboSystem = config[Boolean]("clint.useCombo")
//  val comboSystemType = if(useComboSystem) config[String]("clint.comboSystemType") else null //inkNNOnly, inclassifierOnly, inBoth
//
//  val embedVectorsFile = if(useComboSystem) config[String]("clint.embedVectorsFile") else null
//  val embedVectors = if(useComboSystem) new Word2Vec(embedVectorsFile, None) else null
//  val knnFileEmbed = if(useComboSystem) config[String]("clint.knnFileEmbed") else null
//  logger.info("Begin loading the offline kNN classifier")
//
//  val knnFile = config[String]("clint.knnFile") ///Users/ajaynagesh/Research/code/research/clint/data/knn_offline_list_new.txt"
//  val K = config[Int]("clint.kmeansKvalue") //5
//  val knnClassifier = loadKNNclassifier(knnFile)
//  val knnClassifierEmbed = if(useComboSystem) loadKNNclassifier(knnFileEmbed) else null
//  val bwAccFile = config[String]("clint.accuracyOutputFile") //"/Users/ajaynagesh/Research/code/research/clint/data/accuraciesGupta_knnexpanded.txt"
//  val bwAcc = new BufferedWriter(new FileWriter(new File(bwAccFile)))
//  val kNNmarginTheta = config[Double]("clint.theta") //0.15
//  val classifierConfidenceThreshold = config[Double]("clint.classifierConfidenceThreshold") //0.8
//  val promoteGlobal = config[Boolean]("clint.promoteGlobal") // false
//  val classifierTopNpredictions = if(promoteGlobal) 0 else config[Int]("clint.classifierTopNpredictions") // 10
//  val promoteGlobalPercent = if(promoteGlobal) {
//    logger.info("Global promotion of entities")
//    config[Double]("clint.promoteGlobalPercent")
//  } else {
//    logger.info("Promotion of entities per category")
//    -1.0
//  }
//
//  logger.info("End loading the offline kNN classifier")
//
//  logger.info("promoting seeds")
//  for {
//    cat <- categories
//    ent <- seeds(cat)
//    //norm <- normEntities.get(ent)
//    id <- entityLexicon.get(ent)
//  } promotedEntities.getOrElseUpdate(cat, HashSet.empty[Int]) += id
//
//  for (cat <- categories) promotedPatterns(cat) = HashSet.empty[Int]
//
//  printSeeds(bwAcc)
//
//  logger.info("bootstrapping")
//
//  //TODO: Debug: Why are the entities missing in the inverted index (still 502 are missing, due to normalization of entities ? )
//  val candidateEntities = goldLabels.unzip._2.flatten.toSet.filter ( !entityLexicon.inverseLexicon.get(_).isEmpty  ).map ( entityLexicon.inverseLexicon.get(_).get )
//
//  for (epoch <- 1 to numEpochs) { //Repeat (while some condition is met) {
//
//    //2. Find candidate entities #### --- changing this -->(using the TopK patterns) [see below]
//    //   NOTE: Use all the entities present in the training dataset (minus the entities in the pool) as candidate entities
//    // ----------------------------------------------------------------------------------
//    val newCandidateEntities: Set[Int] =  candidateEntities -- promotedEntities.toSet.unzip._2.flatten
//    logger.info(s"${newCandidateEntities.size} candidate entities present in epoch ${epoch}")
//    // ----------------------------------------------------------------------------------
//
//    //3. [Expansion using kNN classifier]
//    // ----------------------------------------------------------------------------------
////    val expandedEntitiesMap = (for (cat <- categories) yield {
////      val expandedEntities  = if(useComboSystem && (comboSystemType == "inkNNOnly" || comboSystemType == "inBoth"))
////                                expandEntitiesUsingKNN(knnClassifier, knnClassifierEmbed, newCandidateEntities, cat, K)
////                              else
////                                expandEntitiesUsingKNN(knnClassifier, newCandidateEntities, cat, K)
//////      if(cat == "LOC") {
//////        logger.info(s"Expanded entities (LOC) : ${expandedEntities.map ( e => entityLexicon(e._1) ).mkString("\t")}")
//////      }
////      (cat, expandedEntities)
////    }).toMap
//
////    var expandedEntitiesRevMap = HashMap.empty[Int, Array[(String,Double)]]
////    for (cat <- expandedEntitiesMap.keys) {
////      val entities = expandedEntitiesMap.get(cat).get
////      entities.foreach { e =>
////        if (expandedEntitiesRevMap.contains(e._1) )
////          expandedEntitiesRevMap.put(e._1, expandedEntitiesRevMap.get(e._1).get ++ Array((cat,e._2)) )
////        else
////          expandedEntitiesRevMap.put(e._1, Array( (cat,e._2) ))
////      }
////    }
////
////    var nonOVexpandedEntitiesMap = HashMap.empty[String, HashSet[(Int, Double)]]
////    for (cat <- categories) nonOVexpandedEntitiesMap(cat) = HashSet.empty[(Int,Double)]
////    for ( e <- expandedEntitiesRevMap.keys ) {
////      if(expandedEntitiesRevMap.get(e).get.size == 1) { // NOTE: Only add unambiguous entities to the the expanded pool
////        val lbl = expandedEntitiesRevMap.get(e).get.maxBy(_._2)
////        val t =  nonOVexpandedEntitiesMap.get(lbl._1).get
////        nonOVexpandedEntitiesMap.put(lbl._1, nonOVexpandedEntitiesMap.get(lbl._1).get ++ HashSet( (e,lbl._2) ) )
////      }
////    }
////
////    for( (k,v) <- nonOVexpandedEntitiesMap) {
////      logger.info(s"${v.size} entities expanded in epoch ${epoch} for ${k}")
////      logger.info(s"Precision of expanded entities in epoch ${epoch} for ${k} -> ${computeAccuracyExpandedEnts(v.toSet, k)._1}")
////    }
//
//    // ----------------------------------------------------------------------------------
//
//   for (cat <- categories) yield {
//      //1. Find candidate patterns
//      //   Find TopK patterns (using one of the many scoring metrics)
////      val candidatePatterns = getPatterns(promotedEntities(cat).toSet ++ nonOVexpandedEntitiesMap(cat).unzip._1) // NOTE: Expansion Step
//      val candidatePatterns = getPatterns(promotedEntities(cat).toSet) //NOTE: non-expansion
//      val preselectedPatterns = preselectPatterns(cat, candidatePatterns)
//      val selectedPatternsWithScores = selectPatternsWithoutExpansion(/*numPatternsToSelect,*/ cat, preselectedPatterns) //NOTE: non-expansion -- change this to expansion
////      val selectedPatternsWithScores = selectPatterns(/*numPatternsToSelect,*/ cat, preselectedPatterns, nonOVexpandedEntitiesMap(cat))  // NOTE: Expansion Step
////      val selectedPatternsWithScores = selectPatterns(/*numPatternsToSelect,*/ cat, candidatePatterns, nonOVexpandedEntitiesMap(cat)) // No preselection
//      logger.info(s"No of candidate patterns for ${cat}: ${candidatePatterns.size}")
//      logger.info(s"No of pre-selected patterns for ${cat} (used): ${preselectedPatterns.size}")
//      logger.info(s"No of 'selected' patterns with Scores for ${cat}: ${selectedPatternsWithScores.size}")
//      val selectedPatterns = selectedPatternsWithScores.sortBy(- _._2).map(_._2)
//      newlyPromotedPatterns(cat) = HashSet() ++ (selectedPatterns diff promotedPatterns(cat).toSeq ).take(numPatternsToSelect)
//      promotedPatterns(cat) ++= selectedPatterns
//      logger.info(s"${newlyPromotedPatterns(cat).size} new patterns found in epoch ${epoch} for ${cat}")
//      logger.info(sep)
//      logger.info(s"\nPatterns-${}${newlyPromotedPatterns(cat).map ( p => patternLexicon(p).splitOnWhitespace.map {
//        case "@" => "@"
//        case n => wordLexicon(n.toInt)
//      }.mkString(" ")).mkString(s"\nPatterns-${epoch}-${cat}:")}")
//
//    }
//    //-----------------------------------------------------------------------------
//    /****
//      * Classifier based entity selection and promotion
//       */
//    //    //4. Train a multi-class classifier for all the categories.
////    //     Positive examples -- dictionary entities for category c. Negative examples -- dictionary entities of C\c
////    //4.1 Create feature functions
////    //4.2 Create a dataset and populate with set of features
////    //4.3 Call the training API of k-class Logistic Regression classifier
////
////    val dataset = createTrainDataset(promotedEntities, promotedPatterns) // NOTE: Changing this to only promotedEntities (see discussion on Mar 10, 2017)
////    logger.info("Created the dataset")
////
////    val classifier = new LogisticRegressionClassifier[String, String](bias = false)
////    classifier.train(dataset)
////
////    logger.info("Training complete!")
////
////    //5. Pick the entities based on the output of the entity classifier
////
////    val predictions = predictEntityCategories(newCandidateEntities.toArray, classifier) // NOTE: Changing this to newCandidateEntities (see discussion on Mar 10, 2017)
////    logger.info(s"Number of predictions : ${predictions.size}")
////    logger.info(s"Predictions histogram : ${predictions.groupBy(_._2).map(i => (i._1, i._2.size) ).mkString(" :: ")}");
////
////    val newlyPromotedEntities =  HashMap.empty[String, HashSet[Int]]
////    for (cat <- categories) {
////      newlyPromotedEntities.getOrElseUpdate(cat, HashSet.empty[Int])
////    }
////
//////    NOTE: No overlapping promotion of entities (see discussion on Mar 10, 2017)
//////
//////    predictions.zipWithIndex.foreach { p_idx =>
//////      val p = p_idx._1
//////      val idx = p_idx._2
//////      val (scores,predLabel,entity) = (p._1, p._2, p._3)
//////
//////      if(scores.getCount(predLabel) >= classifierConfidenceThreshold) {
//////        promotedEntities(predLabel).add(entity)
//////        newlyPromotedEntities(predLabel).add(entity)
//////      }
//////    }
////
////    var predictionsMap = HashMap.empty[Int, Array[(String,Double)]]
////    predictions.foreach { p =>
////      val (scores,predLabel,entity) = (p._1, p._2, p._3)
////      //if(scores.getCount(predLabel) >= classifierConfidenceThreshold) { // NOTE: Removing the classifier confidence threshold and choosing the top scored classifications irrespective of their absolute scores (See notes of Mar 10, 2017)
////        if(predictionsMap.contains(entity)) {
////          val t = predictionsMap.get(entity).get ++ Array( (predLabel,scores.getCount(predLabel)) )
////          predictionsMap.put(entity, t)
////        }
////        else {
////          val t = Array( (predLabel,scores.getCount(predLabel)) )
////          predictionsMap.put(entity, t)
////        }
////      //}
////    }
////
////    val sortedPredictionsSet = (for (entity <- predictionsMap.keys) yield {
////      val lbl = predictionsMap.get(entity).get.sortBy(- _._2).head
////      (entity, lbl._1, lbl._2)
////    }).toSeq.sortBy(- _._3)//.take(classifierTopNpredictions)
////
////    if(promoteGlobal == false) {
////      val sortedPredictionsByCat = sortedPredictionsSet.groupBy(_._2).map(p => (p._1, p._2.sortBy(- _._3).take(classifierTopNpredictions) ) ).map(_._2).flatten
////
////      for (e_lbl_score <- sortedPredictionsByCat) { // sortedPredictionsSet) {
////        val (entity, lbl, score) = (e_lbl_score._1, e_lbl_score._2, e_lbl_score._3)
////          newlyPromotedEntities(lbl).add(entity)
////          promotedEntities(lbl).add(entity)
////      }
////
////    }
////    else {
////      val numEntitiesToPromote =  math.round(sortedPredictionsSet.size * promoteGlobalPercent).toInt
////      val topXpercentForPromotion = sortedPredictionsSet.take(numEntitiesToPromote)
////      for (e_lbl_score <- topXpercentForPromotion){
////        val (entity, lbl, score) = (e_lbl_score._1, e_lbl_score._2, e_lbl_score._3)
////        newlyPromotedEntities(lbl).add(entity)
////        promotedEntities(lbl).add(entity)
////      }
////    }
//    /****
//      * Classifier based entity selection and promotion
//      * </END> ***/
//    //-----------------------------------------------------------------------------
//    /////**************
//    ///// Ranking based Entity Promotion
//    val newlyPromotedEntities =  HashMap.empty[String, Set[Int]]
//    for (cat <- categories) {
//          newlyPromotedEntities.getOrElseUpdate(cat, Set.empty[Int])
//    }
//    for(cat <- categories){
//      val candidateEntities = getEntities(promotedPatterns(cat).toSet)
//      val selectedEntities = selectEntities(classifierTopNpredictions, cat, candidateEntities)
//      newlyPromotedEntities(cat) = selectedEntities -- promotedEntities(cat)
//      promotedEntities(cat) ++= selectedEntities
//    }
//    /////**************
//    // -----------------------------------------------------------------------------
//
//    logger.info(s"Newly promoted entities histogram :  ${newlyPromotedEntities.map( i => (i._1, i._2.size)).mkString(" :: ")}")
//    logger.info(s"Newly promoted entities --> ${newlyPromotedEntities.map( i => (i._1,  i._2.map( entityLexicon(_) ).mkString(", ") )).mkString(" : " )}")
//
//    bwAcc.write(s"epoch ${epoch}\n")
//    for(cat <- categories) {
//      val entityIds = newlyPromotedEntities(cat)
//      val entities = entityIds.map { e => entityLexicon(e) }
//      bwAcc.write(s"${cat}\t${entities.mkString("\t")}\n")
//    }
//    bwAcc.write("\n")
//
//    logger.info(sep)
//    logger.info(sep)
////    printReport(epoch, newlyPromotedEntities, allExpandedEntities.flatten)
//    val stats = computeAccuracy
//    val perLabelAccuracy = categories.zip(stats._4)
//    logger.info("Accuracy")
//    logger.info(sep)
//    logger.info(s"${epoch}\t${stats._1}\t${stats._2}\t${stats._3}\t${perLabelAccuracy.mkString(" :: ")}")
//    //bwAcc.write(s"${epoch}\t${stats._1}\t${stats._2}\t${stats._3}\n")
//    logger.info(sep)
//    logger.info(sep)
//  }
//
//  bwAcc.close()
//
//  import scala.collection.mutable.HashMap
//  import scala.io._
//
//  def loadKNNclassifier(filename: String) : Map[Int,Array[(Int, Double)]] = {
//    logger.info(s"File is ${filename}")
////    val kNNhash = scala.collection.mutable.HashMap[String,Array[(String, Double)]]()
//    val sourceFileIterator = Source.fromFile(filename).getLines()
//    val pb = new me.tongfei.progressbar.ProgressBar("kNN classifier", sourceFileIterator.size)
//    pb.start
//    val kNNhash = (for (line <- Source.fromFile(filename).getLines() )  yield {
////      println(line)
//      val elements = line.split("\t")
//      val (ent, nnEntsWithScore) = (elements(0).split(" @@ "), elements(1).split(" ;; "))
//      val nearestNeighborEntities = (nnEntsWithScore.map {nnEntWithScore =>
//        val eles = nnEntWithScore.split(" ::: ")
//        val (ent, score) = (eles(0).split(" ## "), eles(1).toDouble)
////        println((ent.mkString("--"), score))
//        (ent(1).toInt, score)
//      })
//      pb.step
////      println(ent.mkString(":") + ".....")
//      (ent(1).toInt -> nearestNeighborEntities)
//    }).toMap
//    pb.stop
//    println(s"kNN hash size: ${kNNhash.size}")
//    kNNhash
//  }
//
//  def expandEntitiesUsingKNN(knnClassifier: Map[Int, Array[(Int,Double)]], candidateEntities : Set[Int], label: String, K: Int) : Set[(Int,Double)] = {
//    val expandedEntities = for(entId <- candidateEntities.par) yield { //OPT
//      val candEntity = StringEscapeUtils.escapeJava(entityLexicon(entId)).replace("(", "").replace(")","")
////      logger.info(s"%%%%%%%%%%%%%% Candidate Entity : ${candEntity}")
//
//      val positiveSimScores = (for(promotedPosEntId <- promotedEntities(label) ) yield {
//        val promotedEntity = StringEscapeUtils.escapeJava(entityLexicon(promotedPosEntId)).replace("(", "").replace(")","")
////        logger.info(s"%%%%% Candidate Entity : ${candEntity} %%%%%%%% promotedEntity : ${promotedEntity}")
////        assert (knnClassifier(promotedEntity).unzip._1.contains(candEntity) )
//        assert (knnClassifier(promotedPosEntId).unzip._1.contains(entId) )
////        val x  = knnClassifier(promotedEntity).find(_._1 == candEntity).get //filter(p => (p._1 == candEntity)).head //OPT
//        val x  = knnClassifier(promotedPosEntId).find(_._1 == entId).get
//        ("positive", x._2)
//      }) //.toSeq.sortBy(- _._2) //OPT
//
//      val negativePromotedEntities = promotedEntities.flatMap(f => f._2).toSet -- promotedEntities(label)
//      val negativeSimScores = (for(promotedNegEntId <- negativePromotedEntities) yield {
//        val promotedEntity = entityLexicon(promotedNegEntId)
////        assert (knnClassifier(promotedEntity).unzip._1.contains(candEntity) )
//        assert (knnClassifier(promotedNegEntId).unzip._1.contains(entId) )
////        val x = knnClassifier(promotedEntity).find(_._1 == candEntity).get //filter(p => (p._1 == candEntity)).head //OPT
//        val x = knnClassifier(promotedNegEntId).find(_._1 == entId).get
//        ("negative", x._2)
//      }) //.toSeq.sortBy (- _._2) //OPT
//
//      val similarityArray = (positiveSimScores ++ negativeSimScores).toSeq.sortBy( - _._2).take(K)
//
//      var posScore = 0.0
//      var negScore = 0.0
//      var posSz = 0
//      var negSz = 0
//
//      for (simEle <- similarityArray) {
//        simEle._1 match {
//          case "positive" => posScore += simEle._2
//                             posSz += 1
//
//          case "negative" => negScore += simEle._2
//                             negSz += 1
//        }
//      }
//
//      if(negSz == 0 && posSz > 0 && posScore / posSz > kNNmarginTheta ) { // NOTE : adding the threshold condition
//        // candidate is not close to any negative instance
//        // we can safely add it to the expandedEntities
////        logger.info(s"${label}- Adding since negSz == 0, ${entityLexicon(entId)}, score is ${posScore}, posSz is ${posSz} ")
//        Some(entId,posScore / posSz)
//      } else if (posSz > 0) {
//        // candidate is close to both positive and negative instances
//        posScore = posScore / posSz
//        negScore = negScore / negSz
//
//        if(posScore > negScore && (posScore - negScore) > kNNmarginTheta) {
////            logger.info(s"${label}- Adding since margin ${posScore - negScore}, posSz = ${posSz}, negSz = ${negSz}, ${entityLexicon(entId)}")
//
//           Some(entId,posScore)
//        }
//        else None
//      }
//      else None
//    }
//
//    expandedEntities.seq.flatten
//  }
//
//  def expandEntitiesUsingKNN(knnClassifier: Map[Int, Array[(Int,Double)]], knnClassifierEmbed: Map[Int, Array[(Int,Double)]], candidateEntities : Set[Int], label: String, K: Int) : Set[(Int,Double)] = {
//    val expandedEntities = for(entId <- candidateEntities.par) yield { //OPT
////      val candEntity = entityLexicon(entId)
////      logger.info(s"%%%%%%%%%%%%%% Candidate Entity : ${candEntity}")
//
//      val positiveSimScores = (for(promotedPosEntId <- promotedEntities(label) ) yield {
////        val promotedEntity = entityLexicon(promotedPosEntId)
////        logger.info(s"%%%%% Candidate Entity : ${candEntity} %%%%%%%% promotedEntity : ${promotedEntity}")
//
//        assert (knnClassifier(promotedPosEntId).unzip._1.contains(entId) )
//        val x  = knnClassifier(promotedPosEntId).find(_._1 == entId).get
//        val y = if (knnClassifierEmbed.contains(promotedPosEntId)) knnClassifierEmbed(promotedPosEntId).find(_._1 == entId).get else (-1,0.0)
//        ("positive", (x._2+y._2)/2 )
//      })
//
//      val negativePromotedEntities = promotedEntities.flatMap(f => f._2).toSet -- promotedEntities(label)
//      val negativeSimScores = (for(promotedNegEntId <- negativePromotedEntities) yield {
////        val promotedEntity = entityLexicon(promotedNegEntId)
//        assert (knnClassifier(promotedNegEntId).unzip._1.contains(entId) )
//        val x = knnClassifier(promotedNegEntId).find(_._1 == entId).get
//        val y = if (knnClassifierEmbed.contains(promotedNegEntId)) knnClassifierEmbed(promotedNegEntId).find(_._1 == entId).get else (-1,0.0)
//        ("negative", (x._2+y._2)/2 )
//      })
//
//      val similarityArray = (positiveSimScores ++ negativeSimScores).toSeq.sortBy( - _._2).take(K)
//
//      var posScore = 0.0
//      var negScore = 0.0
//      var posSz = 0
//      var negSz = 0
//
//      for (simEle <- similarityArray) {
//        simEle._1 match {
//          case "positive" => posScore += simEle._2
//                             posSz += 1
//
//          case "negative" => negScore += simEle._2
//                             negSz += 1
//        }
//      }
//
//      if(negSz == 0 && posSz > 0 && posScore / posSz > kNNmarginTheta ) { // NOTE : adding the threshold condition
//        // candidate is not close to any negative instance
//        // we can safely add it to the expandedEntities
////        logger.info(s"${label}- Adding since negSz == 0, ${entityLexicon(entId)}, score is ${posScore}, posSz is ${posSz} ")
//        Some(entId,posScore / posSz)
//      } else if (posSz > 0) {
//        // candidate is close to both positive and negative instances
//        posScore = posScore / posSz
//        negScore = negScore / negSz
//
//        if(posScore > negScore && (posScore - negScore) > kNNmarginTheta) {
////            logger.info(s"${label}- Adding since margin ${posScore - negScore}, posSz = ${posSz}, negSz = ${negSz}, ${entityLexicon(entId)}")
//
//           Some(entId,posScore)
//        }
//        else None
//      }
//      else None
//    }
//
//    expandedEntities.seq.flatten
//  }
//
//  def expandEntitiesUsingKNN(knnClassifier: HashMap[String, Array[(String,String,Double)]], candidateEntities : Set[Int], label: String) : HashSet[(Int,Double)] = {
//
//    val expandedEntities = HashSet[(Int,Double)]()
//
//    val setOfTopKpoolEntities = for(entId <- candidateEntities) yield {
//      val candEntity = entityLexicon(entId)
//      val mostSimEntitiesOption = knnClassifier.get(candEntity)
//
//      if(mostSimEntitiesOption.nonEmpty){
//        val mostSimEntities = mostSimEntitiesOption.get
//        val positiveSimEntities = ArrayBuffer[(String,Double)]()
//        val negativeSimEntities = ArrayBuffer[(String,Double)]()
//
////      logger.info(candEntity)
//        val (mostSimEnts, lbl, scores) = mostSimEntities.unzip3
//        val intersection = promotedEntities.get(label).get.map { pe => entityLexicon(pe)}.toArray.intersect(mostSimEnts)
//        positiveSimEntities ++= (for(its <- intersection) yield {
//          val idx = mostSimEnts.indexOf(its)
//          val score = scores(idx)
//          (its ,score)
//        })
//
//        for(cat <- categories.filterNot { x => x.equals(label) }) {
//          val intersection = promotedEntities.get(cat).get.map { pe => entityLexicon(pe)}.toArray.intersect(mostSimEnts)
//          val simPromotedEntitiesPerCat = for(its <- intersection) yield {
//            val idx = mostSimEnts.indexOf(its)
//            val score = scores(idx)
//            (its ,score)
//          }
//          negativeSimEntities ++= simPromotedEntitiesPerCat
//        }
//
//        Some(entId, positiveSimEntities.toArray, negativeSimEntities.toArray)
//
//      }
//      else {
//        None
//      }
//    }
//
//    for(kNNaugmentedEntity <- setOfTopKpoolEntities) {
//      if(kNNaugmentedEntity.nonEmpty){
//        val (candEntId, posSimEntities, negSimEntities) = kNNaugmentedEntity.get
//        val positiveScore = posSimEntities.unzip._2.sum / posSimEntities.size
//        val negativeScore = negSimEntities.unzip._2.sum / negSimEntities.size
////          if(! positiveScore.isNaN() && ! negativeScore.isNaN())
////            logger.info(s"debugg - ${(entityLexicon(candEntId), posSimEntities.mkString(":"), negSimEntities.mkString(":"))}\t$positiveScore\t$negativeScore")
//        if(positiveScore > negativeScore && (positiveScore - negativeScore) > kNNmarginTheta) {
////          logger.info(s"Adding ${entityLexicon(candEntId)} to category -> $label")
//          expandedEntities.add((candEntId,positiveScore))
//
//        }
//      }
//    }
//
//    expandedEntities
//  }
//
//  def computeAccuracyExpandedEnts(expandedEnts: Set[(Int, Double)], category: String) : (Double, Double) = {
//
//     val entitiesPredicted = expandedEnts.map { e => entityLexicon(e._1) }
//     val entitiesInGold = goldLabels(category).toSet // convert to set to avoid duplicates
//     val correctEntities = entitiesInGold.intersect(entitiesPredicted)
//
//     val precision = correctEntities.size.toDouble / entitiesPredicted.size
//     val recall = correctEntities.size.toDouble / entitiesInGold.size
//     (precision, recall)
//  }
//
//  def computeAccuracy() : (Double, Double, Int, Array[Double]) = {
//    val stats = for(cat <- categories) yield {
//      val entitiesPredicted = promotedEntities(cat).map { e => entityLexicon(e) }
//      val entitiesInGold = goldLabels(cat).toSet // convert to set to avoid duplicates
//      val correctEntities = entitiesInGold.intersect(entitiesPredicted)
//
//      val precision = correctEntities.size.toDouble / entitiesPredicted.size
//      val recall = correctEntities.size.toDouble / entitiesInGold.size
//      (precision, recall, entitiesPredicted.size)
//    }
//
//    val (precision, recall, size) = stats.unzip3
//    val avgPrecision = precision.sum / precision.size
//    val avgRecall = recall.sum / recall.size
//    val sumSize = size.sum
//    (avgPrecision, avgRecall, sumSize, precision)
//
//  }
//
//  def predictEntityCategories(entities: Array[Int], classifier : LogisticRegressionClassifier[String, String]) = {
//
//    val dataset = new RVFDataset[String, String]()
//
//      val predictions = for (e <- entities) yield {
//        val c = new Counter[String]
//
//        for( cat <- categories) {
//
//          //TODO: Need to verify if the feature creation during test dataset creation is correct
//
//          // Edit distance based features
//          val EDPVal = EditDistPosFeature(entityLexicon(e), cat)
//      		val EDNVal = EditDistNegFeature(entityLexicon(e), cat)
//    	  	c.incrementCount("EDP"+cat, EDPVal)
//  	      c.incrementCount("EDN"+cat, EDNVal)
//
//  	      // Pattern-based features
////    			val PTFVal = PatternTFIDFScore(e, promotedPatterns, cat)
////  	  		c.incrementCount("PTF"+cat, PTFVal)
//
//  	  		val PMIscore = PatternPMIScores(e, promotedPatterns, cat)
//  		    c += PMIscore
//
//  	      // W2V / glove based features
//          c += computeW2Vfeatures(e, cat)
//        }
//
//        val datum =	new RVFDatum[String, String]("~", c)
//    		(classifier.scoresOf(datum), classifier.classOf(datum), e)
//
//      }
//
//    predictions
//  }
//
//  def EditDistPosFeature(candEntity: String, category: String) : Double = {
//   ((for(posEntId <- promotedEntities(category)) yield { // compute edit distance of entity with every entity in the positive pool of category
//      val posEntity = entityLexicon(posEntId)
//      (LevenshteinMetric.compare(posEntity, candEntity).get.toFloat / posEntity.length())
//    })
//    (collection.breakOut)) // To convert HashSet to Array
//    .map { score => if (score < 0.2) 1 else 0 }
//    .max
//  }
//
//  def EditDistNegFeature(candEntity: String, category: String) : Double = {
//
//    val negEntitiesSet = promotedEntities.values.flatten.toSet -- promotedEntities(category).toSet // Set of all entities that do not belong to category
//    val score  = ((for(negEntId <- negEntitiesSet) yield {
//      val negEntity = entityLexicon(negEntId)
//      (LevenshteinMetric.compare(negEntity, candEntity).get.toFloat / negEntity.length())
//    })
//    (collection.breakOut)) // To convert HashSet to Array
//    .map { score => if (score < 0.2) 1 else 0 }
//    .max
//
//    1 - score
//
//  }
//
//  def PatternTFIDFScore(entityID: Int, promotedPatterns: HashMap[String, HashSet[Int]], category: String) : Double = {
//    val freqEntity = entityCounts.counts(entityID)  // freq_e     /////entityCountsFiltered(entityID) //
//    val patternSet = entityToPatterns.index(entityID) // R
//
//    val posEntities = promotedEntities(category)
//    val negEntities = promotedEntities.values.flatten.toSet -- promotedEntities(category).toSet
//
//    var ptfval = 0.0
//    for(r <- patternSet) {
//      val entities_from_r = patternToEntities.index(r)
//      val P_r = entities_from_r.intersect(posEntities).size.toDouble
//      val N_r = entities_from_r.intersect(negEntities).size
//
//      val ps_r = if(P_r > 0 && N_r != 0)
//                   P_r * scala.math.log(P_r) / N_r
//                 else // TODO: verify if this is right ?
//                   0.0
//
//      ptfval += ps_r
//    }
//
//    ptfval = if (freqEntity > 0)
//                ptfval / scala.math.log(freqEntity)
//             else
//                0.0
//
//    // TODO: to normalize the score (ptfval)
//
//    ptfval
//  }
//
//  def PatternPMIScores(entityID: Int, promotedPatterns: HashMap[String, HashSet[Int]], category: String) : Counter[String] = {
//    val promotedPatternForCategory = promotedPatterns(category)
//    val counter = new Counter[String]
//
//    for(patternId <- promotedPatternForCategory){
//      val ep_cnt = entityPatternCount.counts.getOrElse((entityID, patternId), 0) // entityPatternCountFiltered.getOrElse((entityID, patternId), 0)
//      val e_cnt =  entityCounts.counts.getOrElse(entityID, 0) // entityCountsFiltered.getOrElse(entityID, 0)  //
//      val p_cnt = patternCounts.counts.getOrElse(patternId, 0) //patternCountsFiltered.getOrElse(patternId, 0) //
//
//      val p_ep = ep_cnt.toDouble / totalEntityPatternCount // ep_cnt.toDouble / totalEntityPatternCountFiltered //
//      val p_e = e_cnt.toDouble / totalEntityCount // e_cnt.toDouble / totalEntityCountFiltered //
//      val p_p = p_cnt.toDouble / totalPatternCount // p_cnt.toDouble / totalPatternCountFiltered  //
//
//      val pmi = if(p_ep == 0) 0 else math.log( p_ep / (p_e * p_p) )
//      counter.incrementCount(patternLexicon(patternId)+":"+category+":pmi", pmi)
//
//      val pattern = (patternLexicon(patternId).splitOnWhitespace.map {
//          case "@" => "@"
//          case n => wordLexicon(n.toInt)
//      }).mkString(" ")
//
////      logger.info(s"${pattern}:${category}:pmi, $pmi")
//    }
//
//    counter
//  }
//
//  // Input: PromotedEntities in every category, PromotedPatterns in every category
//  // Output:
//  def createTrainDataset(entities: HashMap[String, HashSet[Int]], promotedPatterns: HashMap[String, HashSet[Int]]): RVFDataset[String, String] = {
//
//	  val dataset = new RVFDataset[String, String]()
//
//	  println(s"categories : ${categories.mkString(", ")}")
//
//	  for(cat <- categories){
////	      logger.info(s"[dataset] $cat ")
//		    entities(cat).foreach { e =>
//
//    		  val c = new Counter[String]
//  //  		  logger.info(s"[dataset] Creating features for entity ${entityLexicon(e)}")
//    		  // Edit distance-based features
//    		  val EDPVal = EditDistPosFeature(entityLexicon(e), cat)
//    		  val EDNVal = EditDistNegFeature(entityLexicon(e), cat)
//    		  c.incrementCount("EDP"+cat, EDPVal) // TODO: Check: Should we index the feature with the category too ?
//    		  c.incrementCount("EDN"+cat, EDNVal)
//
//  //  		  logger.info(s"[dataset] feature EDP${cat} has value : $EDPVal")
//  //  			logger.info(s"[dataset] feature EDN${cat} has value : $EDNVal")
//
//    			// Pattern-based features
//  //  			val PTFVal = PatternTFIDFScore(e, promotedPatterns, cat)
//  //  			c.incrementCount("PTF"+cat, PTFVal)
//
//    		  c += PatternPMIScores(e, promotedPatterns, cat)
//
//    			// W2V/Glove - embedding-based features
//    			c += computeW2Vfeatures(e, cat)
//
//    			val datum =	new RVFDatum[String, String](cat, c)
//    			dataset += datum
//  		}
//	  }
//
//	  dataset
//  }
//
//  def computeW2Vfeatures(givenEntityId: Int, category: String) : Counter[String] = {
////    val givenEntity = Word2Vec.sanitizeWord(entityLexicon(givenEntityId))
//    val givenEntity = entityLexicon(givenEntityId)
//    val counter = new Counter[String]
//
////    val entitiesInPool = promotedEntities(category).map { e => Word2Vec.sanitizeWord(entityLexicon(e)) }
//    val entitiesInPool = promotedEntities(category).map { e => entityLexicon(e) }
//    for(entityInPool <- entitiesInPool) {
//      val sim = w2v.sanitizedAvgSimilarity(givenEntity.split(" +"), entityInPool.split(" +"))
////      val sim = w2v.similarity(Word2Vec.sanitizeWord(givenEntity.replace(" ", "_")), Word2Vec.sanitizeWord(entityInPool.replace(" ", "_")))
//      val embedSim = if(useComboSystem && (comboSystemType == "inclassifierOnly" || comboSystemType == "inBoth") )
//                        embedVectors.similarity(Word2Vec.sanitizeWord(givenEntity.replace(" ", "_")), Word2Vec.sanitizeWord(entityInPool.replace(" ", "_")))
//                     else
//                       0
//      val key = s"${givenEntity}:${entityInPool}:${category}"
//      if(useComboSystem && (comboSystemType == "inclassifierOnly" || comboSystemType == "inBoth") )
//        counter.incrementCount(key, (sim._1 + embedSim)/2)
//      else
//        counter.incrementCount(key, sim._1)
//    }
//
//    counter
//  }
//
//  def printPatterns(patterns: HashSet[Int]): Unit = {
//    for(p <- patterns) {
//       val words = patternLexicon(p).splitOnWhitespace.map {
//         case "@" => "@"
//         case n => wordLexicon(n.toInt)
//       }
//       println(words.mkString(" "))
//    }
//  }
//
//  def printEntities(entities: Set[Int]): Unit = {
//    for (e <- entities) {
//      println(entityLexicon(e))
//    }
//  }
//
//  def printSeeds(bwAcc: BufferedWriter): Unit = {
//    println("Seeds")
//    bwAcc.write("Epoch 0\n")
//    for (cat <- categories) {
//      println(s"$cat entities:")
//      for (e <- seeds(cat)) {
//        println(e)
//      }
//      bwAcc.write(s"${cat}\t${seeds(cat).mkString("\t")}\n")
//      println()
//    }
//    bwAcc.write("\n")
//    println("=" * 70)
//    println()
//  }
//
////  def printReport(epoch: Int, newlyPromotedEntities: HashMap[String, HashSet[Int]],  expandedEntities: HashSet[(Int, Double)]): Unit = {
////    println(s"Bootstrapping epoch $epoch")
////    for (cat <- categories) {
////      println(s"$cat entities:")
////      for (e <- newlyPromotedEntities(cat).toSeq.sortBy(e => scoreEntity(e, cat))) {
////        println(scoreEntity(e, cat) + "\t" + entityLexicon(e))
////      }
////      println(s"\n$cat patterns:")
////      for (p <- newlyPromotedPatterns(cat).toSeq.sortBy(p => scorePattern(p, cat, expandedEntities))) {
////        val words = patternLexicon(p).splitOnWhitespace.map {
////          case "@" => "@"
////          case n => wordLexicon(n.toInt)
////        }
////        println(scorePattern(p, cat) + "\t" + words.mkString(" "))
////      }
////      println()
////    }
////    println("=" * 70)
////    println()
////  }
//
//  def getPatterns(entities: Set[Int]): Set[Int] = {
//    entities.flatMap(e => entityToPatterns(e))
//  }
//
//  def getEntities(patterns: Set[Int]): Set[Int] = {
//    patterns.flatMap(p => patternToEntities(p))
//  }
//
//  def preselectPatterns(category: String, patterns: Set[Int]): Set[Int] = {
//    val results = for (p <- patterns) yield {
//      val candidates = patternToEntities(p)
//      val entities = promotedEntities(category)
//      val matches = candidates intersect entities
//      val accuracy = matches.size.toDouble / candidates.size.toDouble
//      if (matches.size <= 1) None // NOTE: Changing check from '==' I think the expected behavior is, do not select patterns if there are less than or equal to 1 overlaps
//      // else if (accuracy < 0.5) None
//      else Some(p)
//    }
//    results.flatten
//  }
//
//  def preselectEntities(category: String, entities: Set[Int]): Set[Int] = {
//    val results = for (e <- entities) yield {
//      val candidates = entityToPatterns(e)
//      val patterns = promotedPatterns(category)
//      val matches = candidates intersect patterns
//      val accuracy = matches.size.toDouble / candidates.size.toDouble
//      if (matches.size == 1) None
//      // else if (accuracy < 0.5) None
//      else Some(e)
//    }
//    results.flatten
//  }
//
////   def selectPatterns(n: Int, category: String, patterns: Set[Int]): Set[Int] = {
////     val scoredPatterns = for (p <- patterns) yield {
////       val patternCount = patternCounts(p)
////       val dividend = promotedEntities(category).map(e => entityPatternCount(e, p)).sum
////       val score = dividend / patternCount
////       (score, p)
////     }
////     scoredPatterns.toSeq.sortBy(_._1).map(_._2).filterNot(promotedPatterns(category).contains).toSet
////   }
//
////   def scorePattern(p: Int, cat: String): Double = {
////     val entities = patternToEntities(p)
////     val patternTotalCount: Double = patternCounts(p)
////     val positiveCount: Double = entities
////       .filter(promotedEntities(cat).contains) // only the ones in the pool
////       .map(entityPatternCount(_, p)) // count them
////       .sum // total sum
////     (positiveCount / patternTotalCount) * math.log(patternTotalCount)
////   }
//
////   SCORES
//
////   def scorePattern(p: Int, cat: String): Double = riloffPattern(p, cat)
//     def scoreEntity(e: Int, cat: String): Double = riloffEntity(e, cat)
//
////   def scorePattern(p: Int, cat: String): Double = collinsPattern(p, cat)
////   def scoreEntity(e: Int, cat: String): Double = collinsEntity(e, cat)
//
////   def scorePattern(p: Int, cat: String): Double = chi2Pattern(p, cat)
////   def scoreEntity(e: Int, cat: String): Double = chi2Entity(e, cat)
//
////   def scorePattern(p: Int, cat: String): Double = mutualInformationPattern(p, cat)
////   def scoreEntity(e: Int, cat: String): Double = mutualInformationEntity(e, cat)
//
////   def scorePattern(p: Int, cat: String): Double = mutualInformationLogFreqPattern(p, cat)
//
//     def scorePatternWithoutExpansion(p: Int, cat: String): Double = PMIpattern(p, cat)
////     def scorePattern(p: Int, cat: String,  expandedEntities: HashSet[(Int, Double)]): Double = softPMIpattern(p, cat, expandedEntities)
//     def scorePattern(p: Int, cat: String,  expandedEntities: HashSet[(Int, Double)]): Double = hardPMIpattern(p, cat, expandedEntities) * math.log(patternCounts(p))
//
//  // number of times pattern `p` matches an entity with category `cat`
//  def positiveEntityCounts(p: Int, cat: String): Double = {
//    patternToEntities(p)
//      .filter(promotedEntities(cat).contains) // only the ones in the pool
////      .map(entityPatternCountFiltered(_, p)) // count them
//      .map(entityPatternCount(_, p)) // count them
//      .sum // total count
//  }
//
//  def positivePatternCounts(e: Int, cat: String): Double = {
//    entityToPatterns(e)
//      .filter(promotedPatterns(cat).contains) // only the ones in the pool
//      .map(entityPatternCount(e, _)) // count them
////      .map(entityPatternCountFiltered(e, _)) // count them
//      .sum // total count
//  }
//
//  def countEntityMentions(cat: String): Double = {
//    promotedEntities(cat).map(entityCounts(_)).sum
////    promotedEntities(cat).map(entityCountsFiltered(_)).sum
//  }
//
//  def countPatternMentions(cat: String): Double = {
//    promotedPatterns(cat).map(patternCounts(_)).sum
////    promotedPatterns(cat).map(patternCountsFiltered(_)).sum
//  }
//
//  // precision of pattern `p` in the set of entities labeled with category `cat`
//  def precisionPattern(p: Int, cat: String): Double = {
//    val total: Double = patternCounts(p) // patternCountsFiltered(p) //
//    val positive: Double = positiveEntityCounts(p, cat)
//    positive / total
//  }
//
//  def precisionEntity(e: Int, cat: String): Double = {
//    val total: Double = entityCounts(e) // entityCountsFiltered(e) //
//    val positive: Double = positivePatternCounts(e, cat)
//    positive / total
//  }
//
//  def riloffPattern(p: Int, cat: String): Double = {
//    val prec = precisionPattern(p, cat)
//    if (prec > 0) {
////      prec * math.log(patternCountsFiltered(p))
//      prec * math.log(patternCounts(p))
//      // prec * math.log(positiveEntityCounts(p, cat))
//    } else {
//      0
//    }
//  }
//
//  def riloffEntity(e: Int, cat: String): Double = {
//    val prec = precisionEntity(e, cat)
//    if (prec > 0) {
////      prec * math.log(entityCountsFiltered(e))
//      prec * math.log(entityCounts(e))
//      // prec * math.log(positivePatternCounts(e, cat))
//    } else {
//      0
//    }
//  }
//
//  def collinsPattern(p: Int, cat: String): Double = {
//    val prec = precisionPattern(p, cat)
//    if (prec > 0.95) {
////        patternCountsFiltered(p)
//      patternCounts(p)
//    } else {
//      0
//    }
//  }
//
//  def collinsEntity(e: Int, cat: String): Double = {
//    val prec = precisionEntity(e, cat)
//    if (prec > 0.95) {
////      entityCountsFiltered(e)
//      entityCounts(e)
//    } else {
//      0
//    }
//  }
//
//  def chi2Pattern(p: Int, cat: String): Double = {
//    val prec = precisionPattern(p, cat)
//    if (prec > 0.5) {
//      val n: Double =  totalEntityCount // totalEntityCountFiltered //
//      val a: Double = positiveEntityCounts(p, cat)
//      val b: Double = patternCounts(p) - positiveEntityCounts(p, cat) // patternCountsFiltered(p) - positiveEntityCounts(p, cat) //
//      val c: Double = countEntityMentions(cat) - positiveEntityCounts(p, cat)
//      val d: Double = n - countEntityMentions(cat) - patternCounts(p) + positiveEntityCounts(p, cat) // n - countEntityMentions(cat) - patternCountsFiltered(p) + positiveEntityCounts(p, cat) //
//      val numerator = n * math.pow(a * d - c * b, 2)
//      val denominator = (a + c) * (b + d) * (a + b) * (c + d)
//      numerator / denominator
//    } else {
//      0
//    }
//  }
//
//  def chi2Entity(e: Int, cat: String): Double = {
//    val prec = precisionEntity(e, cat)
//    if (prec > 0.5) {
//      val n: Double = totalPatternCount // totalPatternCountFiltered //
//      val a: Double = positivePatternCounts(e, cat)
//      val b: Double = entityCounts(e) - positivePatternCounts(e, cat) // entityCountsFiltered(e) - positivePatternCounts(e, cat) //
//      val c: Double = countPatternMentions(cat) - positivePatternCounts(e, cat)
//      val d: Double = n - countPatternMentions(cat) - entityCounts(e) + positivePatternCounts(e, cat) //n - countPatternMentions(cat) - entityCountsFiltered(e) + positivePatternCounts(e, cat) //
//      val numerator = n * math.pow(a * d - c * b, 2)
//      val denominator = (a + c) * (b + d) * (a + b) * (c + d)
//      numerator / denominator
//    } else {
//      0
//    }
//  }
//
//  def mutualInformationPattern(p: Int, cat: String): Double = {
//    val prec = precisionPattern(p, cat)
//    if (prec > 0.5) {
//      val n: Double = totalEntityCount // totalEntityCountFiltered //
//      val a: Double = positiveEntityCounts(p, cat)
//      val b: Double = patternCounts(p) - positiveEntityCounts(p, cat) // patternCountsFiltered(p) - positiveEntityCounts(p, cat) //
//      val c: Double = countEntityMentions(cat) - positiveEntityCounts(p, cat)
//      math.log((n * a) / ((a + c) * (a + b)))
//    } else {
//      0
//    }
//  }
//
//  def PMIpattern(p: Int, cat: String): Double = {
//    val n: Double = totalEntityCount // totalEntityCountFiltered //
//    val numerator = positiveEntityCounts(p, cat)
//    val denom = countEntityMentions(cat) * patternCounts(p) // countEntityMentions(cat) * patternCountsFiltered(p) //
////    math.log( (n * numerator / denom ) )
//    math.log( numerator / denom  )
//  }
//
//  def hardPMIpattern(p: Int, cat: String, expandedEntities: HashSet[(Int, Double)]): Double = {
////    softPMIpattern(p, cat, expandedEntities, promotedEntities, patternToEntities, entityPatternCount, entityCounts, patternCounts)
//    hardPMIpattern(p, cat, expandedEntities, promotedEntities, patternToEntities, entityPatternCount, entityCounts, patternCounts)
//  }
//
//  def softPMIpattern(p: Int, cat: String, expandedEntities: HashSet[(Int, Double)]): Double = {
////    softPMIpattern(p, cat, expandedEntities, promotedEntities, patternToEntities, entityPatternCount, entityCounts, patternCounts)
//    softPMIpattern(p, cat, expandedEntities, promotedEntities, patternToEntities, entityPatternCount, entityCounts, patternCounts)
//  }
//
//  def hardPMIpattern(
//      p: Int,
//      cat: String,
//      expandedEntities: HashSet[(Int, Double)],
//      promotedEntities: HashMap[String, HashSet[Int]], // pools
//      patternToEntities: Index,
//      entityPatternCount: Counts2, // Map[(Int, Int),Int],
//      entityCounts: Counts, //HashMap[Int,Int], // Counts, //  number of mentions per entity
//      patternCounts: Counts // Map[Int,Int]
//  ): Double = {
//    val numerator = positiveEntityCountsHard(p, cat, expandedEntities, promotedEntities, patternToEntities, entityPatternCount)
//    val denom = countEntityMentionsHard(cat, expandedEntities, promotedEntities, entityCounts) * patternCounts(p)
//    math.log(numerator / denom)
//  }
//
//  def softPMIpattern(
//      p: Int,
//      cat: String,
//      expandedEntities: HashSet[(Int, Double)],
//      promotedEntities: HashMap[String, HashSet[Int]], // pools
//      patternToEntities: Index,
//      entityPatternCount: Counts2, // Map[(Int, Int),Int],
//      entityCounts: Counts, //HashMap[Int,Int], // Counts, //  number of mentions per entity
//      patternCounts: Counts // Map[Int,Int]
//  ): Double = {
//    val numerator = positiveEntityCountsSoft(p, cat, expandedEntities, promotedEntities, patternToEntities, entityPatternCount)
//    val denom = countEntityMentionsSoft(cat, expandedEntities, promotedEntities, entityCounts) * patternCounts(p)
//    math.log(numerator / denom)
//  }
//
//  // number of times pattern `p` matches an entity with category `cat` with soft scores attached from the sim of kNN
//  def positiveEntityCountsSoft(
//      p: Int,
//      cat: String,
//      expandedEntities: HashSet[(Int, Double)],
//      promotedEntities: HashMap[String, HashSet[Int]], // pools
//      patternToEntities: Index,
//      entityPatternCount: Counts2 //Map[(Int, Int),Int]
//  ): Double = {
//    val promotedEntitiesCount = patternToEntities(p)
//      .filter(promotedEntities(cat).contains) // only the ones in the pool
//      .map(entityPatternCount(_, p)) // count them
//      .sum // total count
//
//    val expandedEntitiesArray = expandedEntities.toArray.unzip
//    val expandedEntitiesSimCounts = patternToEntities(p)
//      .filter(expandedEntitiesArray._1.contains) // only the ones in the expanded entities
//      .map ( expandedEntitiesArray._1.indexOf(_) ) // get the indices of the entities
//      .map ( expandedEntitiesArray._2(_) ) //get the similarity scores (corresponding to the indices)
//
//   val expandedEntitiesOccurrenceCounts = patternToEntities(p)
//      .filter(expandedEntitiesArray._1.contains) // get the ones in the expanded entities
//      .map(entityPatternCount(_, p)) // count them
//
//   val expandedEntitiesCount = expandedEntitiesSimCounts.zip(expandedEntitiesOccurrenceCounts)
//                     .map(f => f._1 * f._2) // multiply the count with the similarity scores
//                     .sum // and then sum
//
//   promotedEntitiesCount + expandedEntitiesCount
//
//  }
//
//  def positiveEntityCountsHard(
//      p: Int,
//      cat: String,
//      expandedEntities: HashSet[(Int, Double)],
//      promotedEntities: HashMap[String, HashSet[Int]], // pools
//      patternToEntities: Index,
//      entityPatternCount: Counts2 //Map[(Int, Int),Int]
//  ): Double = {
//    val promotedEntitiesCount = patternToEntities(p)
//      .filter(promotedEntities(cat).contains) // only the ones in the pool
//      .map(entityPatternCount(_, p)) // count them
//      .sum // total count
//
//    val expandedEntitiesArray = expandedEntities.toArray.unzip
//    val expandedEntitiesCount = patternToEntities(p)
//      .filter(expandedEntitiesArray._1.contains) // only the ones in the pool
//      .map(entityPatternCount(_, p)) // count them
//      .sum // total count
//
//   promotedEntitiesCount + expandedEntitiesCount
//
//  }
//
//   def countEntityMentionsHard(
//      cat: String,
//      expandedEntities: HashSet[(Int, Double)],
//      promotedEntities: HashMap[String, HashSet[Int]], // pools
//      entityCounts: Counts //HashMap[Int, Int] // Counts //  number of mentions per entity
//  ): Double = {
//    val promotedEntitiesSum = promotedEntities(cat).map(entityCounts(_)).sum
//
//    val expandedEntitiesSum = expandedEntities.map {e =>
//      val e_count = entityCounts(e._1)
//      e_count
//    }
//    .sum
//
//    promotedEntitiesSum + expandedEntitiesSum
//  }
//
//  def countEntityMentionsSoft(
//      cat: String,
//      expandedEntities: HashSet[(Int, Double)],
//      promotedEntities: HashMap[String, HashSet[Int]], // pools
//      entityCounts: Counts //HashMap[Int, Int] // Counts //  number of mentions per entity
//  ): Double = {
//    val promotedEntitiesSum = promotedEntities(cat).map(entityCounts(_)).sum
//
//    val expandedEntitiesSum = expandedEntities.map {e =>
//      val e_count = entityCounts(e._1)
//      val e_score = e._2
//      e_count * e_score
//    }
//    .sum
//
//    promotedEntitiesSum + expandedEntitiesSum
//  }
//
//  def mutualInformationLogFreqPattern(p: Int, cat: String): Double = {
//    val mi = mutualInformationPattern(p, cat)
//    val freq = patternCounts(p) // patternCountsFiltered(p) //
//    if(freq == 0) 0 else mi * math.log(freq)
//  }
//
//  def mutualInformationEntity(e: Int, cat: String): Double = {
//    val prec = precisionEntity(e, cat)
//    if (prec > 0.5) {
//      val n: Double = totalPatternCount // totalPatternCountFiltered //
//      val a: Double = positivePatternCounts(e, cat)
//      val b: Double = entityCounts(e) - positivePatternCounts(e, cat) // entityCountsFiltered(e) - positivePatternCounts(e, cat) //
//      val c: Double = countPatternMentions(cat) - positivePatternCounts(e, cat)
//      math.log((n * a) / ((a + c) * (a + b)))
//    } else {
//      0
//    }
//  }
//
//
//
////   def scorePattern(p: Int, cat: String): Double = {
////     val matches = patternToEntities(p)
////     val pos = promotedEntities(cat) intersect matches
////     val precision = pos.size.toDouble / matches.size.toDouble
////     val score = precision * math.log(pos.size)
////     score
////   }
//
//  def patternString(p: Int): String = {
//    val words = patternLexicon(p).splitOnWhitespace.map {
//      case "@" => "@"
//      case n => wordLexicon(n.toInt)
//    }
//    words.mkString(" ")
//  }
//
////   def scoreEntity(e: Int, cat: String): Double = {
////     val patterns = entityToPatterns(e)
////     val entityTotalCount: Double = entityCounts(e)
////     val positiveCount: Double = patterns
////       .filter(promotedPatterns(cat).contains) // only the ones in the pool
////       .map(entityPatternCount(e, _)) // count them
////       .sum // total sum
////     (positiveCount / entityTotalCount) * math.log(entityTotalCount)
////   }
//
////   def scoreEntity(e: Int, cat: String): Double = {
////     val matches = entityToPatterns(e)
////     val pos = promotedPatterns(cat) intersect matches
////     val precision = pos.size.toDouble / matches.size.toDouble
////     val score = precision * math.log(pos.size)
////     // println(s"${pos.size} $score")
////     score
////   }
//
//  def selectPatterns(/*n: Int,*/ category: String, patterns: Set[Int],  expandedEntities: HashSet[(Int, Double)]): Seq[(Double,Int)] = {
//    val scoredPatterns = for (p <- patterns) yield {
//      (scorePattern(p, category, expandedEntities), p)
//    }
//    val pats = scoredPatterns.toSeq
//      //.filter(_._1 > 0) // NOTE: Removing this condition as we have a metric which is not exactly PMI, hence highly correlated terms can have negative score (see test case example)
//      //.sortBy(-_._1) // NOTE: sorting outside the loop
//      //.map(_._2) // NOTE: To return the patternIDs along with their scores
//      .filterNot(p => promotedPatterns(category).contains(p._2))
//    /*HashSet() ++= */ takeNonOverlapping(pats, /*n,*/ category) //NOTE: Removing the selection of topN as this selection is done in it caller
//  }
//
//  def selectPatternsWithoutExpansion(/*n: Int,*/ category: String, patterns: Set[Int]): Seq[(Double,Int)] = {
//    val scoredPatterns = for (p <- patterns) yield {
//      (scorePatternWithoutExpansion(p, category), p)
//    }
//    val pats = scoredPatterns.toSeq
//      //.filter(_._1 > 0) // NOTE: Removing this condition as we have a metric which is not exactly PMI, hence highly correlated terms can have negative score (see test case example)
//      //.sortBy(-_._1) // NOTE: sorting outside the loop
//      //.map(_._2) // NOTE: To return the patternIDs along with their scores
//      .filterNot(p => promotedPatterns(category).contains(p._2))
//    /*HashSet() ++= */ takeNonOverlapping(pats, /*n,*/ category) //NOTE: Removing the selection of topN as this selection is done in it caller
//  }
//
//  def takeNonOverlapping(xs: Seq[(Double,Int)], /*n: Int,*/ category: String): Seq[(Double,Int)] = {
//    val ids = new ArrayBuffer[(Double,Int)]
//    val pats = new ArrayBuffer[String]
//    var i = 0
//    while (/*ids.size < n &&*/ i < xs.size) {
//      val x = xs(i)
//      val p = patternLexicon(x._2)
//      val similarPatternSeen = pats.exists(containsOrContained(_, p)) || promotedPatterns(category).map(patternLexicon.apply).exists(containsOrContained(_, p))
//      if (!similarPatternSeen) {
//        ids += x
//        pats += p
//      }
//      i += 1
//    }
//    ids.toSeq
//  }
//
//  def containsOrContained(x: String, y: String): Boolean = {
//    x.indexOfSlice(y) >= 0 || y.indexOfSlice(x) >= 0
//  }
//
////   def selectEntities(n: Int, category: String, entities: Set[Int]): Set[Int] = {
////     val scoredPatterns = for (e <- entities) yield {
////       val entityCount = entityCounts(e)
////       val dividend = promotedPatterns(category).map(p => entityPatternCount(e, p)).sum
////       val score = dividend / entityCount
////       (score, e)
////     }
////     scoredPatterns.toSeq.sortBy(_._1).map(_._2).filterNot(promotedEntities(category).contains).toSet
////   }
//
//  def selectEntities(n: Int, category: String, entities: Set[Int]): Set[Int] = {
//    val scoredPatterns = for (e <- entities) yield {
//      (scoreEntity(e, category), e)
//    }
//    scoredPatterns.toSeq
//      .filter(_._1 > 0)
//      .sortBy(-_._1)
//      .map(_._2)
//      .filterNot(promotedEntities(category).contains)
//      .take(n)
//      .toSet
//    // scoredPatterns.toSeq.sortBy(_._1).filter(_._1 > 0).map(_._2).filterNot(promotedEntities(category).contains).take(n).toSet
//  }
//
//  class Index(val index: Map[Int, Set[Int]]) {
//    def apply(id: Int): Set[Int] = index.getOrElse(id, Set.empty[Int])
//  }
//
//  object Index {
//    def loadFrom(file: File): Index = {
//      val source = Source.fromFile(file)
//      val entries = (for (line <- source.getLines) yield {
//        val Array(entity, patterns) = line.split("\t")
//        entity.toInt -> patterns.splitOnWhitespace.map(_.toInt).toSet
//      }).toList
//      source.close()
//      new Index(entries.toMap)
//    }
//  }
//
//  class Counts(val counts: Map[Int, Int]) {
//    def apply(id: Int): Double = counts.getOrElse(id, 0).toDouble
//  }
//
//  object Counts {
//    def loadFrom(file: File): Counts = {
//      val source = Source.fromFile(file)
//      val entries = (for (line <- source.getLines) yield {
//        val Array(id, count) = line.splitOnWhitespace
//        id.toInt -> count.toInt
//      }).toList
//      source.close()
//      new Counts(entries.toMap)
//    }
//  }
//
//  class Counts2(val counts: Map[(Int, Int), Int]) {
//    def apply(id1: Int, id2: Int): Double = counts.getOrElse((id1, id2), 0).toDouble
//  }
//
//  object Counts2 {
//    def loadFrom(file: File): Counts2 = {
//      val source = Source.fromFile(file)
//      val entries = (for (line <- source.getLines) yield {
//        val Array(id1, id2, count) = line.splitOnWhitespace
//        (id1.toInt, id2.toInt) -> count.toInt
//      }).toList
//      source.close()
//      new Counts2(entries.toMap)
//    }
//  }
//
//  def readMap(file: File): Map[String, String] = {
//    val source = Source.fromFile(file)
//    val items = for (line <- source.getLines()) yield {
//      val Array(k, v) = line.split("\t")
//      (k, v)
//    }
//    val map = items.toMap
//    source.close()
//    map
//  }
//}
//
//// OLD SEEDS
//
////  seeds from "Unsupervised discovery of negative categories in lexicon bootstrapping"
////  val seeds = Map(
////    "ANTIBODY" -> Seq("MAb", "IgG", "IgM", "rituximab", "infliximab"),
////    "CELL" -> Seq("RBC", "HUVEC", "BAEC", "VSMC", "SMC"),
////    "CELL-LINE" -> Seq("PC12", "CHO", "HeLa", "Jurkat", "COS"),
////    "DISEASE" -> Seq("asthma", "hepatitis", "tuberculosis", "HIV", "malaria"),
////    "DRUG" -> Seq("acetylcholine", "carbachol", "heparin", "penicillin", "tetracyclin"),
////    "PROCESS" -> Seq("kinase", "ligase", "acetyltransferase", "helicase", "binding"),
////    "MUTATION" -> Seq("Leiden", "C677T", "C282Y", "35delG"), // there is one called "null" in the paper, we omitted it
////    "PROTEIN" -> Seq("p53", "actin", "collagen", "albumin", "IL-6"),
////    "SYMPTOM" -> Seq("anemia", "fever", "hypertension", "hyperglycemia", "cough"),
////    "TUMOR" -> Seq("lymphoma", "sarcoma", "melanoma", "osteosarcoma", "neuroblastoma")
////  )
//
////  Seed randomly choosen by eyballing the CoNLL 03 train dataset
////  val seeds = Map (
////     "PER" -> Seq("Yasushi Akashi", "Franz Fischler", "Hendrix", "S. Campbell"),
////     "ORG" -> Seq("European Commission", "Paribas", "EU"),
////     "LOC" -> Seq("LONDON", "U.S.", "Florida", "Ross County" )
////  )
//
////  To select the entities that occur most frequently in patterns // Seed Selection
////  val y = entityToPatterns.index.toSeq.sortBy(-_._2.size)
////  val z = y.take(200)
////  for(x <- z) {
////    println(s"${entityLexicon(x._1)} -> (sz) ${x._2.size}")
////  }
//
//
////  Seeds selected as a subset of sorting 'entityPatterns.invertedIndex' in decreasing order of size
////  0. Seeds for N categories
////    val seeds = Map (
////     "PER" -> Seq("Clinton", "Boris Yeltsin", "Mother Teresa", "Bob Dole"),
////     "ORG" -> Seq("U.N.","Reuters", "ISS Inc", "NATO","White House" ),
////     "LOC" -> Seq("U.S.", "Britain", "London", "Hong Kong", "South Africa" )
////    )
//
///***
// * Process
// *
// *
//(oxidation,12,Process)
//(isogeometric analysis,8,Process)
//(chemical reaction,7,Process)
//(SNG,6,Process)
//(TIs,6,Process)
//(AFM,6,Process)
//(simulation,6,Process)
//(BEM,6,Process)
//(SW - SVR,6,Process)
//(solar radiation,5,Process)
//*/
//
///***
//*
//* Material
//*
//*
//(water,14,Material)
//(oxide,10,Material)
//(simulations,9,Material)
//(liquid,9,Material)
//(surface,7,Material)
//(solution,7,Material)
//(experimental data,6,Material)
//(algorithm,6,Material)
//(carbon,6,Material)
//(Newman – Penrose constants,6,Material)
//
//*/
//
///****
//*
//* Task
//*
//*
//(D - flat directions,5,Task)
//(microscopy,5,Task)
//(energy – momentum scheme,4,Task)
//(GFRFs,3,Task)
//(radiative spacetimes,3,Task)
//(non - autonomous systems,3,Task)
//(polymer crystallization,3,Task)
//(excitation energies,3,Task)
//(Design semantics,3,Task)
//(AAMM,3,Task)
//*/