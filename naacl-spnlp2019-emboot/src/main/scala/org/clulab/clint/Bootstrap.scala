package org.clulab.clint

import java.io._
import scala.io._
import scala.collection.mutable.{ HashMap, HashSet, ArrayBuffer }
import com.typesafe.scalalogging.LazyLogging
import com.typesafe.config.ConfigFactory
import ai.lum.common.ConfigUtils._
import ai.lum.common.StringUtils._

object Bootstrap extends App with LazyLogging {

  val config = ConfigFactory.load()
  val indexDir = config[File]("clint.index-dir")

  val numEpochs = 10

  // seeds from "Unsupervised discovery of negative categories in lexicon bootstrapping"
//  val seeds = Map(
//    "ANTIBODY" -> Seq("MAb", "IgG", "IgM", "rituximab", "infliximab"),
//    "CELL" -> Seq("RBC", "HUVEC", "BAEC", "VSMC", "SMC"),
//    "CELL-LINE" -> Seq("PC12", "CHO", "HeLa", "Jurkat", "COS"),
//    "DISEASE" -> Seq("asthma", "hepatitis", "tuberculosis", "HIV", "malaria"),
//    "DRUG" -> Seq("acetylcholine", "carbachol", "heparin", "penicillin", "tetracyclin"),
//    "PROCESS" -> Seq("kinase", "ligase", "acetyltransferase", "helicase", "binding"),
//    "MUTATION" -> Seq("Leiden", "C677T", "C282Y", "35delG"), // there is one called "null" in the paper, we omitted it
//    "PROTEIN" -> Seq("p53", "actin", "collagen", "albumin", "IL-6"),
//    "SYMPTOM" -> Seq("anemia", "fever", "hypertension", "hyperglycemia", "cough"),
//    "TUMOR" -> Seq("lymphoma", "sarcoma", "melanoma", "osteosarcoma", "neuroblastoma")
//  )

    // Seed randomly choosen by eyballing the CoNLL 03 train dataset
//  val seeds = Map (
//     "PER" -> Seq("Yasushi Akashi", "Franz Fischler", "Hendrix", "S. Campbell"),
//     "ORG" -> Seq("European Commission", "Paribas", "EU"),
//     "LOC" -> Seq("LONDON", "U.S.", "Florida", "Ross County" )
//  )
  
    // Seeds selected as a subset of sorting 'entityPatterns.invertedIndex' in decreasing order of size
    val seeds = Map (
     "PER" -> Seq("Clinton", "Boris Yeltsin", "Mother Teresa", "Bob Dole"),
     "ORG" -> Seq("U.N.","Reuters", "ISS Inc", "NATO","White House" ),
     "LOC" -> Seq("U.S.", "Britain", "London", "Hong Kong", "South Africa" )
    )
    
  // an array with all category names
  val categories = seeds.keys.toArray
  val numCategories = categories.size

  // maps from category name to (entity or pattern) ids
  val promotedEntities = HashMap.empty[String, HashSet[Int]]
  val promotedPatterns = HashMap.empty[String, HashSet[Int]]

  val newlyPromotedEntities = HashMap.empty[String, Set[Int]]
  val newlyPromotedPatterns = HashMap.empty[String, Set[Int]]

  logger.info("loading data")
  val wordLexicon = LexiconBuilder.loadLexicon(new File(indexDir, "word.lexicon"))
  val entityLexicon = LexiconBuilder.loadLexicon(new File(indexDir, "entity.lexicon"))
  val normEntities = readMap(new File(indexDir, "entity.normalized"))
  val patternLexicon = LexiconBuilder.loadLexicon(new File(indexDir, "entityPatterns.lexicon"))
  val entityToPatterns = Index.loadFrom(new File(indexDir, "entityPatterns.invertedIndex"))
  val patternToEntities = Index.loadFrom(new File(indexDir, "entityPatterns.forwardIndex"))
  val entityCounts = Counts.loadFrom(new File(indexDir, "entity.counts"))
  val patternCounts = Counts.loadFrom(new File(indexDir, "entityPatterns.counts"))
  val entityPatternCount = Counts2.loadFrom(new File(indexDir, "entityId.entityPatternId.counts"))
  val totalEntityCount = entityCounts.counts.values.sum
  val totalPatternCount = patternCounts.counts.values.sum

  val goldLabelsFile = "/Users/ajaynagesh/Research/code/research/clint/data/conll.goldlabels"
  val goldLabels = Source.fromFile(goldLabelsFile).getLines.map {
  l => 
    val tmp = l.split("\t")
    val k = tmp(0)
    val v = tmp.tail
    k -> v }
  .toMap
  
  logger.info("promoting seeds")
  for {
    cat <- categories
    ent <- seeds(cat)
    norm <- normEntities.get(ent)
    id <- entityLexicon.get(norm)
  } promotedEntities.getOrElseUpdate(cat, HashSet.empty[Int]) += id

  for (cat <- categories) promotedPatterns(cat) = HashSet.empty[Int]

  printSeeds()

  val bwAcc = new BufferedWriter(new FileWriter(new File("/Users/ajaynagesh/Research/code/research/clint/data/accuraciesBaseline.txt")))

  
  logger.info("bootstrapping")
  var k = 5
  for (epoch <- 1 to numEpochs) {
    for (cat <- categories) {
      // patterns
      val candidatePatterns = getPatterns(promotedEntities(cat).toSet)
      val preselectedPatterns = preselectPatterns(cat, candidatePatterns)
      val selectedPatterns = selectPatterns(k, cat, preselectedPatterns)
      // val selectedPatterns = selectPatterns(k, cat, candidatePatterns)
      newlyPromotedPatterns(cat) = selectedPatterns -- promotedPatterns(cat)
      promotedPatterns(cat) ++= selectedPatterns
      // entities
      val candidateEntities = getEntities(promotedPatterns(cat).toSet)
      // val preselectedEntities = preselectEntities(cat, candidateEntities)
      // val selectedEntities = selectEntities(k, cat, preselectedEntities)
      val selectedEntities = selectEntities(k, cat, candidateEntities)
      newlyPromotedEntities(cat) = selectedEntities -- promotedEntities(cat)
      promotedEntities(cat) ++= selectedEntities
    }
    // k += 5
//    printReport(epoch)
    val stats = computeAccuracy
    println("Accuracy")
    println("---------------------------------------------------")
    println(s"${epoch}\t${stats._1}\t${stats._2}\t${stats._3}")
    bwAcc.write(s"${epoch}\t${stats._1}\t${stats._2}\t${stats._3}\n")
    println("---------------------------------------------------")
  }
  
  bwAcc.close()

  def computeAccuracy() : (Double, Double, Int) = {
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
    (avgPrecision, avgRecall, sumSize)
    
  }
  
  def printSeeds(): Unit = {
    println("Seeds")
    for (cat <- categories) {
      println(s"$cat entities:")
      for (e <- seeds(cat)) {
        println(e)
      }
      println()
    }
    println("=" * 70)
    println()
  }

  def printReport(epoch: Int): Unit = {
    println(s"Bootstrapping epoch $epoch")
    for (cat <- categories) {
      println(s"$cat entities:")
      for (e <- newlyPromotedEntities(cat).toSeq.sortBy(e => scoreEntity(e, cat))) {
        println(scoreEntity(e, cat) + "\t" + entityLexicon(e))
      }
      println(s"\n$cat patterns:")
      for (p <- newlyPromotedPatterns(cat).toSeq.sortBy(p => scorePattern(p, cat))) {
        val words = patternLexicon(p).splitOnWhitespace.map {
          case "@" => "@"
          case n => wordLexicon(n.toInt)
        }
        println(scorePattern(p, cat) + "\t" + words.mkString(" "))
      }
      println()
    }
    println("=" * 70)
    println()
  }

  def getPatterns(entities: Set[Int]): Set[Int] = {
    entities.flatMap(e => entityToPatterns(e))
  }

  def getEntities(patterns: Set[Int]): Set[Int] = {
    patterns.flatMap(p => patternToEntities(p))
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

  def preselectEntities(category: String, entities: Set[Int]): Set[Int] = {
    val results = for (e <- entities) yield {
      val candidates = entityToPatterns(e)
      val patterns = promotedPatterns(category)
      val matches = candidates intersect patterns
      val accuracy = matches.size.toDouble / candidates.size.toDouble
      if (matches.size == 1) None
      // else if (accuracy < 0.5) None
      else Some(e)
    }
    results.flatten
  }

//   def selectPatterns(n: Int, category: String, patterns: Set[Int]): Set[Int] = {
//     val scoredPatterns = for (p <- patterns) yield {
//       val patternCount = patternCounts(p)
//       val dividend = promotedEntities(category).map(e => entityPatternCount(e, p)).sum
//       val score = dividend / patternCount
//       (score, p)
//     }
//     scoredPatterns.toSeq.sortBy(_._1).map(_._2).filterNot(promotedPatterns(category).contains).toSet
//   }

//   def scorePattern(p: Int, cat: String): Double = {
//     val entities = patternToEntities(p)
//     val patternTotalCount: Double = patternCounts(p)
//     val positiveCount: Double = entities
//       .filter(promotedEntities(cat).contains) // only the ones in the pool
//       .map(entityPatternCount(_, p)) // count them
//       .sum // total sum
//     (positiveCount / patternTotalCount) * math.log(patternTotalCount)
//   }

  // SCORES

   def scorePattern(p: Int, cat: String): Double = riloffPattern(p, cat)
   def scoreEntity(e: Int, cat: String): Double = riloffEntity(e, cat)

//   def scorePattern(p: Int, cat: String): Double = collinsPattern(p, cat)
//   def scoreEntity(e: Int, cat: String): Double = collinsEntity(e, cat)

//   def scorePattern(p: Int, cat: String): Double = chi2Pattern(p, cat)
//   def scoreEntity(e: Int, cat: String): Double = chi2Entity(e, cat)

//  def scorePattern(p: Int, cat: String): Double = mutualInformationPattern(p, cat)
//  def scoreEntity(e: Int, cat: String): Double = mutualInformationEntity(e, cat)

  // number of times pattern `p` matches an entity with category `cat`
  def positiveEntityCounts(p: Int, cat: String): Double = {
    patternToEntities(p)
      .filter(promotedEntities(cat).contains) // only the ones in the pool
      .map(entityPatternCount(_, p)) // count them
      .sum // total count
  }

  def positivePatternCounts(e: Int, cat: String): Double = {
    entityToPatterns(e)
      .filter(promotedPatterns(cat).contains) // only the ones in the pool
      .map(entityPatternCount(e, _)) // count them
      .sum // total count
  }

  def countEntityMentions(cat: String): Double = {
    promotedEntities(cat).map(entityCounts(_)).sum
  }

  def countPatternMentions(cat: String): Double = {
    promotedPatterns(cat).map(patternCounts(_)).sum
  }

  // precision of pattern `p` in the set of entities labeled with category `cat`
  def precisionPattern(p: Int, cat: String): Double = {
    val total: Double = patternCounts(p)
    val positive: Double = positiveEntityCounts(p, cat)
    positive / total
  }

  def precisionEntity(e: Int, cat: String): Double = {
    val total: Double = entityCounts(e)
    val positive: Double = positivePatternCounts(e, cat)
    positive / total
  }

  def riloffPattern(p: Int, cat: String): Double = {
    val prec = precisionPattern(p, cat)
    if (prec > 0) {
      prec * math.log(patternCounts(p))
      // prec * math.log(positiveEntityCounts(p, cat))
    } else {
      0
    }
  }

  def riloffEntity(e: Int, cat: String): Double = {
    val prec = precisionEntity(e, cat)
    if (prec > 0) {
      prec * math.log(entityCounts(e))
      // prec * math.log(positivePatternCounts(e, cat))
    } else {
      0
    }
  }

  def collinsPattern(p: Int, cat: String): Double = {
    val prec = precisionPattern(p, cat)
    if (prec > 0.95) {
      patternCounts(p)
    } else {
      0
    }
  }

  def collinsEntity(e: Int, cat: String): Double = {
    val prec = precisionEntity(e, cat)
    if (prec > 0.95) {
      entityCounts(e)
    } else {
      0
    }
  }

  def chi2Pattern(p: Int, cat: String): Double = {
    val prec = precisionPattern(p, cat)
    if (prec > 0.5) {
      val n: Double = totalEntityCount
      val a: Double = positiveEntityCounts(p, cat)
      val b: Double = patternCounts(p) - positiveEntityCounts(p, cat)
      val c: Double = countEntityMentions(cat) - positiveEntityCounts(p, cat)
      val d: Double = n - countEntityMentions(cat) - patternCounts(p) + positiveEntityCounts(p, cat)
      val numerator = n * math.pow(a * d - c * b, 2)
      val denominator = (a + c) * (b + d) * (a + b) * (c + d)
      numerator / denominator
    } else {
      0
    }
  }

  def chi2Entity(e: Int, cat: String): Double = {
    val prec = precisionEntity(e, cat)
    if (prec > 0.5) {
      val n: Double = totalPatternCount
      val a: Double = positivePatternCounts(e, cat)
      val b: Double = entityCounts(e) - positivePatternCounts(e, cat)
      val c: Double = countPatternMentions(cat) - positivePatternCounts(e, cat)
      val d: Double = n - countPatternMentions(cat) - entityCounts(e) + positivePatternCounts(e, cat)
      val numerator = n * math.pow(a * d - c * b, 2)
      val denominator = (a + c) * (b + d) * (a + b) * (c + d)
      numerator / denominator
    } else {
      0
    }
  }

  def mutualInformationPattern(p: Int, cat: String): Double = {
    val prec = precisionPattern(p, cat)
    if (prec > 0.5) {
      val n: Double = totalEntityCount
      val a: Double = positiveEntityCounts(p, cat)
      val b: Double = patternCounts(p) - positiveEntityCounts(p, cat)
      val c: Double = countEntityMentions(cat) - positiveEntityCounts(p, cat)
      math.log((n * a) / ((a + c) * (a + b)))
    } else {
      0
    }
  }

  def mutualInformationEntity(e: Int, cat: String): Double = {
    val prec = precisionEntity(e, cat)
    if (prec > 0.5) {
      val n: Double = totalPatternCount
      val a: Double = positivePatternCounts(e, cat)
      val b: Double = entityCounts(e) - positivePatternCounts(e, cat)
      val c: Double = countPatternMentions(cat) - positivePatternCounts(e, cat)
      math.log((n * a) / ((a + c) * (a + b)))
    } else {
      0
    }
  }



//   def scorePattern(p: Int, cat: String): Double = {
//     val matches = patternToEntities(p)
//     val pos = promotedEntities(cat) intersect matches
//     val precision = pos.size.toDouble / matches.size.toDouble
//     val score = precision * math.log(pos.size)
//     score
//   }

  def patternString(p: Int): String = {
    val words = patternLexicon(p).splitOnWhitespace.map {
      case "@" => "@"
      case n => wordLexicon(n.toInt)
    }
    words.mkString(" ")
  }

//   def scoreEntity(e: Int, cat: String): Double = {
//     val patterns = entityToPatterns(e)
//     val entityTotalCount: Double = entityCounts(e)
//     val positiveCount: Double = patterns
//       .filter(promotedPatterns(cat).contains) // only the ones in the pool
//       .map(entityPatternCount(e, _)) // count them
//       .sum // total sum
//     (positiveCount / entityTotalCount) * math.log(entityTotalCount)
//   }

//   def scoreEntity(e: Int, cat: String): Double = {
//     val matches = entityToPatterns(e)
//     val pos = promotedPatterns(cat) intersect matches
//     val precision = pos.size.toDouble / matches.size.toDouble
//     val score = precision * math.log(pos.size)
//     // println(s"${pos.size} $score")
//     score
//   }

  def selectPatterns(n: Int, category: String, patterns: Set[Int]): Set[Int] = {
    val scoredPatterns = for (p <- patterns) yield {
      (scorePattern(p, category), p)
    }
    val pats = scoredPatterns.toSeq
      .filter(_._1 > 0)
      .sortBy(-_._1)
      .map(_._2)
      .filterNot(promotedPatterns(category).contains)
    takeNonOverlapping(pats, n, category).toSet
  }

  def takeNonOverlapping(xs: Seq[Int], n: Int, category: String): Seq[Int] = {
    val ids = new ArrayBuffer[Int]
    val pats = new ArrayBuffer[String]
    var i = 0
    while (ids.size < n && i < xs.size) {
      val x = xs(i)
      val p = patternLexicon(x)
      val similarPatternSeen = pats.exists(containsOrContained(_, p)) || promotedPatterns(category).map(patternLexicon.apply).exists(containsOrContained(_, p))
      if (!similarPatternSeen) {
        ids += x
        pats += p
      }
      i += 1
    }
    ids.toSeq
  }

  def containsOrContained(x: String, y: String): Boolean = {
    x.indexOfSlice(y) >= 0 || y.indexOfSlice(x) >= 0
  }

//   def selectEntities(n: Int, category: String, entities: Set[Int]): Set[Int] = {
//     val scoredPatterns = for (e <- entities) yield {
//       val entityCount = entityCounts(e)
//       val dividend = promotedPatterns(category).map(p => entityPatternCount(e, p)).sum
//       val score = dividend / entityCount
//       (score, e)
//     }
//     scoredPatterns.toSeq.sortBy(_._1).map(_._2).filterNot(promotedEntities(category).contains).toSet
//   }

  def selectEntities(n: Int, category: String, entities: Set[Int]): Set[Int] = {
    val scoredPatterns = for (e <- entities) yield {
      (scoreEntity(e, category), e)
    }
    scoredPatterns.toSeq
      .filter(_._1 > 0)
      .sortBy(-_._1)
      .map(_._2)
      .filterNot(promotedEntities(category).contains)
      .take(n)
      .toSet
    // scoredPatterns.toSeq.sortBy(_._1).filter(_._1 > 0).map(_._2).filterNot(promotedEntities(category).contains).take(n).toSet
  }

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
