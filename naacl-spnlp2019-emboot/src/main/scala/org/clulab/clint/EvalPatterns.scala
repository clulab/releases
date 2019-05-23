package org.clulab.clint

import java.io._
import scala.io.Source
import com.typesafe.scalalogging.LazyLogging
import com.typesafe.config.ConfigFactory
import ai.lum.common.ConfigUtils._

object EvalPatterns extends App with LazyLogging {

  import Bootstrap._

  val config = ConfigFactory.load()
  val indexDir = config[File]("clint.index-dir")

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

  val scoredPatterns = readPatterns("/Users/marcov/Desktop/FINAL_RESULTS/sgd_ranked_patterns_without_margin.txt")

  for (pat <- scoredPatterns.keys) {
    val patId = patternLexicon(pat)
    val entIds = patternToEntities(patId)
    val ents = entIds.map(entityLexicon.apply)
    val counts = entIds.map(entityPatternCount(_, patId))
    for ((e, c) <- ents zip counts) {
      println(s"$e\t$c\t${fmtScores(scoredPatterns(pat))}")
    }
  }
  


  def fmtScores(scores: Map[String, Double]): String = {
    scores.toSeq.map(kv => s"${kv._1}:${kv._2}").mkString(" ")
  }

  def encodePattern(pattern: String): String = {
    pattern.split("\\s+").map { w =>
      if (w == "@ENTITY") "@" else wordLexicon(w).toString
    }.mkString(" ")
  }


  def readPatterns(filename: String) = {
    val source = Source.fromFile(filename)
    val scoredPatterns = source.getLines().flatMap { line =>
      // ['PER', 'LOC', 'ORG', 'MISC']
      val Array(label, freq, scoresStr, patternStr) = line.split("\t")
      val pattern = patternStr.trim
      if (pattern == "</s>") {
        None
      } else {
        val scores = scoresStr.trim.drop(1).dropRight(1).trim.split("\\s+").map(_.toDouble)
        Some(encodePattern(pattern) -> Map("PER" -> scores(0), "LOC" -> scores(1), "ORG" -> scores(2), "MISC" -> scores(3)))
      }
    }
    val res = scoredPatterns.toMap
    source.close()
    res
  }

}
