package org.clulab.clint

import java.io._

import scala.io.Source
import com.typesafe.scalalogging.LazyLogging
import com.typesafe.config.ConfigFactory
import ai.lum.common.ConfigUtils._

object BuildEntityToPatternsIndex extends App with LazyLogging {

  val config = ConfigFactory.load()
  val indexDir = config[File]("clint.index-dir")

  logger.info("loading entity pattern lexicon")
  val patternLexicon = LexiconBuilder.loadLexemeToIndex(new File(indexDir, "patterns.lexicon"))

  logger.info("building inverted index")
  val entityToPatternsIndex = new InvertedIndex

  val dump = Source.fromFile(new File(indexDir, "entityPatterns.dump"))
  for (line <- dump.getLines()) {
    val pattern = Pattern(line)
    patternLexicon.get(pattern.withoutEntityIds) match {
      case None => ()
      case Some(patternId) =>
        val Seq(entityId) = pattern.entityIds
        entityToPatternsIndex.add(entityId, patternId)
    }
  }

  logger.info("writing files")
  entityToPatternsIndex.saveTo(new File(indexDir, "entityToPatterns.index"))
  entityToPatternsIndex.writeCounts(new File(indexDir, "entityId.patternId.counts"))

}