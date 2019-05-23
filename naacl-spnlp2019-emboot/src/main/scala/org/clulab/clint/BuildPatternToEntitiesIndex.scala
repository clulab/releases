package org.clulab.clint

import java.io._
import scala.io.Source
import com.typesafe.scalalogging.LazyLogging
import com.typesafe.config.ConfigFactory
import ai.lum.common.ConfigUtils._

object BuildPatternToEntitiesIndex extends App with LazyLogging {
  val config = ConfigFactory.load()
  val indexDir = config[File]("clint.index-dir")

  logger.info("loading pattern lexicon")
  val patternLexicon = LexiconBuilder.loadLexemeToIndex(new File(indexDir, "patterns.lexicon"))

  logger.info("building index")
  val patternToEntitiesIndex = new InvertedIndex

  val dump = Source.fromFile(new File(indexDir, "entityPatterns.dump"))
  for (line <- dump.getLines()) {
    val pattern = Pattern(line)
    patternLexicon.get(pattern.withoutEntityIds) match {
      case None => ()
      case Some(patternId) =>
        val Seq(entityId) = pattern.entityIds
        patternToEntitiesIndex.add(patternId, entityId)
    }
  }

  logger.info("writing files")
  patternToEntitiesIndex.saveTo(new File(indexDir, "patternToEntities.index"))
}
