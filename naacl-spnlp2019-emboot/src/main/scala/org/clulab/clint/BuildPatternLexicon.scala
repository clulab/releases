package org.clulab.clint

import java.io._

import scala.io.Source
import com.typesafe.scalalogging.LazyLogging
import com.typesafe.config.ConfigFactory
import ai.lum.common.ConfigUtils._

object BuildPatternLexicon extends App with LazyLogging {

  val config = ConfigFactory.load()
  val indexDir = config[File]("clint.index-dir")

  logger.info("Loading the entityPatterns.dump file ")
  val dump = Source.fromFile(new File(indexDir, "entityPatterns.dump"))

  // create the patterns.lexicon, patterns.counts and patterns.total files
  logger.info("Creating the patterns.lexicon")
  val lexicon = new LexiconBuilder
  for (line <- dump.getLines) {
    val pattern = Pattern(line)
    val patternId = lexicon.add(pattern.withoutEntityIds)
  }
  dump.close()

  writeFile(new File(indexDir, "patterns.total"), lexicon.totalCount.toString)
  lexicon.saveTo(new File(indexDir, "patterns.lexicon"))
  lexicon.writeCounts(new File(indexDir, "patterns.counts"))

  lexicon.writeCountsEmboot(new File(indexDir, "pattern_vocabulary_emboot.txt"), true) // 2nd argument: whether it is an pattern lexicon or not

  def writeFile(file: File, string: String): Unit = {
    val writer = new BufferedWriter(new FileWriter(file))
    writer.write(string)
    writer.close()
  }

}
