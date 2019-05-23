package org.clulab.clint

import java.io._
import scala.collection.mutable.HashMap
import org.clulab.odin.Mention
import org.clulab.processors.Document
import com.typesafe.scalalogging.LazyLogging
import com.typesafe.config.ConfigFactory
import ai.lum.common.ConfigUtils._
import ai.lum.common.StringUtils._
import ai.lum.common.Serializer

object BuildLexicons extends App with LazyLogging {

  val config = ConfigFactory.load()
  val docsDir:File = config[File]("clint.docs-dir")
  val indexDir = config[File]("clint.index-dir")

  val wordLexicon = new LexiconBuilder
  val entityLexicon = new LexiconBuilder
  val candidateFinderName = config[String]("clint.candFinderToUse")

  val finder = candidateFinderName match {
    case "ontonotes" => new OracleCandidateFinderOntonotes
    case "conll" => new OracleCandidateFinder
    case "scienceie" => new OracleCandidateFinderScienceIE
  }

  println(s"docsDir : $docsDir")
  for (f <- docsDir.listFiles() if f.getName().endsWith(".ser")) {

    logger.info(s"Loading ${f.getName()}")
    val doc = Serializer.deserialize[Document](f)

    logger.info("Populating word lexicon")
    for (s <- doc.sentences; w <- s.words) {
      wordLexicon.add(w)
    }

    logger.info("Searching for entity candidates")
    val candidates = finder.findCandidates(doc).sorted

    logger.info("Populating entity lexicon")
    for (c <- candidates)
    {
      entityLexicon.add(c.text)
    }

  }

  logger.info("Writing the lexicons")
  wordLexicon.saveTo(new File(indexDir, "word.lexicon"))
  writeFile(new File(indexDir, "entity.total"), entityLexicon.totalCount.toString)
  entityLexicon.saveTo(new File(indexDir, "entity.lexicon"))
  entityLexicon.writeCounts(new File(indexDir, "entity.counts"))

  entityLexicon.writeCountsEmboot(new File(indexDir, "entity_vocabulary.emboot.txt"), false) // 2nd argument: whether it is an pattern lexicon or not

  def writeFile(file: File, string: String): Unit = {
    val writer = new BufferedWriter(new FileWriter(file))
    writer.write(string)
    writer.close()
  }

}
