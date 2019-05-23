//package org.clulab.clint
//
//import java.io._
//import scala.io.Source
//import com.typesafe.scalalogging.LazyLogging
//import com.typesafe.config.ConfigFactory
//import ai.lum.common.ConfigUtils._
//
//object BuildRelationPatternLexicon extends App with LazyLogging {
//
//  val config = ConfigFactory.load()
//  val indexDir = config[File]("clint.index-dir")
//
//  indexRelationPatterns(new File(indexDir, "relationPatterns.dump"))
//
//  def indexRelationPatterns(file: File): Unit = {
//    val dump = Source.fromFile(file)
//    val lexicon = new LexiconBuilder
//    for (line <- dump.getLines) {
//      val pattern = Pattern(line)
//      lexicon.add(pattern.withoutEntityIds)
//    }
//    dump.close()
//    writeFile(new File(indexDir, "relationPatterns.total"), lexicon.totalCount.toString)
//    lexicon.keepIfAbove(1) // add to config file
//    lexicon.saveTo(new File(indexDir, "relationPatterns.lexicon"))
//    lexicon.writeCounts(new File(indexDir, "relationPatterns.counts"), -1)
//  }
//
//  def writeFile(file: File, string: String): Unit = {
//    val writer = new BufferedWriter(new FileWriter(file))
//    writer.write(string)
//    writer.close()
//  }
//
//}
