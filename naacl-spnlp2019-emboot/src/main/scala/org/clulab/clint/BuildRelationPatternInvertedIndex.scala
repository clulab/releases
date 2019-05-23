//package org.clulab.clint
//
//import java.io._
//import scala.io.Source
//import scala.collection.mutable.{ HashMap, HashSet }
//import com.typesafe.scalalogging.LazyLogging
//import com.typesafe.config.ConfigFactory
//import ai.lum.common.ConfigUtils._
//
//object BuildRelationPatternInvertedIndex extends App with LazyLogging {
//
//  val config = ConfigFactory.load()
//  val indexDir = config[File]("clint.index-dir")
//
//  logger.info("loading relation pattern lexicon")
//  val lexicon = LexiconBuilder.loadLexemeToIndex(new File(indexDir, "relationPatterns.lexicon"))
//
//  logger.info("building inverted index")
//  val index = new InvertedIndex2
//  val dump = Source.fromFile(new File(indexDir, "relationPatterns.dump"))
//  for (line <- dump.getLines()) {
//    val pattern = Pattern(line)
//    lexicon.get(pattern.withoutEntityIds) match {
//      case None => ()
//      case Some(patternId) =>
//        val Seq(entityId1, entityId2) = pattern.entityIds
//        index.add(entityId1, entityId2, patternId)
//    }
//  }
//
//  logger.info("writing files")
//  index.saveTo(new File(indexDir, "relationPatterns.invertedIndex"))
//  index.writeCounts(new File(indexDir, "entityId.relationPatternId.counts"))
//  index.writeCounts2(new File(indexDir, "entityId.entityId.relationPatternId.counts"))
//
//  class InvertedIndex2 {
//
//    val index = HashMap.empty[Int, HashSet[Int]]
//    val counts = HashMap.empty[(Int, Int), Int] withDefaultValue 0
//    val counts2 = HashMap.empty[(Int, Int, Int), Int] withDefaultValue 0
//
//    def add(entity1Id: Int, entity2Id: Int, patternId: Int): Unit = {
//      index.getOrElseUpdate(entity1Id, HashSet.empty[Int]) += patternId
//      index.getOrElseUpdate(entity2Id, HashSet.empty[Int]) += patternId
//      counts((entity1Id, patternId)) += 1
//      counts((entity2Id, patternId)) += 1
//      counts2((entity1Id, entity2Id, patternId)) += 1
//    }
//
//    def saveTo(file: File): Unit = {
//      val writer = new BufferedWriter(new FileWriter(file))
//      for (entityId <- index.keys) {
//        val patternIds = index(entityId).mkString(" ")
//        writer.write(s"$entityId\t$patternIds\n")
//      }
//      writer.close()
//    }
//
//    def writeCounts(file: File): Unit = {
//      val writer = new BufferedWriter(new FileWriter(file))
//      for ((entityId, patternId) <- counts.keys) {
//        val count = counts((entityId, patternId))
//        writer.write(s"$entityId $patternId $count\n")
//      }
//      writer.close()
//    }
//
//    def writeCounts2(file: File): Unit = {
//      val writer = new BufferedWriter(new FileWriter(file))
//      for ((e1id, e2id, patternId) <- counts2.keys) {
//        val count = counts2((e1id, e2id, patternId))
//        writer.write(s"$e1id $e2id $patternId $count\n")
//      }
//      writer.close()
//    }
//
//  }
//
//}
