package org.clulab.clint

import java.io.{BufferedWriter, File, FileWriter}

import scala.collection.mutable.{HashMap, HashSet}
import scala.io.Source
import ai.lum.common.StringUtils._

import scala.collection.mutable

/**
  * Created by ajaynagesh on 10/3/17.
  */
class InvertedIndex {

  val index = HashMap.empty[Int, HashSet[Int]]
  val counts = HashMap.empty[(Int, Int), Int] withDefaultValue 0

  def add(entityId: Int, patternId: Int): Unit = {
    index.getOrElseUpdate(entityId, HashSet.empty[Int]) += patternId
    counts((entityId, patternId)) += 1
  }

  def saveTo(file: File): Unit = {
    val writer = new BufferedWriter(new FileWriter(file))
    for (entityId <- index.keys) {
      val patternIds = index(entityId).mkString(" ")
      writer.write(s"$entityId\t$patternIds\n")
    }
    writer.close()
  }

  def writeCounts(file: File): Unit = {
    val writer = new BufferedWriter(new FileWriter(file))
    for ((entityId, patternId) <- counts.keys) {
      val count = counts((entityId, patternId))
      writer.write(s"$entityId $patternId $count\n")
    }
    writer.close()
  }

}

object InvertedIndex {
  def loadFrom(file: File): Map[Int, Set[Int]] = {
    val source = Source.fromFile(file)
    val entries = (for (line <- source.getLines) yield {
      val Array(entity, patterns) = line.split("\t")
      entity.toInt -> patterns.splitOnWhitespace.map(_.toInt).toSet
    }).toList.toMap
    source.close()
    entries
  }
}
