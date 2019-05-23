package org.clulab.clint

import java.io.File

import com.typesafe.config.ConfigFactory
import com.typesafe.scalalogging.LazyLogging
import ai.lum.common.ConfigUtils._

import scala.io._
import java.io._

import scala.util.Random
import scala.collection.mutable

import spray.json._
import spray.json.DefaultJsonProtocol._

/**
  * Created by ajaynagesh on 6/4/17.
  */
object ConstructRandomSeeds extends App with LazyLogging {

  val config = ConfigFactory.load()

  val goldLabelsFile:String =  config[String]("clint.goldLabelsFile")
  val topK = 20
  val subsetToChoose = 10
  val randomSeedSetFileBaseName = "data/RandomSeedSet.ontonotes"

  val goldLabels = Source.fromFile(goldLabelsFile).getLines.map {
    l =>
      val tmp = l.split("\t")
      val k = tmp(0)
      val v = tmp.tail
      k -> v }
    .toMap

  val labels = goldLabels.keySet.toArray

  //  var perMap = new scala.collection.mutable.HashMap[String, Int]()
  //  var orgMap = new scala.collection.mutable.HashMap[String, Int]()
  //  var locMap = new scala.collection.mutable.HashMap[String, Int]()
  //  var miscMap = new scala.collection.mutable.HashMap[String, Int]()
  //  for (l <- goldLabels("PER")) {
  //    val cnt = perMap.getOrElse(l,0)
  //    perMap.put(l,cnt+1)
  //  }
  //  for (l <- goldLabels("ORG")) {
  //    val cnt = orgMap.getOrElse(l,0)
  //    orgMap.put(l,cnt+1)
  //  }
  //  for (l <- goldLabels("LOC")) {
  //    val cnt = locMap.getOrElse(l,0)
  //    locMap.put(l,cnt+1)
  //  }
  //  for (l <- goldLabels("MISC")) {
  //    val cnt = miscMap.getOrElse(l,0)
  //    miscMap.put(l,cnt+1)
  //  }

  val labelMapArr = for(l <- labels) yield {
    val lmap = new mutable.HashMap[String, Int]()
    (l, lmap)
  }

  for((l, lmap) <- labelMapArr){
    for (e <- goldLabels(l)) {
      val cnt = lmap.getOrElse(e,0)
      lmap.put(e, cnt+1)
    }
  }

  val topKArr = for((l,lmap) <- labelMapArr) yield {
    (l, lmap.map(x => (x._1,x._2)).toArray.sortBy(- _._2).take(topK).toMap.map(_._1).toArray)
  }

  val randomSetArr = for(i <- 1 to 5) yield {
    val r = new Random(i)
    val randSet = for ((lbl,topKele) <- topKArr) yield {
      (lbl, r.shuffle(topKele.toSeq).take(subsetToChoose) )
    }
    randSet.toMap
  }

  for (i <- 0 to randomSetArr.size-1){
    val bw = new FileWriter(new File(s"${randomSeedSetFileBaseName}${i}"))
    bw.write(randomSetArr(i).toJson.prettyPrint)
    bw.close
  }


  //  val topKper = perMap.map(x => (x._1,x._2)).toArray.sortBy(- _._2).take(topK)
//  val topKorg = orgMap.map(x => (x._1,x._2)).toArray.sortBy(- _._2).take(topK)
//  val topKloc = locMap.map(x => (x._1,x._2)).toArray.sortBy(- _._2).take(topK)
//  val topKmisc = miscMap.map(x => (x._1,x._2)).toArray.sortBy(- _._2).take(topK)

//  val bw = new FileWriter(new File(top20entsFile))
//  bw.write(s"PER\t${topKper.map(x => x._1).mkString("\t")}\n")
//  bw.write(s"ORG\t${topKorg.map(x => x._1).mkString("\t")}\n")
//  bw.write(s"LOC\t${topKloc.map(x => x._1).mkString("\t")}\n")
//  bw.write(s"MISC\t${topKmisc.map(x => x._1).mkString("\t")}\n")
//  bw.close

//  val perarr = topKper.toMap.map(_._1).toArray
//  val orgarr = topKorg.toMap.map(_._1).toArray
//  val locarr = topKloc.toMap.map(_._1).toArray
//  val miscarr = topKmisc.toMap.map(_._1).toArray


//  val randomSetArr = for(i <- 1 to 5) yield {
//    val r = new Random(i)
//    val perSet = r.shuffle(perarr.toSeq).take(subsetToChoose)
//    val orgSet = r.shuffle(orgarr.toSeq).take(subsetToChoose)
//    val locSet = r.shuffle(locarr.toSeq).take(subsetToChoose)
//    val miscSet = r.shuffle(miscarr.toSeq).take(subsetToChoose)
//    Array(perSet, orgSet, locSet, miscSet)
//  }

//  val T = Source.fromFile("RandomSeedSet.0").getLines().mkString("\n")
//  val jsonT = T.parseJson
//  println(s"Hello world: ${jsonT.prettyPrint}")
//  val X = jsonT.convertTo[Map[String,Seq[String]]]

}
