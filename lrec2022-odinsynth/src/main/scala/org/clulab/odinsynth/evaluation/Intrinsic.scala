package org.clulab.odinsynth.evaluation

import org.clulab.odinsynth._
import scala.io.StdIn.readLine
import scala.util.control.Breaks._
import ai.lum.odinson._
import scala.io.Source

object Intrinsic extends App {
  def readSpec(path: String): Pair[Document, Set[Spec]] = {
    val specsFile = Source.fromFile(path)
    val lines = specsFile.getLines.toSeq
    val doc = Document.fromJson(lines(1))
    val specs = Spec.fromString(lines(0)).toSet

    // println(doc.toJson)
    //println(specs)
    (doc, specs)
  }

  def findRule(doc: Document, specs: Set[Spec]) {
    //
    val fieldNames = Set("word", "tag", "lemma")
    //
    val searcher = new Searcher(doc, specs, fieldNames)
    // start recording time
    val t0 = System.nanoTime()
    // find rule
    val rule = searcher.findAll.iterator.next
    // stop recording time
    val t1 = System.nanoTime()
    // print
    if (rule != null) {
    println(s"found-rule<${rule.rule.pattern}>\t${(t1-t0)/1e9d}")
    } else {
      println(s"time-out-<${i}>")
    }
  }

  // read testing file
  val evalPath =
    "/data/nlp/corpora/odinsynth/data/rules100k_unrolled/test_names"
  val evalFile = Source.fromFile(evalPath)
  //
  var i = 0
  //
  breakable {
    for (l <- evalFile.getLines) {
      val ruleFiles = l.split("\t")
      //println(s"steps: ${ruleFiles(0)}")
      //println(s"specs: ${ruleFiles(1)}")
      val spec = readSpec(ruleFiles(1))
      findRule(spec._1, spec._2)

      i += 1
      if (i > 50) {
        break
      }
    }
  }

}
