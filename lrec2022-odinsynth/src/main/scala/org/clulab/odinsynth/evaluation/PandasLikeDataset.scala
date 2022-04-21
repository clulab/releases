package org.clulab.odinsynth.evaluation

import org.clulab.odinsynth.using
import org.clulab.odinsynth.Spec

import org.clulab.processors.fastnlp.FastNLPProcessor

import ai.lum.odinson.ExtractorEngine
import ai.lum.odinson.extra.ProcessorsUtils

import java.io.PrintWriter
import java.io.File

import scala.io.Source
import scala.collection.mutable
import scala.collection
import scala.util.Random
import java.nio.file.Path
import java.nio.file.Paths
import java.nio.file.StandardCopyOption
import java.nio.file.Files
import org.clulab.odinsynth.evaluation.tacred.PatternDirection
import org.clulab.odinsynth.evaluation.tacred.SubjObjDirection
import org.clulab.odinsynth.evaluation.tacred.ObjSubjDirection

/**
  * Reads a csv or tsv file, potentially with header and single index
  * The access is done based on the line number
  * This class also has some functionality tied with TACRED (e.g. "subj_start", etc)
  *
  * @param lines a sequence of maps
  *              the map is from headerName to value
  *              for example: lines.head("pattern")
  */
case class PandasLikeDataset(val lines: IndexedSeq[Map[String, String]]) {

  def foreach[U](f: Map[String, String] => U): Unit = {
    lines.foreach(f)
  }

  def map[U](f: Map[String, String] => U): Seq[U] = {
    lines.map(f)
  }

  def flatMap[U](f: Map[String, String] => Iterable[U]): Seq[U] = {
    lines.flatMap(f)
  }

  def apply(idx: Int): Map[String, String] = {
    lines(idx)
  }

  def getSentence(idx: Int, name: String = "tokens"): Seq[String] = {
    return apply(idx)(name).tail.init.split(", ").map(it => it.tail.init).toSeq
  }

  def filter(lambda: (Map[String, String]) => Boolean): PandasLikeDataset = {
    PandasLikeDataset(lines.filter(lambda))
  }

  def get(idx: Int, field: String): String = {
    apply(idx)(field)
  }

  /**
   * Get the sentence, but replace the highlighted part with its corresponding type
   */
  def getSentenceWithTypes(idx: Int): Seq[String] = {
    getSentenceWithTypesAndSpec(idx)._1
  }

  /**
    * Get the data for this line
    * The sentence is returned with types, the spec is returned with types
    *
    * @param idx: the line
    * @return a tuple consisting of tokens
    */
  def getSentenceWithTypesAndSpec(idx: Int): (Seq[String], Spec) = {
    val line = apply(idx)
    val sentence = line("tokens").tail.init.split(", ").map(it => it.tail.init).toSeq

    val reversed = line("reversed").toInt
    val startFirstLabel  = if (reversed == 0) "subj_start" else "obj_start"
    val endFirstLabel    = if (reversed == 0) "subj_end"   else "obj_end"
    val startSecondLabel = if (reversed == 0) "obj_start"  else "subj_start"
    val endSecondLabel   = if (reversed == 0) "obj_end"    else "subj_end"
    val firstLabelName   = if (reversed == 0) "subj_type"  else "obj_type"
    val secondLabelName  = if (reversed == 0) "obj_type"   else "subj_type"

    val firstPart: Seq[String] = sentence.take(line(startFirstLabel).toInt)
    val between:   Seq[String] = sentence.slice(line(endFirstLabel).toInt+1, line(startSecondLabel).toInt)
    val lastPart:  Seq[String] = sentence.drop(line(endSecondLabel).toInt + 1)
    val text = firstPart ++ Seq(line(firstLabelName)) ++ between ++ Seq(line(secondLabelName)) ++ lastPart

    val spec = Spec("text", idx, firstPart.length, (firstPart.length + between.length + 1) + 1)

    return (text, spec)
  }


  def getData(): (Seq[Seq[String]], Seq[Spec]) = {
    val resultedSentences = mutable.ListBuffer.empty[Seq[String]]
    val resultedSpecs     = mutable.ListBuffer.empty[Spec]
    for (idx <- 0 until length()) {
      val (text, spec) = getSentenceWithTypesAndSpec(idx)
      resultedSentences.append(text)
      resultedSpecs.append(spec)
    }
    (resultedSentences.toSeq, resultedSpecs.toSeq)
  }

  def length(): Int = lines.length

}
object PandasLikeDataset {
  def apply(path: String, header: Boolean = true, index: Boolean = true, sep: String = "\t"): PandasLikeDataset = {
    val (lines, headerNames) = using(Source.fromFile(path)) { it => 
      val allLines: IndexedSeq[String] = it.getLines().toIndexedSeq
      val headerNames: IndexedSeq[String] = if (header) {
        val split = allLines.head.split(sep)
        if (index) {
          split.tail
        } else {
          split
        }
      } else {
        val genericNames = (0 until allLines.head.split(sep).length).map { it => f"col_$it" }
        if (index) {
          genericNames.tail
        } else {
          genericNames
        }
      }
      (allLines.tail.map { it => if(index) it.split(sep).tail else it.split(sep) }, headerNames)
    }
    return PandasLikeDataset(lines.map(it => headerNames.zip(it).toMap))
  }

  def fromLines(allLines: IndexedSeq[String], header: Boolean = true, index: Boolean = true, sep: String = "\t"): PandasLikeDataset = {
    val headerNames: IndexedSeq[String] = if (header) {
      val split = allLines.head.split(sep)
      if (index) {
        split.tail
      } else {
        split
      }
    } else {
      val genericNames = (0 until allLines.head.split(sep).length).map { it => f"col_$it" }
      if (index) {
        genericNames.tail
      } else {
        genericNames
      }
    }
    val lines = allLines.tail.map { it => if(index) it.split(sep).tail else it.split(sep) }

    return PandasLikeDataset(lines.map(it => headerNames.zip(it).toMap))

  }

}

/**
  * A holder for the clusters data as an underlying PandasLikeDataset
  * Looks like:
  *   \tsubj_start\tsubj_end\tsubj_type\tobj_start\tobj_end obj_type\thighlighted\ttokens\treversed\trelation\thighlighted_string
  * On each subsequent line, the first value is the index
  * 
  * @param pld
  */
case class ClusterDataset(pld: PandasLikeDataset) {
  /**
   * Get the sentence, but replace the highlighted part with its corresponding type
   */
  def getSentenceWithTypes(idx: Int): Seq[String] = {
    getSentenceWithTypesAndSpecWithTypes(idx)._1
  }
  
  /**
    * The sentence, say: "She is an American actress and singer" with "is an American" as the part of interest becomes:
    * PERSON is an American TITLE and singer and the part of interest is "PERSON is an American TITLE"
    *
    * @param idx
    * @return
    */
  def getSentenceWithTypesAndSpecWithTypes(idx: Int): (Seq[String], Spec) = {
    val line = apply(idx)
    val sentence = line("tokens").tail.init.split(", ").map(it => it.tail.init).toSeq // Drop '[' and ']'. Split by ', ' to recover the original sentence

    val reversed = line("reversed").toInt
    val startFirstLabel  = if (reversed == 0) "subj_start" else "obj_start"
    val endFirstLabel    = if (reversed == 0) "subj_end"   else "obj_end"
    val startSecondLabel = if (reversed == 0) "obj_start"  else "subj_start"
    val endSecondLabel   = if (reversed == 0) "obj_end"    else "subj_end"
    val firstLabelName   = if (reversed == 0) "subj_type"  else "obj_type"
    val secondLabelName  = if (reversed == 0) "obj_type"   else "subj_type"

    val firstPart: Seq[String] = sentence.take(line(startFirstLabel).toInt)
    val between:   Seq[String] = sentence.slice(line(endFirstLabel).toInt+1, line(startSecondLabel).toInt)
    val lastPart:  Seq[String] = sentence.drop(line(endSecondLabel).toInt + 1)
    val text = firstPart ++ Seq(line(firstLabelName)) ++ between ++ Seq(line(secondLabelName)) ++ lastPart

    val spec = Spec("text", idx, firstPart.length, (firstPart.length + between.length + 1) + 1)

    return (text, spec)
  }

  def getSentence(idx: Int): Seq[String] = {
    Seq.empty
  }

  def getSentenceAndSpec(idx: Int): (Seq[String], Spec) = {
    val line = apply(idx)
    val sentence = line("tokens").tail.init.split(", ").map(it => it.tail.init).toSeq // Drop '[' and ']'. Split by ', ' to recover the original sentence
    val reversed = line("reversed").toInt
    val startFirstLabel  = if (reversed == 0) "subj_start" else "obj_start"
    val endSecondLabel   = if (reversed == 0) "obj_end"    else "subj_end"

    
    val spec = Spec("text", idx, line(startFirstLabel).toInt, line(endSecondLabel).toInt + 1)

    return (sentence, spec)
    
  }

    /**
      * The sentence, say: "She is an American actress and singer" with "is an American" as the part of interest becomes:
      * PERSON is an American TITLE and singer and the part of interest remains "is an American"
      *
      * @param idx
      * @return
      */
    def getSentenceWithTypesAndSpecWithoutTypes(idx: Int): (Seq[String], Spec) = {
    val line = apply(idx)
    val sentence = line("tokens").tail.init.split(", ").map(it => it.tail.init).toSeq // Drop '[' and ']'. Split by ', ' to recover the original sentence

    val reversed = line("reversed").toInt
    val startFirstLabel  = if (reversed == 0) "subj_start" else "obj_start"
    val endFirstLabel    = if (reversed == 0) "subj_end"   else "obj_end"
    val startSecondLabel = if (reversed == 0) "obj_start"  else "subj_start"
    val endSecondLabel   = if (reversed == 0) "obj_end"    else "subj_end"
    val firstLabelName   = if (reversed == 0) "subj_type"  else "obj_type"
    val secondLabelName  = if (reversed == 0) "obj_type"   else "subj_type"

    val firstPart: Seq[String] = sentence.take(line(startFirstLabel).toInt)
    val between:   Seq[String] = sentence.slice(line(endFirstLabel).toInt+1, line(startSecondLabel).toInt)
    val lastPart:  Seq[String] = sentence.drop(line(endSecondLabel).toInt + 1)
    val text = firstPart ++ Seq(line(firstLabelName)) ++ between ++ Seq(line(secondLabelName)) ++ lastPart

    val spec = Spec("text", idx, firstPart.length + 1, (firstPart.length + between.length + 1))

    return (text, spec)
  }



  def getData(): (Seq[Seq[String]], Seq[Spec]) = {
    val resultedSentences = mutable.ListBuffer.empty[Seq[String]]
    val resultedSpecs     = mutable.ListBuffer.empty[Spec]
    for (idx <- 0 until length()) {
      val (text, spec) = getSentenceWithTypesAndSpecWithoutTypes(idx)
      resultedSentences.append(text)
      resultedSpecs.append(spec)
    }
    
    (resultedSentences.toSeq, resultedSpecs.toSeq)
  }

  def apply(idx: Int) = pld(idx)

  def length(): Int = pld.length()

  def getFirstObjects: Seq[String] = {
    getDirectionality match {
      case SubjObjDirection => pld.lines.map(_("subj_type")).distinct
      case ObjSubjDirection => pld.lines.map(_("obj_type")).distinct
    }
  }

  def getSecondObjects: Seq[String] = {
    getDirectionality match {
      case SubjObjDirection => pld.lines.map(_("obj_type")).distinct
      case ObjSubjDirection => pld.lines.map(_("subj_type")).distinct
    }
  }

  def getDirectionality: PatternDirection = {
    assert(pld.lines.map(_("reversed").toInt).toSet.size == 1)
    val reversed = pld.lines.map(_("reversed").toInt).head
    PatternDirection.fromIntValue(reversed)
  }

  def getRelation: String = {
    assert(pld.lines.map(_("relation")).toSet.size == 1)
    val relation = pld.lines.map(_("relation")).head
    relation
  }

}
object ClusterDataset {
  def apply(path: String): ClusterDataset = ClusterDataset(PandasLikeDataset(path))
}