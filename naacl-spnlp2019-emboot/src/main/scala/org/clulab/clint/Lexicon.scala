package org.clulab.clint

import java.io._

import com.typesafe.config.ConfigFactory
import ai.lum.common.ConfigUtils._

import scala.io.Source
import scala.collection.mutable.HashMap

class LexiconBuilder extends Serializable {

  private var lexicon = new HashMap[Int, String]
  private var inverseLexicon = new HashMap[String, Int]
  private var counts = new HashMap[Int, Int]

  def add(s: String): Int = {
    if (inverseLexicon contains s) {
      val i = inverseLexicon(s)
      counts.put(i, counts.get(i).get+1)
      i
    } else {
      val i = lexicon.size
      lexicon +=  (i -> s)
      inverseLexicon += (s -> i)
      counts += (i -> 1)
      i
    }
  }

  def apply(i: Int): String = lexicon(i)

  def apply(s: String): Int = inverseLexicon(s)

  def get(i: Int): Option[String] = lexicon.lift(i)

  def get(s: String): Option[Int] = inverseLexicon.get(s)

  def contains(s: String): Boolean = inverseLexicon.contains(s)

  def exists(i: Int): Boolean = i < lexicon.size

  def size: Int = lexicon.size

  def totalCount: Int = counts.values.toArray.sum

  def toLexicon: Lexicon = new Lexicon(lexicon.toMap, inverseLexicon.toMap)

  def toIndexToLexeme: IndexToLexeme = new IndexToLexeme(lexicon.toMap)

  def toLexemeToIndex: LexemeToIndex = new LexemeToIndex(inverseLexicon.toMap)

  def saveTo(filename: String): Unit = saveTo(new File(filename))

  def saveTo(file: File): Unit = {
    val writer = new BufferedWriter(new FileWriter(file))
    writer.write(s"${lexicon.size}\n")
    for ((i,str) <- lexicon.toArray.sortBy(_._1)) { //NOTE: not necessary; Just to compare the lexicon to the previous version
      writer.write(s"$i\t${str}\n")
    }
    writer.close()
  }

  def writeCounts(file: File): Unit = {
    val writer = new BufferedWriter(new FileWriter(file))
    for ((i,cnt) <- counts.toArray.sortBy(_._1)) { //NOTE: not necessary; Just to compare the lexicon to the previous version
      writer.write(s"$i\t$cnt\n")
    }
    writer.close()
  }

  def writeCountsEmboot(file: File, isPatternLexicon: Boolean): Unit = {
    println(s"Writing the lexicon file for EmBoot :  ${file.toString}")
    val writer = new BufferedWriter(new FileWriter(file))

    writer.write("</s>\t0\n") // Write the dummy symbol <eos> character

    val lexiconStringCounts = (if (isPatternLexicon) {
      val config = ConfigFactory.load()
      val indexDir = config[File]("clint.index-dir")
      val wordLexicon = LexiconBuilder.loadLexicon(new File(indexDir, "word.lexicon"))

      // Convert the pattern from wordIds to a readable string using the word.lexicon
      lexicon.map { pat =>
        val patId = pat._1
        val patString = pat._2
        val patReadableString = patString.split(" ").map {
          case "@" => "@ENTITY"
          case n => wordLexicon(n.toInt)
        }.mkString(" ")
        (patId, patReadableString)
      }
    } else {
      lexicon
    })
    .map { lex => // lexicon entry (id -> string) where string is aprropriately replaced by readable string for pattern lexicon
      val lexId = lex._1
      val lexString = lex._2
      val lexCount = counts.get(lexId).get
      lexString -> lexCount
    }
    .toArray // convert to Array
    .sortBy(- _._2) // sort in decreasing order of counts

    // write the counts to the file
    for ((lexString,count) <- lexiconStringCounts){
      writer.write(lexString + "\t" + count + "\n")
    }

    writer.close()
  }

}

object LexiconBuilder {

  def loadCounts(file: File): Map[Int, Int] = {
    println(s"Reading count file : ${file.getName}")
    Source.fromFile(file).getLines().toArray.map { t =>
      val fields = t.split("\t")
      fields(0).toInt -> fields(1).toInt
    }.toMap

  }
  def loadLexicon(filename: String): Lexicon = loadLexicon(new File(filename))

  def loadLexicon(file: File): Lexicon = {
    val source = Source.fromFile(file)
    val lines = source.getLines()
    lines.next().toInt // skip size
    val lexicon = new HashMap[Int, String]
    val inverseLexicon = new HashMap[String, Int]
    for (line <- lines) {
      val (is, s) = try {
      val Array(is, s) = line.split("\t")
      (is, s)
      } catch {
        case e => println(line); throw e
      }
      val i = is.toInt
      lexicon += (i -> s)
      inverseLexicon += (s -> i)
    }
    new Lexicon(lexicon.toMap, inverseLexicon.toMap)
  }

  def loadLexemeToIndex(filename: String): LexemeToIndex = loadLexemeToIndex(new File(filename))

  def loadLexemeToIndex(file: File): LexemeToIndex = {
    val source = Source.fromFile(file)
    val lines = source.getLines()
    lines.next() // skip size
    val inverseLexicon = new HashMap[String, Int]
    for (line <- lines) {
      val Array(is, s) = line.split("\t")
      val i = is.toInt
      inverseLexicon += (s -> i)
    }
    new LexemeToIndex(inverseLexicon.toMap)
  }

  def loadIndexToLexeme(filename: String): IndexToLexeme = loadIndexToLexeme(new File(filename))

  def loadIndexToLexeme(file: File): IndexToLexeme = {
    val source = Source.fromFile(file)
    val lines = source.getLines()
    lines.next().toInt // skip size
    val lexicon = new HashMap[Int, String]
    for (line <- lines) {
      val Array(is, s) = line.trim.split("\t")
      val i = is.toInt
      lexicon += (i -> s)
    }
    new IndexToLexeme(lexicon.toMap)
  }

}

class IndexToLexeme(val lexicon: Map[Int, String]) {

  def apply(i: Int): String = lexicon(i)

  def get(i: Int): Option[String] = lexicon.lift(i)

  def contains(s: String): Boolean = lexicon.values.toArray.contains(s)

  def exists(i: Int): Boolean = i < lexicon.size

  def size: Int = lexicon.size

}

class LexemeToIndex(val inverseLexicon: Map[String, Int]) {

  def apply(s: String): Int = inverseLexicon(s)

  def get(s: String): Option[Int] = inverseLexicon.get(s)

  def contains(s: String): Boolean = inverseLexicon.contains(s)

  def exists(i: Int): Boolean = i < inverseLexicon.size

  def size: Int = inverseLexicon.size

}

class Lexicon(val lexicon: Map[Int, String], val inverseLexicon: Map[String, Int]) {

  def apply(i: Int): String = lexicon(i)

  def apply(s: String): Int = inverseLexicon(s)

  def get(i: Int): Option[String] = lexicon.lift(i)

  def get(s: String): Option[Int] = inverseLexicon.get(s)

  def contains(s: String): Boolean = inverseLexicon.contains(s)

  def exists(i: Int): Boolean = i < lexicon.size

  def size: Int = lexicon.size

}
