package org.clulab.clint

import java.io.{File, FileWriter}

import scala.collection.mutable.{ArrayBuffer, HashMap, StringBuilder}
import org.clulab.odin.Mention
import org.clulab.processors.Document
import com.typesafe.scalalogging.LazyLogging
import com.typesafe.config.ConfigFactory
import ai.lum.common.ConfigUtils._
import ai.lum.common.StringUtils._
import ai.lum.common.FileUtils._
import ai.lum.common.Serializer

import scala.collection.mutable

/***
  * NOTE: DEPRECATED .. see BuildFilesForEmboot.scala ....
  */
object BuildEmbootFile extends App with LazyLogging {

  val config = ConfigFactory.load()
  val docsDir:File = config[File]("clint.docs-dir")
  val maxTokens = config[Int]("clint.max-tokens")
  val candidateFinderName = config[String]("clint.candFinderToUse")

  // Output variable
  val entityFile:File =  config[File]("clint.entityFile") // "ontonotes_entity_vocabulary.txt" --> </s>\t0 +  Sorted(List(<entity>\t<count>))
  val patternFile:File = config[File]("clint.patternFile") //"ontonotes_pattern_vocabulary.txt" --> </s>\t0 + Sorted(List(<pattern>\t<count>))
  val dataFile:File = config[File]("clint.dataFile") //"ontonotes_training_data.txt" --> List(<label>\t<mention>\t<pattern around mention>)
  val labelFile:File = config[File]("clint.labelFile") //"ontonotes_labels.txt" --> List(<mention>\t<label>)
  val patLabelFile:File = config[File]("clint.patLabelFile") //"ontonotes_pattern_labels.txt" --> List(<pattern>\t<label>)
  val entityLabelCountFile:FileWriter = new FileWriter(config[File]("clint.entityLabelCountFile")) //"ontonotes_entity_label_counts.txt" --> List(<count>\t<entity>\t<label>)

  val finder = candidateFinderName match {
    case "ontonotes" => new OracleCandidateFinderOntonotes

    case "conll" => new OracleCandidateFinder

    case "scienceie" => new OracleCandidateFinderScienceIE
  }
  val entityVocabulary = new Vocabulary
  val patternVocabulary = new Vocabulary
  val data = new StringBuilder
  val labels = new StringBuilder
  var patternLabels = new StringBuilder

  var entityLabelCount = new mutable.HashMap[(String, String),Int]()

  for (f <- docsDir.listFilesByWildcard("*.ser")) {
    logger.info(s"loading ${f.getName()}")
    val doc = Serializer.deserialize[Document](f)
//    logger.info(doc.sentences.map{s =>
//      val sent = s.words.mkString(" ")
//      val ne = s.entities.map { x => x.mkString(", ")}.getOrElse("NONE")
//      sent +"\t:\t" + ne
//    }.mkString("\n"))
    logger.info("searching for entity candidates")
    val candidates = finder.findCandidates(doc).sorted
    logger.info("collecting data")
    logger.info(candidates.map { c => c.text+":"+c.label }.mkString("\t"))
    for (c <- candidates) {
      val patterns = patternsAroundMention(c, maxTokens)
      if (patterns.nonEmpty) {
        // only add if we found patterns
        entityVocabulary.add(c.text)
        for (p <- patterns) {
          patternVocabulary.add(p)
          patternLabels.append(p + "\t" + c.label + "\n")
        }
        val row = (c.text +: patterns).mkString("\t")
        data.append(row + "\n")
        labels.append(c.text + "\t" + c.label + "\n")
      }
      val curCount = entityLabelCount.getOrElse((c.text,c.label), 1)
      entityLabelCount.put((c.text,c.label), curCount+1)
    }
  }

  logger.info("sorting vocabularies")
  entityVocabulary.sortInPlace()
  patternVocabulary.sortInPlace()

  logger.info("writing data")
  entityFile.writeString(entityVocabulary.dump)
  patternFile.writeString(patternVocabulary.dump)
  dataFile.writeString(data.toString)
  labelFile.writeString(labels.toString)
  patLabelFile.writeString(patternLabels.toString)

  for (entLabel <- entityLabelCount.keys){
    val cnt = entityLabelCount(entLabel)
    entityLabelCountFile.write(cnt + "\t" + entLabel._1 + "\t" + entLabel._2 + "\n")
  }
  entityLabelCountFile.close
  // this is the end


  def auxVerbs = Set("be", "is", "are", "was", "were", "have", "has", "had", "do", "did", "does")
  def isListItem(s: String) = """\d+\.?""".r.findFirstIn(s).isDefined

  def patternsAroundMention(m: Mention, maxTokens: Int): Seq[String] = {
    val sentence = m.sentenceObj
    val entity = "@ENTITY"
    logger.info(sentence.words.mkString(" "))
    //logger.info(m.start + " " + m.end )
    val leftWords = for {
      n <- 1 to maxTokens if m.start >= n
      start = m.start - n
      end = m.start
      tags = sentence.tags.get.slice(start, end)
      // should contain a verb or noun
      if tags.exists(tag => tag.startsWith("N") || tag.startsWith("V"))
      // don't start with a conjunction
      if tags.head != "CC"
      words = sentence.words.slice(start, end)
      // don't start with punctuation
      if !words.head.isPunctuation
      // filter bad patterns
      if !(words.size == 1 && (auxVerbs.contains(words.head) || isListItem(words.head)))
    } yield words
    val rightWords = for {
      n <- 1 to maxTokens if m.end + n <= sentence.size
      start = m.end
      end = start + n
      tags = sentence.tags.get.slice(start, end)
      // should contain a verb or noun
      if tags.exists(tag => tag.startsWith("N") || tag.startsWith("V"))
      words = sentence.words.slice(start, end)
      // don't end with punctuation
      if !words.last.isPunctuation
      // filter bad patterns
      if !(words.size == 1 && (auxVerbs.contains(words.head) || isListItem(words.head)))
    } yield words
    val leftPatterns = leftWords.map(ws => (ws :+ entity).mkString(" "))
    val rightPatterns = rightWords.map(ws => (entity +: ws).mkString(" "))
    val surroundPatterns = for {
      left <- leftWords
      right <- rightWords
    } yield (left ++ Seq(entity) ++ right).mkString(" ")
    leftPatterns ++ rightPatterns ++ surroundPatterns
  }

}

class Vocabulary {

  val EOS = "</s>"

  private var symbols = new ArrayBuffer[String]
  private var counts = new ArrayBuffer[Int]
  private var symbolToId = new HashMap[String, Int]

  // always add EOS at the beginning
  add(EOS, 0)

  def size: Int = symbols.size

  def add(symbol: String, count: Int = 1): Int = {
    if (!symbolToId.contains(symbol)) {
      symbolToId += (symbol -> symbols.size)
      symbols += symbol
      counts += 0
    }
    val symbolId = symbolToId(symbol)
    counts(symbolId) += count
    symbolId
  }

  def sortedVocabulary: Vocabulary = {
    val vocab = new Vocabulary
    for (i <- symbols.indices.sortBy(counts).reverse) {
      val symbol = symbols(i)
      val count = counts(i)
      vocab.add(symbol, count)
    }
    vocab
  }

  def sortInPlace(): Unit = {
    val sorted = sortedVocabulary
    symbols = sorted.symbols
    counts = sorted.counts
    symbolToId = sorted.symbolToId
  }

  def dump: String = {
    val builder = new StringBuilder
    for ((symbol, count) <- symbols zip counts) {
      builder.append(s"$symbol\t$count\n")
    }
    builder.toString
  }

}