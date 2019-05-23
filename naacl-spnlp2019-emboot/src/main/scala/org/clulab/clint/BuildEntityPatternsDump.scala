package org.clulab.clint

import java.io._
import scala.io.Source
import org.clulab.odin.Mention
import org.clulab.processors.Document
import com.typesafe.scalalogging.LazyLogging
import com.typesafe.config.ConfigFactory
import ai.lum.common.ConfigUtils._
import ai.lum.common.StringUtils._
import ai.lum.common.Serializer

object BuildEntityPatternsDump extends App with LazyLogging {

  val config = ConfigFactory.load()
  val docsDir:File = config[File]("clint.docs-dir")
  val indexDir = config[File]("clint.index-dir")
  val maxTokens = config[Int]("clint.max-tokens")
  val candidateFinderName = config[String]("clint.candFinderToUse")

  logger.info("Loading lexicons")
  val wordLexemeToIndex = LexiconBuilder.loadLexemeToIndex(new File(indexDir, "word.lexicon"))
  val wordIndexToLexeme = LexiconBuilder.loadIndexToLexeme(new File(indexDir, "word.lexicon")) //Note: To avoid double loading
  val entityLexicon = LexiconBuilder.loadLexicon(new File(indexDir, "entity.lexicon"))

  // Output File
  val entityPatterns = new BufferedWriter(new FileWriter(new File(indexDir, "entityPatterns.dump")))

  // Emboot Output files
  val trainingDataEmbootFile = new BufferedWriter(new FileWriter(new File(indexDir, "training_data_with_labels_emboot.txt")))
  val entityLabelsEmbootFile = new BufferedWriter(new FileWriter(new File(indexDir, "entity_labels_emboot.txt")))
  val patternLabelsEmbootFile = new BufferedWriter(new FileWriter(new File(indexDir, "pattern_labels_emboot.txt")))
  val entityLabelsCountEmbootFile = new BufferedWriter(new FileWriter(new File(indexDir, "entity_label_counts_emboot.txt")))


  val finder = candidateFinderName match {
    case "ontonotes" => new OracleCandidateFinderOntonotes
    case "conll" => new OracleCandidateFinder
    case "scienceie" => new OracleCandidateFinderScienceIE
  }

  val entityLabels = for (f <- docsDir.listFiles() if f.getName().endsWith(".ser")) yield {

    logger.info(s"Loading ${f.getName()}")
    val doc = Serializer.deserialize[Document](f)

    logger.info("Searching for entity candidates")
    val mentionCandidates = finder.findCandidates(doc).sorted

    logger.info("Getting entity patterns")
    for (c <- mentionCandidates) {
      val patternMentionsForCand = for (p <- patternsAroundMention(c, maxTokens, wordLexemeToIndex, entityLexicon)) yield {
        entityPatterns.write(p.withEntityIds + "\n")
        p.patternString(wordIndexToLexeme)
      }
      if(patternMentionsForCand.size > 0) { // Write only those mentions to the emboot training file which have a non-zero size context
        trainingDataEmbootFile.write(s"${c.label}\t${c.text}\t${patternMentionsForCand.mkString("\t")}\n")
        entityLabelsEmbootFile.write(s"${c.text}\t${c.label}\n")
        for (pat <- patternMentionsForCand) {
          patternLabelsEmbootFile.write(s"${pat}\t${c.label}\n")
        }
      }
    }

    mentionCandidates.map{c =>
      (c.text,c.label)
    }.toArray

  }

  val entityLabelsCounts = entityLabels.flatMap(i => i).map(j => (j,1)).groupBy(_._1).map(k => (k._1, k._2.length)).toArray
  for (((entity,label),count) <- entityLabelsCounts) {
    entityLabelsCountEmbootFile.write(s"${count}\t${entity}\t${label}\n")
  }

  entityPatterns.close()
  trainingDataEmbootFile.close()
  entityLabelsEmbootFile.close()
  patternLabelsEmbootFile.close()
  entityLabelsCountEmbootFile.close()

  def auxVerbs = Set("be", "is", "are", "was", "were", "have", "has", "had", "do", "did", "does")
  def isListItem(s: String) = """\d+\.?""".r.findFirstIn(s).isDefined

  def patternsAroundMention(
      m: Mention,
      maxTokens: Int, // max number of tokens to one side
      wordLexicon: LexemeToIndex,
      entityLexicon: Lexicon
  ): Seq[Pattern] = {
    val sentence = m.sentenceObj
    val entity = mkEntity(m.text)
    val leftWords = for {
      n <- 1 to maxTokens if m.start >= n
      start = m.start - n
      end = m.start
      tags = sentence.tags.get.slice(start, end) // NOTE: Commented for ScienceIE Dataset
      // should contain a verb or noun
      if tags.exists(tag => tag.startsWith("N") || tag.startsWith("V")) // NOTE: Commented for ScienceIE Dataset
      // don't start with a conjunction
      if tags.head != "CC" // NOTE: Commented for ScienceIE Dataset
      words = sentence.words.slice(start, end)
      // don't start with punctuation
      if !words.head.isPunctuation
      // filter bad patterns
      if !(words.size == 1 && (auxVerbs.contains(words.head) || isListItem(words.head)))
    } yield words.map(mkWord)
    val rightWords = for {
      n <- 1 to maxTokens if m.end + n <= sentence.size
      start = m.end
      end = start + n
      tags = sentence.tags.get.slice(start, end) // NOTE: Commented for ScienceIE Dataset
      // should contain a verb or noun
      if tags.exists(tag => tag.startsWith("N") || tag.startsWith("V")) // NOTE: Commented for ScienceIE Dataset
      words = sentence.words.slice(start, end)
      // don't end with punctuation
      if !words.last.isPunctuation
      // filter bad patterns
      if !(words.size == 1 && (auxVerbs.contains(words.head) || isListItem(words.head)))
    } yield words.map(mkWord)
    val leftPatterns = leftWords.map(ws => new Pattern(ws :+ entity))
    val rightPatterns = rightWords.map(ws => new Pattern(entity +: ws))
    val surroundPatterns = for {
      left <- leftWords
      right <- rightWords
    } yield new Pattern(left ++ Seq(entity) ++ right)
    leftPatterns ++ rightPatterns ++ surroundPatterns
  }

  def mkEntity(s: String): Entity = new Entity(entityLexicon(s))
  def mkWord(s: String): Word = new Word(wordLexemeToIndex(s))

  // TODO filter by these patterns instead?
  //   /^VB/+ (/^JJ/|IN|DT)* (?=@Entity)
  //   /^(NN|JJ)/+ (/^JJ/|IN|DT)* (?=@Entity)
  //   (?<=@Entity) /^VB/+ ([chunk='B-NP'][chunk='I-NP']*)?
  //   (?<=@Entity) /^VB/+ IN

}
