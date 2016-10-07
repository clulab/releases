package preprocessing.agiga

import  java.io.PrintWriter
import java.util.Date

import edu.arizona.sista.odin.{ExtractorEngine, Mention}
import edu.arizona.sista.processors.{Sentence, Document}
import edu.arizona.sista.utils.StringUtils
import extractionUtils._
import scala.collection.parallel.ForkJoinTaskSupport


import scala.collection.mutable.ArrayBuffer

/**
  * Created by bsharp on 3/1/16.
  */
class ExtractFromAgiga {

}

object ExtractFromAgiga {

  def splitAgigaDocIntoSmallerDocuments(doc: Document, n: Int): Array[Document] = {
    val docsOut = new ArrayBuffer[Document]
    val sentences = doc.sentences.toIterator
    var sStack = new ArrayBuffer[Sentence]
    var sentCounter: Int = 0
    while (sentences.hasNext) {
      sStack.append(sentences.next())
      sentCounter += 1

      if (sentCounter == n) {
        docsOut.append(new Document(sStack.toArray))
        // Clear out the stack
        sStack = new ArrayBuffer[Sentence]
        // Reset the counter
        sentCounter = 0
      }
    }

    if (sStack.nonEmpty) docsOut.append(new Document(sStack.toArray))

    docsOut.toArray
  }

  def printSentInfo(doc: Document, n: Int) = {
    val s = doc.sentences(n)
    println("Words: " + s.words.mkString(","))
    println("StartOffsets: " + s.startOffsets.mkString(", "))
    println("EndOffsets: " + s.endOffsets.mkString(", "))
    println("Edges")
  }


  def main(args: Array[String]): Unit = {

    val props = StringUtils.argsToProperties(args)
    val view = props.getProperty("view", "lemmasWithTags")

    // Will keep track of the number of causal mentions in all documents
    var causalCount:Int = 0

    // Initialize the Odin Extractor
    val source = io.Source.fromURL(getClass.getResource(props.getProperty("rules_file", "/grammars/causal/CMBasedRules.yml")))
    val rules = source.mkString
    println(rules)
    source.close()
    val actions = new WorldTreeActions
    val extractor = ExtractorEngine(rules, actions)

    // Load the agiga files
    val agigaDir = props.getProperty("data_dir", "/data/nlp/corpora/agiga/data/xml/")
    val agigaFileList = ProcessAgiga.findFiles(agigaDir, "xml.gz").map(_.getAbsolutePath)

    // Limit the parallelization
    val nthreads = StringUtils.getInt(props, "nthreads", 1)

    for (agigaFile <- agigaFileList) {
      println (s"Loading the agiga file: $agigaDir$agigaFile")
      val doc = ProcessAgiga.agigaDocToDocument(agigaFile)

      println("Agiga file has " + doc.sentences.length + " sentences.")
      val sp = agigaFile.split("/")
      val filename = sp(sp.length - 1)

      // Create the prefix for the output files
      val fileOut = props.getProperty("output_prefix", "causalOut/causalOut_agiga_") + filename

      val docGroups = splitAgigaDocIntoSmallerDocuments(doc, StringUtils.getInt(props, "num_docs_per_output_file", 10000)).zipWithIndex.par
      docGroups.tasksupport = new ForkJoinTaskSupport(new scala.concurrent.forkjoin.ForkJoinPool(nthreads))

      for ((group, groupId) <- docGroups) {

        var currCausalCount:Int = 0

        // Run odin and extract patterns
        println(s"Processing Document Group $groupId, which has " + group.sentences.length + " sentences.")
        val mentionsRaw = extractor.extractFrom(group).sortBy(m => (m.sentence, m.getClass.getSimpleName))
        println ("Finished extracting " + mentionsRaw.length + " raw mentions")

        // Filter out mentions which are entirely contained within others from the same sentence
        val mentions = collapseMentions(mentionsRaw)
        println ("After collapsing, there are " + mentions.length + " distinct mentions.")

        // Display Mentions
        //mentions.foreach(displayMention(_))
        //mentions.foreach(m => pwMentions.println(mentionToString(m)))

        // Group the mentions that are in the same sentence
        val sorted: Seq[Mention] = mentions.sortBy(_.sentence)

        // Iterate through the Causal event mentions and store those which are causal
        val causal = for {
          e <- sorted
          if e.matches("Causal")
        } yield e

        // Write the details of the causal events
        if (causal.length > 0) {
          // Initialize output files for each subset of the documents
          val pwDetail = new PrintWriter(fileOut + "_" + groupId + ".detail")
          val pwArgs = new PrintWriter(fileOut + "_" + groupId + ".args")

          for (e <- causal) {
            currCausalCount += 1
            pwDetail.println(mentionToString(e))
            val (causes, effects, examples) = causalArgumentsToTuple(e, filterContent = true, collapseNE = true, view)
            pwArgs.println(causes.mkString(",") + "\t-->\t" + effects.mkString(","))
            // Flush the PrintWriters
            pwDetail.flush()
            pwArgs.flush()
          }

          // Housekeeping
          pwDetail.close()
          pwArgs.close()
        }

        causalCount += currCausalCount
        println (s"In this group $currCausalCount causal events found\n")
      }
    }


    // Display total number of causal events found
    println("*******")
    println(s"$causalCount Causal Events found!")
  }

}
