package edu.arizona.sista.extraction

import java.io.{PrintWriter, _}
import java.util.Date

import edu.arizona.sista.odin.{ExtractorEngine, Mention}
import edu.arizona.sista.processors.DocumentSerializer
import edu.arizona.sista.processors.fastnlp.FastNLPProcessor
import extractionUtils._
import preprocessing.agiga.ProcessAgiga


/**
  * Created by bsharp on 3/8/16.
  */
object ExtractFromWiki extends App {

  // Extracts the wiki file index (e.g. 34693 from the example on next line), returns as Int
  // /data/nlp/corpora/AriResources/wikipedia/simple_wiki_annotated/simplewiki_34693.ser
  // fixme - this is a hack!
  def extractWikiIndex(fn:String):Int = {
    val split = fn.split("(\\.|_)")
    split(split.length - 2).toInt
  }

  def sortWikiFiles(fns:Seq[String]):Seq[String] = {
    val withIndex = fns.map(fn => (extractWikiIndex(fn), fn))
    // Sort by the index
    val sorted = withIndex.sortBy(_._1)
    // return the sorted files
    sorted.unzip._2
  }

  // Used for time stamping
  val pwTime = new PrintWriter("/home/bsharp/wikiCausalTimer_mar17.txt")
  val start = new Date()
  pwTime.println("START TIME STAMP: " + start)
  pwTime.flush()

  lazy val proc = new FastNLPProcessor(withDiscourse = false)

  // Initialize the Odin Extractor
  //val source = io.Source.fromURL(getClass.getResource("/grammars/causal/rules.yml"))
//  val source = io.Source.fromURL(getClass.getResource("/grammars/causal/precisionRules.yml"))
  val source = io.Source.fromURL(getClass.getResource("/grammars/causal/CMBasedRules.yml"))
  val rules = source.mkString
  println(rules)
  source.close()
  val actions = new WorldTreeActions
  val extractor = ExtractorEngine(rules, actions)

  // SIMPLE WIKI
  val fileOut = "/home/bsharp/causal/tempOut/causalOut_CM_simpleWikiBig_mar19b_"
  val dir = "/data/nlp/corpora/AriResources/wikipedia/simple_wiki_annotated"
  val filenames = ProcessAgiga.findFiles(dir, "ser").map(_.getAbsolutePath)
  val sorted = sortWikiFiles(filenames)
  val view = "lemmasWithTags"

  // Load the wiki files and process
  var groupCounter:Int = 0
  for (fGroup <- sorted.indices.grouped(5000)) {
    if (extractWikiIndex(sorted(fGroup.head)) < 150000 || extractWikiIndex(sorted(fGroup.head)) > 150500) {

      // Initialize the output files
      val pwGroupInfo = new PrintWriter(fileOut + "_" + groupCounter + ".groupInfo")
      val pwDetail = new PrintWriter(fileOut + "_" + groupCounter + ".detail")
      val pwArgs= new PrintWriter(fileOut + "_" + groupCounter + ".args")
      val pwSents = new PrintWriter(fileOut + "_" + groupCounter + "_all.sents")
      val pwMentions = new PrintWriter(fileOut + "_" + groupCounter + "_all.mentions")

      for (fIdx <- fGroup) {
        val fn = sorted(fIdx)
        // Store the wikipedia filename in the group log
        pwGroupInfo.println(fn)

        if (!fn.endsWith("simplewiki_34693.ser")) {

          println("Loading the serialized doc from " + fn)
          val ds = new DocumentSerializer
          val reader = new BufferedReader(new FileReader(fn))
          val doc = ds.load(reader)
          reader.close()
          println("Loaded the serialized doc from " + fn)
          println ("Processing Document " + fIdx + ", which has " + doc.sentences.length + " sentences.")
          println ("TIME STAMP: " + new Date())
          pwTime.println ("TIME STAMP: " + new Date())
          pwTime.flush()

          // Print the sentences to a file
          doc.sentences.foreach(s => pwSents.println(sentenceToString(s)))

          // Run odin and extract patterns
          val mentionsRaw = extractor.extractFrom(doc).sortBy(m => (m.sentence, m.getClass.getSimpleName))
          //++ extractor.extractFrom(testDoc).sortBy(m => (m.sentence, m.getClass.getSimpleName))
          // Display the found mentions
          val mentions = collapseMentions(mentionsRaw)
          //mentions.foreach(displayMention(_))
          mentions.foreach(m => pwMentions.println(mentionToString(m)))

          // Group the mentions that are in the same sentence
          val sorted: Seq[Mention] = mentions.sortBy(_.sentence)

          // Iterate through the Causal event mentions and print the arguments
          // TODO: do this much better!
          for (e <- sorted) {
            if (e.matches("Causal")) {
              pwDetail.println(mentionToString(e))
              val (causes, effects, examples) = causalArgumentsToTuple(e, filterContent = true, collapseNE = true, view)
              pwArgs.println (causes.mkString(",") + "\t-->\t" + effects.mkString(","))
              // Flush the PrintWriters
              pwDetail.flush()
              pwArgs.flush()
            }
          }
        }


      }

      // Housekeeping
      pwGroupInfo.close()
      pwDetail.close()
      pwArgs.close()
      pwSents.close()
      pwMentions.close()

    }


    groupCounter += 1
  }



  // Used for time stamping
  val end = new Date()
  pwTime.println("END TIME STAMP: " + end)
  pwTime.close()


}
