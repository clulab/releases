package edu.arizona.sista.extraction

import edu.arizona.sista.odin.{ExtractorEngine, Mention}
import edu.arizona.sista.processors.Document
import edu.arizona.sista.processors.fastnlp.FastNLPProcessor
import edu.arizona.sista.utils.StringUtils
import extractionUtils.{WorldTreeActions, causalArgumentsToTuple, collapseMentions, mentionToString}
import preprocessing.agiga.ProcessAgiga

import java.io.PrintWriter

/**
  * Created by rebeccas on 2/9/17.
  */
object ExtractFromFreeText {

  val processor = new FastNLPProcessor(withDiscourse = false)

  def mkPartialAnnotation(text: String): Document = {
    val doc = processor.mkDocument(text)
        processor.tagPartsOfSpeech(doc)
        processor.lemmatize(doc)
        processor.parse(doc)
        doc.clear()
        doc
  }

  def main(args: Array[String]): Unit = {
    val props = StringUtils.argsToProperties(args)
    val inputDirectory = props.getProperty("input_dir", "NONE")
    val fileExtension = props.getProperty("input_file_extension", "NONE")
    val outputDirectory = props.getProperty("output_dir", "NONE")
    val view = props.getProperty("view", "lemmasWithTags")

    // Initialize the ODIN extractor
    val rulesSource = io.Source.fromFile(props.getProperty("rules_file", "CMBasedRules.yml"))
    val rules = rulesSource.mkString
    println(rules)
    rulesSource.close()
    val actions = new WorldTreeActions
    val extractor = ExtractorEngine(rules, actions)

    // Get the list of files from the input directory
    val files = ProcessAgiga.findFiles(inputDirectory, fileExtension)

    // Iterate over the files:
    for (file <- files) {
      val fPath = file.getAbsolutePath
      val filename = file.getName

      // Initialize output files for each subset of the documents
      val outputFilePrefix = s"$outputDirectory/$filename.out"
      val pwDetail = new PrintWriter(outputFilePrefix + ".detail")
      val pwArgs = new PrintWriter(outputFilePrefix + ".args")

      // Convert the lines to documents
      val source = scala.io.Source.fromFile(fPath)
      val lines = source.getLines().toArray

      var causalCount = 0
      var lineCounter = 0

      for (line <- lines) {
        if (lineCounter % 1000 == 0) {
          println (s"Processing line $lineCounter of ${lines.length}")
        }
        lineCounter += 1

        val doc = mkPartialAnnotation(line)

        var currCausalCount:Int = 0

        // Run odin and extract patterns
        val mentionsRaw = extractor.extractFrom(doc).sortBy(m => (m.sentence, m.getClass
            .getSimpleName))
//        println ("Finished extracting " + mentionsRaw.length + " raw mentions")

        // Filter out mentions which are entirely contained within others from the same sentence
        val mentions = collapseMentions(mentionsRaw)
//        println ("After collapsing, there are " + mentions.length + " distinct mentions.")

        // Display Mentions
        //mentions.foreach(displayMention(_))
        //mentions.foreach(m => pwMentions.println(mentionToString(m)))

        // Iterate through the Causal event mentions and keep those which are causal
        // Then, group the causal mentions that are in the same sentence
        val causal: Seq[Mention] = mentions.filter(_.matches("Causal")).sortBy(_.sentence)

        // Write the details of the causal events to the files
        if (causal.nonEmpty) {
          for (e <- causal) {
            currCausalCount += 1
            pwDetail.println(mentionToString(e))
            val (causes, effects, examples) = causalArgumentsToTuple(e, filterContent = true, collapseNE = true, view)
            pwArgs.println(causes.mkString(",") + "\t-->\t" + effects.mkString(","))
            // Flush the PrintWriters
            pwDetail.flush()
            pwArgs.flush()
          }
        }

        causalCount += currCausalCount
        if (currCausalCount > 0) println (s"In this line, $currCausalCount causal events found\n")

      }
      println (s"FINAL COUNT: In this file $causalCount causal events found\n")

      // Housekeeping
      source.close()
      pwDetail.close()
      pwArgs.close()
    }

  }

}

