package preprocessing

import edu.arizona.sista.odin.{Mention, ExtractorEngine}
import edu.arizona.sista.processors.fastnlp.FastNLPProcessor
import extractionUtils._


/**
  * Created by bsharp on 3/19/16.
  */
object OdinRulesChecker extends App {
  val proc = new FastNLPProcessor(withDiscourse = false)

  // Load the Odin Rules
  val source = io.Source.fromURL(getClass.getResource("/grammars/causal/CMBasedRules.yml"))
  val rules = source.mkString
  println(rules)
  source.close()
  val actions = new WorldTreeActions
  val extractor = ExtractorEngine(rules, actions)

  var input = ""

  while (input != "EXIT") {
    input = scala.io.StdIn.readLine("Sentence to process (or EXIT): ")

    val doc = proc.annotate(proc.mkDocument(input))
    for (s <- doc.sentences) println(sentenceToString(s))

    // Run odin and extract patterns
    val mentionsRaw = extractor.extractFrom(doc).sortBy(m => (m.sentence, m.getClass.getSimpleName))
    println ("Finished extracting " + mentionsRaw.length + " raw mentions")

    // Filter out mentions which are entirely contained within others from the same sentence
    val mentions = mentionsRaw//collapseMentions(mentionsRaw)
    println ("After collapsing, there are " + mentions.length + " distinct mentions.")

    // Display Mentions
    mentions.foreach(displayMention(_))

    // Group the mentions that are in the same sentence
    val sorted: Seq[Mention] = mentions.sortBy(_.sentence)

    // Iterate through the Causal event mentions and print the arguments
    for (e <- sorted) {
      if (e.matches("Causal")) {
        println ("-------- CAUSAL -------------")
        println(mentionToString(e))
        val (causes, effects, examples) = causalArgumentsToTuple(e, filterContent = true, collapseNE = true, view = "lemmas")
        println(causes.mkString(",") + "\t-->\t" + effects.mkString(","))
        println ("-----------------------------\n")
      }
    }

  }




}
