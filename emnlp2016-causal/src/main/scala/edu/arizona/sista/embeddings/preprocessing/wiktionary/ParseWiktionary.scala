package preprocessing.wiktionary

import java.io.PrintWriter

import edu.arizona.sista.processors.fastnlp.FastNLPProcessor
import edu.arizona.sista.utils.StringUtils

import scala.collection.mutable.ArrayBuffer
import scala.io.BufferedSource
import scala.util.control.Breaks._

/**
 * Convert a dump of the Simple Wiktionary (in their internal format) to a simple structured dictionary format
 * containing the defined word, part of speech, and defining sentence.
 *
 * Created by bsharp on 4/30/15.
 */
class ParseWiktionary {

}


object ParseWiktionary{
  val processor = new FastNLPProcessor()

  def parse(filename:String):(Array[(String, String, String)],Array[(String, String, String)]) = {
    val definitions = new ArrayBuffer[(String, String, String)]
    println (" * Loading file ")
    val source = scala.io.Source.fromFile(filename, "UTF-8")
    //val lines = source.getLines().toArray
    println (" * Parsing pages ")
    //val rawPages = getRawPages(lines)
    val rawPages = getRawPages(source)
    println (" * Filtering raw pages ")
    val (defs, exs) = filterRawpages(rawPages)
    println (" * Complete ")
    (defs, exs)
  }

  def filterRawpages(in:Array[String]):(Array[(String,String, String)],Array[(String,String, String)])  = {
    val definitions = new ArrayBuffer[(String, String, String)]
    val examples = new ArrayBuffer[(String, String, String)]
    val titlePattern = "<title.*?>(.*?)</title>".r
    val textPattern = "(?s)<text.*?>(.*)</text>".r
    for (index <- 0 until in.size) {

      val rawDef = in(index)

      val titleMatches = titlePattern.findAllMatchIn(rawDef).toArray
      if (titleMatches.size > 0) {
        val title = titleMatches(0).group(1)

        if (!title.contains(":")) {
          println(title)
          val textMatches = textPattern.findAllMatchIn(rawDef).toArray
          assert (textMatches.size == 1)
          val text = textMatches(0).group(1)
          val (defsInText, examplesInText) = processDefinitionText(title, text)
          definitions.appendAll(defsInText)
          examples.appendAll(examplesInText)
        }
      }

    }

    (definitions.toArray, examples.toArray)
  }

  def processDefinitionText(word:String, in:String):(Array[(String, String, String)], Array[(String, String, String)]) = {
    val defs = new ArrayBuffer[(String, String, String)]
    val examples = new ArrayBuffer[(String, String, String)]

    val definitionSections = new ArrayBuffer[String]
    val sectionText = new StringBuilder
    var pos:String = ""
    val lines = in.split("\n")

    // Regular expressions for the POS, examples, and the definitions
    //val posPattern = "^==( ?[A-Z][a-z]+( [A-Z][a-z]+)?) ?==".r
    //val posPattern = "^==( ?[A-Z][a-z]+( [A-Z][a-z]+)?) ?==".r
    val examplePattern = "^#:(.*)".r
    val definitionPattern = "^#([^:*].*)".r

    val partsOfSpeech = Array("noun", "verb", "adjective", "adverb")

    // Look through each line and see if there is something we need
    for (line1 <- lines) {

      val line = line1.replaceAll("##", "#")
      val lineLower = line.toLowerCase()

      // Find the pos, example, or definition
      //val posMatches = posPattern.findAllMatchIn(line).toArray
      val exMatches = examplePattern.findAllMatchIn(line).toArray
      val defMatches = definitionPattern.findAllMatchIn(line).toArray

      //println (line)

      for (onePOS <- partsOfSpeech) {
        if (lineLower.contains(onePOS)) {
          if ((lineLower.contains("==" + onePOS + "==")) || (lineLower.contains("== " + onePOS + " =="))) {
            pos = onePOS
            //println ("\t POS MATCH: " + pos)
          }
        }
      }

      if (defMatches.size > 0) {
        val currDef = defMatches(0).group(1).toLowerCase
        val newDef = filterText(currDef).trim()
        if (isValid(pos) && newDef != "") {
          println (word + ": " + pos + ": " + newDef)
          defs.append((word, pos, newDef))
        }
      }
      if (exMatches.size > 0) {
        val currEx = exMatches(0).group(1).toLowerCase
        val newEx = filterText(currEx).trim()
        if (isValid(pos) && newEx != "") {
          println ("EX - " + word + ": " + pos + ": " + newEx)
          examples.append((word, pos, newEx))
        }
      }
    }

    (defs.toArray, examples.toArray)
  }

  def filterText(in:String):String = {
    var out = in
    //println(in)

    // If there is a context, add it to the definition as natural language
    val contextPattern = "\\{\\{context\\|(.*?)\\}\\}".r
    val contextMatches = contextPattern.findAllMatchIn(in).toArray
    if (contextMatches.size > 0) {
      val context = contextMatches(0).group(1)
      out = contextPattern.replaceAllIn(out, " (in " + context + ") ")
    }

    // remove additional marked text (i.e. transitivity, plurality, etc)
    out = out.replaceAll("\\{\\{.*?\\}\\}[;,]?", "")

    // remove some extra characters
    out = out.replaceAll("\\[\\[", "")
    out = out.replaceAll("\\]\\]", "")
    out = out.replaceAll("\\'\\'\\'", "")
    out = out.replaceAll("\\'\\'", "\"")
    out = out.replaceAll("&lt;math&gt;", "")
    out = out.replaceAll("&lt;.*?&gt;", " ")
    out = out.replaceAll("&lt;/math&gt;", "")
    out = out.replaceAll("&quot;", "\"")
    out = out.replaceAll("&amp;", "")

    //&lt;span style=&quot;background-color: coral; width: 80px&quot;&gt;&amp;nbsp;&amp;nbsp;&amp;nbsp;&amp;nbsp;&lt;/span&gt

    out
  }

  // If the POS is one of the desired types
  def isValid(pos:String):Boolean = {
    if (pos.contains("noun") || pos.contains("verb") || pos.contains("adjective")){
      return true
    }
    false
  }

  // Makes an array of Page texts
  //def getRawPages (lines:Array[String]:Array[String] = {
  def getRawPages (source:BufferedSource):Array[String] = {
    val out = new ArrayBuffer[String]
    val currPage = new StringBuilder
    var addLine = false

    //for (lineIndex <- 0 until lines.size) {
    var numLines:Int = 0
    breakable {
      for (lineOrig <- source.getLines()) {
        //val line = lines(lineIndex).trim()
        val line = lineOrig.trim()

        // Detect a new page
        if (line == "<page>") {
          // If the previous page contained anything, add it to the array
          if (!currPage.isEmpty) {
            val pageText = currPage.toString()
            out.append(pageText)
          }
          // Clear out the previous contents
          currPage.clear()
          // Set the boolean to adding mode
          addLine = true
        }
        // If the page is ended, set the boolean to not add
        else if (line == "</page>") {
          addLine = false
        }
        // If it's a regular line of text, add to the string builder if within a page
        else {
          if (line.contains("<title>MediaWiki") || line.contains("<title>Template:")) {
            addLine = false
          }
          if (addLine) {
            currPage.append(line + "\n")
          }
        }

        // Progress bar, for larger tasks
        numLines += 1
        if ((numLines % 10000000) == 0) print(numLines + "  ")
        if ((numLines % 100000000) == 0) println("")

      }
    }


    out.toArray
  }

  def convertPOS(in:String):String = {
    val out = in.toLowerCase
    if (out.contains("adverb")) return "adverb"
    else if (out.contains("verb")) return "verb"
    else if (out.contains("noun")) return "noun"
    else if (out.contains("adjective")) return "adjective"

    else {
      println ("ERROR: unexpected POS type")
      sys.exit(1)
    }

    ""
  }

  // Adds the defined word when it is not included in the definition
  // TODO: refine this?
  def refineDefinition(in:String, word:String):String = {
    val doc = processor.mkDocument(in)
    val firstSent = doc.sentences(0).getSentenceText()
    val inWords = firstSent.split(" ")
    //if (word == "mutation") println ("inWords: [" + inWords.mkString(",") + "]")
    if (inWords.slice(0,5).contains(word)) {
      return firstSent
    }
    else {
      val out = word + " is " + firstSent
      //if(word=="mutation") println("OUT: " + out)
      return out
    }

    ""
  }

  def saveWiktionaryData(filename:String, data:Array[(String, String, String)]) {
    val pw = new PrintWriter(filename)
    for (term <- data) {
      val word = term._1
      val pos = convertPOS(term._2)
      //val text = refineDefinition(term._3, word)
      val text = term._3
      pw.write(word + "\t" + pos + "\t" + text + "\n")
    }
    pw.close()
  }

  def main(args:Array[String]) {
    val props = StringUtils.argsToProperties(args)
    val inputFilename = props.getProperty("input.file")
    val defOutFile = props.getProperty("definition.out.file")
    val exOutFile = props.getProperty("examples.out.file")

    // Retrieve the verbose definitions and examples from the wiktionary text
    val (defs, exs) = parse(inputFilename)

    // output the definitions
    saveWiktionaryData(defOutFile, defs)
    saveWiktionaryData(exOutFile, exs)

    println("Finished...")

  }
}
