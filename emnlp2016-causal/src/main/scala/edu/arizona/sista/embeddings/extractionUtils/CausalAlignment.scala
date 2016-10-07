package extractionUtils

import java.io.PrintWriter
import preprocessing.CreateGoldbergInput
import preprocessing.agiga.ProcessAgiga
import scala.collection.mutable.ArrayBuffer

/**
  * Created by bsharp on 4/4/16.
  */
object CausalAlignment {

  val stopWords = CreateGoldbergInput.stopWords

  def loadData(dir:String, fileExtension: String, lenThreshold:Int): (Seq[String], Seq[String]) = {
    // Initialize the output sequences
    val pairs = new ArrayBuffer[(String, String)]

    // Get the files
    val files = ProcessAgiga.findFiles(dir, fileExtension).map(f => f.getAbsolutePath)

    // Parse each file
    for {
      f <- files
      filePairs = parseFile(f, lenThreshold)
    } pairs.appendAll(filePairs)

    // Split the loaded pairs into parallel sequences for src and dst
    val (src, dst) = pairs.unzip

    (src, dst)
  }

  // Get all the (src, dst) pairs in the file
  def parseFile(fn:String, lenTreshold:Int): Seq[(String, String)] = {
    val out = new ArrayBuffer[(String, String)]
    // Read file
    val source = scala.io.Source.fromFile(fn)
    val lines = source.getLines()
    for {
      l <- lines
      pairs = parseLine(l, lenTreshold)
    } out.appendAll(pairs)

    source.close()
    out
  }

  // Parses a line from the text file into the goldberg (word,context) pair format
  // Also updates the word counter (wc) and the context counter (cc)
  def parseLine(line:String, lenThreshold:Int): Seq[(String, String)] = {
    // TODO: handle negation! (possibly by affixing neg to each word)
    val out = new ArrayBuffer[(String, String)]

    // Split apart the causes and effects
    val args = line.split("\t-->\t")
    // If the line is broken, skip
    if (args.length != 2) {
      println ("Error, invalid line: " + line)
    }
    // Otherwise, handle
    else {
      // FIXME: Handle multi-word causes
      // --> right now, this is a FLAT representation! :(
      // Make an array of all the cause words
      val causesRaw = args(0).split(",").map(_.trim)
      val causes = filterArguments(causesRaw, lenThreshold)
      // Make an array of all the cause words
      val effectsRaw = args(1).split(",").map(_.trim)
      val effects = filterArguments(effectsRaw, lenThreshold)

      if (!causes.isEmpty && !effects.isEmpty) {
        for {
          c <- causes
          e <- effects
        } out.append((c, e))
      }
    }

    // Return!
    out.toSeq
  }

  def filterArguments(args:Array[String], lengthThreshold:Int):Array[String] = {
    val out = new ArrayBuffer[String]
    for (arg <- args){
      // Tokenize on whitespace
      val raw = arg.split(" ")
      // Remove words which are stop words or not a valid POS tag
      val filtered = raw.filter(w => CreateGoldbergInput.filterContent(w))
      // Remove the POS tag from the remaining words
      // TODO: should I do this?
      val words = filtered.map(s => CreateGoldbergInput.splitTag(s)._1)

      if (!words.isEmpty) {
        out.append(words.mkString(" "))
      }
    }
    out.toArray
  }


  // Write the text to a file, one element per line
  def saveTxt(txt:Seq[String], fn:String): Unit = {
    val pw = new PrintWriter(fn)
    for (line <- txt) {
      pw.println(line)
      pw.flush()
    }
    pw.close()
  }



}
