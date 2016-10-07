package edu.arizona.sista.qa.translation

import java.io.PrintWriter

import edu.arizona.sista.utils.StringUtils

import scala.collection.mutable.ArrayBuffer

/**
  * Created by bsharp on 4/4/16.
  */
object CausalAlignment {

  val stopWords = Array("be", "become", "do", "get", "have", "lot", "object", "other", "person", "say", "something",
    "take", "thing", "will", "factor", "company")


  // Returns two parallel sequences, one with the SOURCE phrases, one with the DESTINATION phrases
  def loadData(dir:String, fileExtension: String, lenThreshold:Int): (Seq[String], Seq[String]) = {
    // Initialize the output sequences
    val pairs = new ArrayBuffer[(String, String)]

    // Get the files
    val files = FreeTextAlignmentUtils.findFiles(dir, fileExtension).map(f => f.getAbsolutePath)

    // Parse each file
    for {
      f <- files
      filePairs = parseFile(f, lenThreshold)
    } pairs.appendAll(filePairs)

    // Split the loaded pairs into parallel sequences for src and dst
    val (src, dst) = pairs.unzip

    println (s"Finished loading CausalAlignment data from $dir (from files with ext $fileExtension).")
    println (s"\t${src.length} pairs loaded after filtering with a lenThreshold of $lenThreshold.")
    (src, dst)
  }

  // Get all the (src, dst) pairs in the file
  def parseFile(fn:String, lenTreshold:Int): Seq[(String, String)] = {
    val out = new ArrayBuffer[(String, String)]
    // Read file
    val lines = scala.io.Source.fromFile(fn).getLines()
    for {
      l <- lines
      pairs = parseLine(l, lenTreshold)
    } out.appendAll(pairs)

    out
  }

  // Parses a line from the text file into the goldberg (word,context) pair format
  // Also updates the word counter (wc) and the context counter (cc)
  def parseLine(line:String, lenThreshold:Int): Seq[(String, String)] = {
    // TODO: handle negation! (possibly by affixing neg to each word)
    val out = new ArrayBuffer[(String, String)]

    // Split apart the causes and effects
    val args = line.split("\t-->\t")
    assert (args.length == 2)
    // FIXME: Handle multi-word causes
    // --> right now, this is a FLAT representation! :(
    // Make an array of all the cause phrases
    val causesRaw = args(0).split(",").map(_.trim)
    // here, causes is still an array of filtered phrases
    val causes = filterArguments(causesRaw)
    // Make an array of all the cause words
    val effectsRaw = args(1).split(",").map(_.trim)
    val effects = filterArguments(effectsRaw)


    if (!causes.isEmpty && !effects.isEmpty) {
      for {
        c <- causes
        if c.split(" ").length <= lenThreshold || lenThreshold == 0
        e <- effects
        if e.split(" ").length <= lenThreshold || lenThreshold == 0
      } out.append((c, e))
    }

    // Return!
    out.toSeq
  }

  def filterArguments(args:Array[String]):Array[String] = {
    val out = new ArrayBuffer[String]
    for (arg <- args){
      // Tokenize on whitespace
      val raw = arg.split(" ")
      // Remove words which are stop words or not a valid POS tag
      val filtered = raw.filter(w => filterContent(w))
      // Remove the POS tag from the remaining words
      // TODO: should I do this?
      val words = filtered.map(s => splitTag(s)._1)

      if (!words.isEmpty) {
        out.append(words.mkString(" "))
      }
    }
    out.toArray
  }

  def filterContent(s:String):Boolean = {
    val (token, tag) = splitTag(s)

    // If the word is a valid part of speech and it's not a stop word...
    if (isValidTag(tag) && !stopWords.contains(token)) return true
    // otherwise...
    false
  }

  // Take a tagged token and split it into (token, tag)
  // Example:   happy_JJ => (happy, JJ)
  def splitTag(s:String): (String, String) = {
    val split = s.split("_")

    // Handle weird cases where there seems to be no POS tag affixed
    if (split.length == 1) return (s, "UNK")

    // The general case:
    val len = split.length
    val token = split.slice(0, len - 1).mkString("_")
    val tag = split(len - 1)
    (token, tag)
  }

  def isValidTag (tag:String): Boolean = {
    tag.startsWith("NN") || tag.startsWith("VB") //|| tag.startsWith("JJ")
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

  // Makes a GIZA++ matrix from the previously generated Source and Destination Files
  def makeGizaMatrix(gizaPath:String, wdir:String, transPrefix:String, direction:String,
                     causeFilename:String, effectFilename:String, maxFertility:Int = 10) {

    var srcFilename:String = ""
    var dstFilename:String = ""

    if (direction == "c2e") {
      srcFilename = causeFilename
      dstFilename = effectFilename
    } else if (direction == "e2c") {
      srcFilename = effectFilename
      dstFilename = causeFilename
    } else {
      println ("ERROR: Translation direction must be either \"c2e\" or \"e2c\"")
    }

    MakeTranslationMatrix.makeGizaTranslationFile(wdir,
      transPrefix,
      mode = direction,
      buildFilenames = false,
      providedSrcFilename = srcFilename,
      providedDstFilename = dstFilename,
      gizaPath = gizaPath,
      maxFertility = maxFertility
    )
  }

  def main (args:Array[String]): Unit = {
    val props = StringUtils.argsToProperties(args)

    val gizaPath = props.getProperty("gizaPath")
    val lenThreshold = StringUtils.getInt(props, "lenThreshold", 0)
    val maxFertility = StringUtils.getInt(props, "maxFertility", 10)
    val translationDirection = props.get("direction", "c2e")
    val filenamePrefixGiven = props.getProperty("outputFilenamePrefix")
    val filenamePrefix = s"${filenamePrefixGiven}_${translationDirection}_argLenThresh${lenThreshold}_mf$maxFertility"
    val (wdir, transPrefix) = FreeTextAlignmentUtils.extractDir(filenamePrefix)         // Find working directory

    // MAKE a new matrix
    val inputDir = props.getProperty("inputDirectory")
    val ext = props.getProperty("fileExt")
    val (srcTxt, dstText) = loadData(inputDir, ext, lenThreshold)

    // Generate aligned src and dst files
    val srcFile = props.getProperty("srcFile")
    val dstFile = props.getProperty("dstFile")
    saveTxt(srcTxt, wdir + "/" +  srcFile)
    saveTxt(dstText, wdir + "/" + dstFile)

    // Generate the matrix and priors
    println (s"Making causal matrix... $filenamePrefix.matrix")
    makeGizaMatrix(gizaPath, wdir, transPrefix, "ntos", srcFile, dstFile, maxFertility)
    println (s"Making priors... $filenamePrefix.priors")
    FreeTextAlignmentUtils.makePriorsFromStrings(srcTxt ++ dstText, wdir, s"${transPrefix}_${translationDirection}_ml1001")




  }

}
