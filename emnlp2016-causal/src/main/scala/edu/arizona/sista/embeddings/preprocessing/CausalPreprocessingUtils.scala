package preprocessing

import scala.collection.mutable.ArrayBuffer

/**
  * Created by bsharp on 5/13/16.
  */
object CausalPreprocessingUtils {

  val stopWords = Array("be", "become", "do", "get", "have", "lot", "object", "other", "person", "say", "something",
    "take", "thing", "will", "factor", "company")

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

}
