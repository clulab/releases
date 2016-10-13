package preprocessing

import java.io.{PrintWriter, FileReader, BufferedReader}

import edu.arizona.sista.embeddings.postprocessing.CalculatePMI
import edu.arizona.sista.struct.Counter
import edu.arizona.sista.utils.StringUtils
import agiga.ProcessAgiga.findFiles
import scala.sys.process._

import scala.collection.mutable.ArrayBuffer

/**
  * Created by bsharp on 3/19/16.
  */
object CreateGoldbergInput {

  val stopWords = Array("be", "become", "do", "get", "have", "lot", "object", "other", "person", "say", "something",
                        "take", "thing", "will", "factor", "company")//, "cause", "know", "people", "go", "anything", "see", "make", "try", "think",
                        //"call", "someone", "nothing", "keep")


  // Parses a line from the text file into the goldberg (word,context) pair format
  // Also updates the word counter (wc) and the context counter (cc)
  def parseLine(line:String,
                wc:Counter[String], cc:Counter[String], rawCounter:Int,
                wTracker:ArrayBuffer[Int], cTracker:ArrayBuffer[Int],
                lenThreshold:Int,
                noiseFiltering:Boolean = false): (Seq[(String, String)], Int) = {

    // TODO: handle negation! (possibly by affixing neg to each word)
    val out = new ArrayBuffer[(String, String)]
    var rawCountOut:Int = rawCounter

    // Split apart the causes and effects
    val args = line.split("\t-->\t")
    // Check for broken input lines, if found, skip
    if (args.length != 2) {
      println ("Error found in line: " + line)
    }
    else {
      var causes = new Array[String](0)
      var effects = new Array[String](0)

      val causesRaw = args(0).split(",").map(_.trim)
      val effectsRaw = args(1).split(",").map(_.trim)

      // FIXME: Handle multi-word causes
      // --> right now, this is a FLAT representation! :(
      // Make an array of all the cause words
      causes = parseArguments(causesRaw, wTracker, lenThreshold, noiseFiltering)
      // Make an array of all the cause words
      effects = parseArguments(effectsRaw, cTracker, lenThreshold, noiseFiltering)

      //      }


      // Raw Counter Update
      rawCountOut += causesRaw.length * effectsRaw.length

      if (!causes.isEmpty && !effects.isEmpty) {
        // Cartesian product...
        for (c <- causes){
          wc.incrementCount(c)
          for (e <- effects) {
            cc.incrementCount(e)
            out.append((c,e))
          }
        }
      }
    }


    // Return!
    (out.toSeq, rawCountOut)
  }

  def parseArguments(args:Array[String], lengthTracker:ArrayBuffer[Int], lengthThreshold:Int, noiseFiltering:Boolean):Array[String] = {
    val out = new ArrayBuffer[String]
    val nerRatio = 0.5
    for (arg <- args){
      // Tokenize on whitespace
      val raw = arg.split(" ")

      // Check what percentage of the argument's tokens are NERs -- relevant for noise filtering
      val nTokens:Double = raw.length
      val tags = raw.map(s => splitTag(s)._2)
      val nNER:Double = tags.count(tag => tag == "NER")

      // Only include the argument if the NER ratio is <= 0.5 OR if we aren't filtering
      if (nNER / nTokens <= nerRatio || !noiseFiltering) {

        // Remove words which are stop words or not a valid POS tag
        assert (!raw.contains("the_DT"))
        val filtered = raw.filter(w => filterContent(w))
        assert (!filtered.contains("the_DT"))

        // Check that there remains at least one noun
        val nTokens1: Double = filtered.length
        val tags1 = filtered.map(s => splitTag(s)._2)
        val nNN = tags1.count(tag => tag.startsWith("NN"))

        if (nNN > 0 || !noiseFiltering) {
          // Remove the POS tag from the remaining words
          val words = filtered.map(s => splitTag(s)._1)
          for {
            w <- words
            if w == "do"
          } {println("eek!")
            sys.exit()}
          // Track the length of the argument
          lengthTracker.append(words.length)
          // Store the words to go out, if there aren't too many
          if (words.length <= lengthThreshold || lengthThreshold == 0) {
            for (w <- words) out.append(w.toLowerCase)
          }
        }
      }

    }
    out.toArray
  }

  def filterContent(s:String):Boolean = {
    val (token, tag) = splitTag(s)

    // If the word is a valid part of speech and it's not a stop word...
    if (isValidTag(tag) && !stopWords.contains(token)) {
//      println ("\t\tRetaining: " + s)
      return true
    }
    // otherwise...
//    println ("Removing: " + s)
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
    tag.startsWith("NN") || tag.startsWith("VB") // || tag.startsWith("JJ")  || tag.startsWith("RB")
  }





  def main(args:Array[String]): Unit = {
    //val props = StringUtils.argsToProperties(args)
    val lenThreshold:Int = 0 // Discard any causes/effects which AFTER filtering are longer than this
    val threshold:Int = 0 // Remove words which appear less than this number of times
    val nBins:Int = 5    // The number of bins for the PMI weighting

    // Get the files which have the causal mentions in them
    //val mentionsDir = "/lhome/bsharp/causal/simpleWiki_mar19b"
    //val mentionsDir = "/lhome/bsharp/causal/agiga+wiki"
    //val mentionsDir = "/lhome/bsharp/causal/causalOut_mar30"  // Used for paper
    //val mentionsDir = "/lhome/bsharp/causal/causalOut_apr1"
    val mentionsDir = "/lhome/bsharp/emnlp2016-replication/causalOut_oct7"
    val mentionsFiles = findFiles(mentionsDir, ".argsC")

    // Specify the output files (which will be the input to word2vecf)
    //val outputPrefix = "/lhome/bsharp/causal/agiga_mar30_nyt_eng_199407_causal"
    //val outputPrefix = s"/lhome/bsharp/causal/goldbergInput/agiga_apr1_AllTemp_causal_threshold$threshold"
    //val outputPrefix = s"/lhome/bsharp/causal/goldbergInput/DEBUG_a+w_mar30_NV_causal_thr${threshold}_pmi${nBins}NF" // last used?
    val outputPrefix = s"/lhome/bsharp/causal/goldbergInput/replicate_oct7_c2e_noPMI"
    //val outputPrefix = "/lhome/bsharp/causal/simpleWiki_mar19b_causal"
    //val outputPrefix = "/lhome/bsharp/causal/agiga_mar19b_causal"
    // a) for the words and context
    val output = new PrintWriter(outputPrefix + ".contexts")
    // b) for the word vocab
    val outputWV = new PrintWriter(outputPrefix + ".wv")
    // c) for the context vocab
    val outputCV = new PrintWriter(outputPrefix + ".cv")

    val wordsCounter = new Counter[String]
    val contextCounter = new Counter[String]
    var argRawCounter:Int = 0
    var argFlatCounter:Int = 0
    val wordsLengthTracker = new ArrayBuffer[Int]
    val contextLengthTracker = new ArrayBuffer[Int]

    // See how many multiples we have...
    val pairCounter = new Counter[(String, String)]
    val causeWordCounter = new Counter[String]
    val effectWordCounter = new Counter[String]

    val pairsFinal = new ArrayBuffer[(String, String)]

    // Iterate through the files and process them
    for (file <- mentionsFiles) {
      // Read the file
      val source = scala.io.Source.fromFile(file)
      val lines = source.getLines().toArray
      source.close()

      for (line <- lines) {
        // Process the lines
        val (pairs, currRawCount) = parseLine(line, wordsCounter, contextCounter, argRawCounter, wordsLengthTracker,
          contextLengthTracker, lenThreshold, noiseFiltering = false)
//        for {
//          p <- pairs
//          if (p._1 == "the") || (p._2 == "the")
//        } sys.exit()
        argRawCounter = currRawCount
        pairsFinal.insertAll(pairsFinal.length, pairs)
        // Print each pair to the output file, space-delineated
        for (pair <- pairs) {
          // TODO -- fix me! make me optional depending on pmi -- or better make both?
//          output.println(s"${pair._1} ${pair._2}")      // WITHOUT PMI
          pairCounter.incrementCount(pair)
          causeWordCounter.incrementCount(pair._1)
          effectWordCounter.incrementCount(pair._2)
          argFlatCounter += 1
        }
      }
    }

//    val nCopies:Int = 50
//    val multipleCopies = new Array.fill[ArrayBuffer[(String, String)]]

    // Display the info:
    val wordsAvgLen = wordsLengthTracker.toArray.sum.toDouble / wordsLengthTracker.length.toDouble
    val contextsAvgLen = contextLengthTracker.toArray.sum.toDouble / contextLengthTracker.length.toDouble
    println (s"A total of ${argRawCounter} raw pairs were found, with an average length of:")
    println (s"\tcauses: $wordsAvgLen")
    println (s"\teffects: $contextsAvgLen")
    println (s"Using the (too) simple cartesian product of *individual* words, there were $argFlatCounter pairs generated.")

    // Display the pair counter info
    println ("-----------------------------------------------")
    println ("\nHighest occuring pairs:")
    for (pair <- pairCounter.sorted(descending = true).slice(0,100)) {
      println (pair.toString())
    }
    println ("-----------------------------------------------")
    var numSingleMentionsCause:Int = 0
    for (causeCount <- causeWordCounter.toSeq.unzip._2) {
      if (causeCount == 1) {
        numSingleMentionsCause += 1
      }
    }
    var numSingleMentionsEffect:Int = 0
    for (effectCount <- effectWordCounter.toSeq.unzip._2) {
      if (effectCount == 1) {
        numSingleMentionsEffect += 1
      }
    }
    var removeSingleCauseEffectCounter:Int = 0


    // Incorporate the PMI weighting
    // Load the PMI data
    val pmiFile = "/lhome/bsharp/causal/pmi/pmiCounter_N.txt"
    val pmi = Counter.loadFrom[String](new BufferedReader(new FileReader(pmiFile)))
    val pmiMaxes = CalculatePMI.findQuantileValues(pmi, nBins)

    // Initialize the new pairs and vocab counters for the weighted pairs
    val pairsWithPMIWeighting = new ArrayBuffer[(String, String)]
    val pmiContextCounter = new Counter[String]
    val pmiWordsCounter = new Counter[String]
    for ((c,e) <- pairsFinal) {
      val ce = CalculatePMI.makeString(c.toLowerCase, e.toLowerCase)
      val pmiValue = pmi.getCount(ce)
      // Check that the key is in the counter, else default to wt = 1
      var wt:Int = 1
      if (pmi.contains(ce)) {
        for (bin <- 1 until pmiMaxes.length) {
          if (pmiValue <= pmiMaxes(bin) && pmiValue > pmiMaxes(bin - 1)) {
            wt = bin + 1
          }
        }
      }
      //println ("pmiMaxes: " + pmiMaxes.mkString(", "))
      //println ("curr pmi: " + pmiValue)
      for (j <- 0 until wt) {
        //println ("  Adding... ")
        pairsWithPMIWeighting.append((c,e))
        pmiWordsCounter.incrementCount(c)
        pmiContextCounter.incrementCount(e)
      }

    }

    // Printing to files:
    // Print the main output file
    // Added in the PMI 5/17/16
    // TODO: make an option~!
//    for ((c, e) <- pairsWithPMIWeighting) {
//      output.println(s"$c $e")
//    }
//    println (s"Before PMI weighting: ${pairsFinal.length} pairs, after: ${pairsWithPMIWeighting.length} pairs")

//    // Print the vocabulary files
//    for (pair <- pmiWordsCounter.toSeq) {
//      outputWV.println(s"${pair._1} ${pair._2}")
//    }
//    for (pair <- pmiContextCounter.toSeq) {
//      outputCV.println(s"${pair._1} ${pair._2}")
//    }

    for ((c,e) <- pairsFinal) {
      if (causeWordCounter.getCount(c) >= threshold && effectWordCounter.getCount(e) >= threshold) {
        output.println(s"$c $e")
      } else {
        removeSingleCauseEffectCounter += 1
      }
    }
    println (s"Out of ${causeWordCounter.size} cause words, $numSingleMentionsCause have only one occurrence.")
    println (s"Out of ${effectWordCounter.size} effect words, $numSingleMentionsEffect have only one occurrence.")
    println (s"Out of ${pairsFinal.size} total pairs, removing using a threshold of $threshold for causes and effects would remove $removeSingleCauseEffectCounter pairs.")

    // Print the vocabulary files
    for (pair <- wordsCounter.toSeq) {
      outputWV.println(s"${pair._1} ${pair._2}")
    }
    for (pair <- contextCounter.toSeq) {
      outputCV.println(s"${pair._1} ${pair._2}")
    }

    // Housekeeping
    output.close()
    outputWV.close()
    outputCV.close()
  }

}

object SwitchArgs extends App {

  // Command line call
  def exe(cmd:String) {
    println("Running command " + cmd)
    val exitCode = cmd.!
    if(exitCode != 0)
      throw new RuntimeException("ERROR: failed to execute command " + cmd)
  }
  val prefix = "/lhome/bsharp/causal/goldbergInput/replicate_oct7"
  val direction = "c2e_noPMI"
  val fn = s"${prefix}_$direction.contexts"
  //val fn = "/lhome/bsharp/causal/goldbergInput/agiga+wiki_mar30_AllLemmas_causal_threshold0.contexts"
  //val prefix = s"/lhome/bsharp/causal/goldbergInput/a+w_mar30_NV" // last used
//  val prefix = s"/lhome/bsharp/causal/goldbergInput/replicate_oct7_e2c"
//  val suffix = "thr0_pmi5NF"
//  val fn = s"${prefix}_causal_${suffix}.contexts"

  //val pw = new PrintWriter("/lhome/bsharp/causal/goldbergInput/agiga+wiki_mar30_AllLemmas_effectToCause_threshold0.contexts")
//  val pw = new PrintWriter(s"${prefix}_e2c_${suffix}.contexts")
  val pw = new PrintWriter("/lhome/bsharp/causal/goldbergInput/replicate_oct7_e2c_noPMI.contexts")
  val source = scala.io.Source.fromFile(fn)
  val lines = source.getLines()
  for {
    line <- lines
    args = line.split(" ")
    c = args(0)
    e = args(1)
  } pw.println(s"$e $c")
  pw.close()
  source.close()

//  exe(s"cp ${prefix}_causal_${suffix}.cv ${prefix}_e2c_${suffix}.wv")
//  exe(s"cp ${prefix}_causal_${suffix}.wv ${prefix}_e2c_${suffix}.cv")
  exe(s"cp ${prefix}_$direction.cv ${prefix}_e2c_noPMI.wv")
  exe(s"cp ${prefix}_$direction.wv ${prefix}_e2c_noPMI.cv")
}