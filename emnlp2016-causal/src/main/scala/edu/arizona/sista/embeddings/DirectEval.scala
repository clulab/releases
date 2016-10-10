package edu.arizona.sista.embeddings

import java.io.{BufferedReader, FileReader, PrintWriter}

import _root_.extractionUtils.{CausalAlignment, TranslationMatrixLimited}
import edu.arizona.sista.qa.word2vec.Word2vec
import edu.arizona.sista.struct.{Counter, Lexicon}
import edu.arizona.sista.utils.StringUtils

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

/**
  * Created by bsharp on 3/31/16.
  */
object DirectEval {

  val TARGET_LABEL:Int = 1
  val OTHER_LABEL:Int = 0

  // Load the (e1, e2, label) pairs from the input file
  def loadPairs(fn:String, label:Int):Seq[(String, String, Int)] = {
    val pairs = new ArrayBuffer[(String, String, Int)]

    val lines = scala.io.Source.fromFile(fn, "UTF-8").getLines()
    // Split the lines and add the pair
    for {
      line <- lines
      pair = line.split("\t")
      e1 = pair(0)
      e2 = pair(1)
    } pairs.append((e1, e2, label))

    pairs
  }

  // Takes the pairs (e1, e2, label) and returns a Seq of (cosineSimilarity, label)
  def calculateAlignments (pairs: Seq[(String, String, Int)], matrix:TranslationMatrixLimited, lambda:Double): Seq[(Double, Int)] = {
    val alignments = new ArrayBuffer[(Double, Int)]

    // Iterate through the pairs and find the cosine similarity for each, given the vectors
    for {
      pair <- pairs
      e1 = pair._1 // src
      e2 = pair._2 // dst
      label = pair._3
      prob = matrix.prob(Array(e1.toLowerCase()), Array(e2.toLowerCase()), lambda)
    } alignments.append((prob, label))

    alignments
  }

  // Takes the pairs (e1, e2, label) and returns a Seq of (cosineSimilarity, label)
  def calculateCosineSims (pairs: Seq[(String, String, Int)], w2v1:Word2vec, w2v2:Word2vec): Seq[(Double, Int)] = {
    val sims = new ArrayBuffer[(Double, Int)]

    // Iterate through the pairs and find the cosine similarity for each, given the vectors
    for {
      pair <- pairs
      e1 = pair._1
      e2 = pair._2
      label = pair._3
      sim = cosineSimilarity(w2v1, w2v2, e1, e2)
    } sims.append((sim, label))

    sims
  }

  // Methods from Word2Vec, but altered to query two different sets of vectors
  def cosineSimilarity (w2v1:Word2vec, w2v2: Word2vec, e1:String, e2: String): Double = {
    val v1o = w2v1.matrix.get(e1)
    if(v1o.isEmpty) return -1
    val v2o = w2v2.matrix.get(e2)
    if(v2o.isEmpty) return -1
    dotProduct(v1o.get, v2o.get)
  }

  // Method from Word2Vec (unchanged)
  private def dotProduct(v1:Array[Double], v2:Array[Double]):Double = {
    assert(v1.length == v2.length) //should we always assume that v2 is longer? perhaps set shorter to length of longer...
    var sum = 0.0
    var i = 0
    while(i < v1.length) {
      sum += v1(i) * v2(i)
      i += 1
    }
    sum
  }

  def interpolate(s1:Seq[(Double, Int)], s2:Seq[(Double, Int)], lambda:Double): Seq[(Double, Int)] =
  for {
    i <- s1.indices
    label = s1(i)._2
    a = s1(i)._1
    b = s2(i)._1
    interpolated = (a * lambda) + (b * (1.0 - lambda))
  } yield (interpolated, label)

  // Makes a counter for each of the testing pairs for how many times that pair occurs in extracted training data
  def makeBinnedCounts(extractedA:Seq[String], extractedB:Seq[String], pairs:Seq[(String, String, Int)]): Counter[Int] = {
    val counter = new Counter[Int]
    for(pIdx <- pairs.indices) {
      val p = pairs(pIdx)
      val e1 = p._1
      val relevantIndices = getIndices(e1, extractedA)
      val e2 = p._2
      val numMatching = countMatches(e2, extractedB, relevantIndices)
      counter.setCount(pIdx, numMatching)
      println (s"SemEval pair $pIdx complete.")
    }

    counter
  }

  // Helper method to find the indices of sequence elements which contain a string, s
  def getIndices(s:String, seq:Seq[String]): Seq[Int] = {
    // Filter by the elements that contain the string
    val contains = seq.zipWithIndex.filter(item => item._1.contains(s))
    // Return the indices
    contains.unzip._2
  }

  // Check the elements at each index to see if the element contains the string, return the number of matches
  def countMatches(s:String, seq:Seq[String], indices:Seq[Int]): Int = {
    // Only look at the requested elements of the sequence
    val relevant = seq.zipWithIndex.filter(item => indices.contains(item._2))
    // Count how many of the relevant pairs contain s
    relevant.unzip._1.count(str => str.contains(s))
  }

  // Uses the counter to score each pair: the score is the number of matches for each indexed pair
  // (Note: the pair index is the counter key)
  def scoreByMatching(c:Counter[Int], pairs:Seq[(String, String, Int)]): Seq[(Double, Int)] = {
    val seq = c.toSeq

    // Make the 'scored' list to return, where the score is the number of times a pair occurred
    val scored = new Array[(Double, Int)](pairs.length)
    for (i <- pairs.indices) {
      scored(i) = (c.getCount(i), pairs(i)._3)    // (numOccurrences, label)
    }

    scored
  }

  // Finds the tie-aware average MRR of the items with relevant label
  def avgMRR[L](sorted:Seq[((Double, L), Int)], relevantLabel:L): (Double, Seq[Double]) = {

    //val MRRs = new Array[Double](sorted.length)
    val MRRs = new ArrayBuffer[Double]
    var MRRSum:Double = 0.0
    var numTarget:Double = 0.0

    // Organize the scores in groups (so that all ties are together), and sorted by descending score
    val groupedByScore: Seq[(Double, Seq[((Double, L), Int)])] = sorted.groupBy(_._1._1).toSeq.sortBy(- _._1)

    // For each tie group:
    for (i <- groupedByScore.indices) {
      // Retrieve the scored items in that tie group
      val itemsInGroup: Seq[((Double, L), Int)] = groupedByScore(i)._2
      for (item <- itemsInGroup) {
        val similarity = item._1
        val label = similarity._2
        val currMRR:Double = 1.0 / (i+1).toDouble
        // Store the MRR for the item in the correct position
        val originalIndex = item._2
        //MRRs(originalIndex) = currMRR

        // Add relevant MRRs to the running total
        if (label == relevantLabel) {
          MRRs.append(currMRR)
          MRRSum += currMRR
          numTarget += 1.0
        }
      }
    }

    (MRRSum / numTarget, MRRs)
  }

  // Takes the sorted, scored items a Seq[((score, label), originalIndex)], also takes the label for relevant items
  // Returns the tie-aware MAP as a Double
  def MAP[L](sorted:Seq[((Double, L), Int)], relevantLabel:L): Double = {
    val avgPrecisions = new ArrayBuffer[Double]

    // Organize the scores in groups (so that all ties are together), and sorted by descending score
    val groupedByScore: Seq[(Double, Seq[((Double, L), Int)])] = sorted.groupBy(_._1._1).toSeq.sortBy(- _._1)

    // For each tie group:
    for (i <- groupedByScore.indices) {
      // Retrieve the scored items in that tie group
      val itemsInGroup: Seq[((Double, L), Int)] = groupedByScore(i)._2
      // Check to see if there are any relevant items in the group
      val numRelevant = itemsInGroup.unzip._1.count(scored => scored._2 == relevantLabel)
      if (numRelevant > 0) {
        // Make a slice
        val slice = groupedByScore.slice(0, i + 1).unzip._2.flatten.unzip._1

        // Find the average precision for this slice
        val sliceAvgPrecision = findPrecision(slice, relevantLabel)

        // 5/23 modification: Add the average precision for the slice once for EACH relevant item in the tie-group!
        for (j <- 0 until numRelevant){
          avgPrecisions.append(sliceAvgPrecision)
        }
      }

    }

//    for (i <- sorted.indices) {
//      val currPair = sorted(i)._1
//      val currLabel = currPair._2
//      // If we're at a relevant result
//      if (currLabel == relevantLabel) {
//        // Find the precision for everything up to and including this result
//        val slice = sorted.slice(0, i + 1)
//        val slicePrecision = findPrecision(slice.unzip._1, relevantLabel)
//        avgPrecisions.append(slicePrecision)
//      }
//    }

    // Find and return the mean
    avgPrecisions.sum / avgPrecisions.length.toDouble
  }

  def MAPNotTieAware[L](sorted:Seq[((Double, L), Int)], relevantLabel:L): Double = {
    val avgPrecisions = new ArrayBuffer[Double]

//    // Organize the scores in groups (so that all ties are together), and sorted by descending score
//    val groupedByScore: Seq[(Double, Seq[((Double, L), Int)])] = sorted.groupBy(_._1._1).toSeq.sortBy(- _._1)
//
//    // For each tie group:
//    for (i <- groupedByScore.indices) {
//      // Retrieve the scored items in that tie group
//      val itemsInGroup: Seq[((Double, L), Int)] = groupedByScore(i)._2
//      // Check to see if there are any relevant items in the group
//      val numRelevant = itemsInGroup.unzip._1.count(scored => scored._2 == relevantLabel)
//      if (numRelevant > 0) {
//        // Make a slice
//        val slice = groupedByScore.slice(0, i + 1).unzip._2.flatten.unzip._1
//
//        // Find the average precision for this slice
//        val sliceAvgPrecision = findPrecision(slice, relevantLabel)
//        avgPrecisions.append(sliceAvgPrecision)
//      }
//
//    }

    for (i <- sorted.indices) {
      val currPair = sorted(i)._1
      val currLabel = currPair._2
      // If we're at a relevant result
      if (currLabel == relevantLabel) {
        // Find the precision for everything up to and including this result
        val slice = sorted.slice(0, i + 1)
        val slicePrecision = findPrecision(slice.unzip._1, relevantLabel)
        avgPrecisions.append(slicePrecision)
      }
        }
    avgPrecisions.sum / avgPrecisions.length.toDouble
  }

  def findPrecision[L](slice:Seq[(Double, L)], relevantLabel:L): Double = {
    val total:Double = slice.length
    if (total == 0.0) throw new RuntimeException ("Error: empty slice!")
    val numRelevant:Double = slice.count(pair => pair._2 == relevantLabel)

    numRelevant / total
  }

  // Takes the reverse SORTED labeled cosine similarities and returns values for a precision-recall curve
  // TIE-AWARE
  def recallPrecisionCurve[L](sorted:Seq[(Double, L)], relevantLabel:L): Seq[(Double, Double)] = {
    val curvePoints = new ArrayBuffer[(Double, Double)]
    val numPairs:Double = sorted.length.toDouble
    val numRelevant:Double = sorted.count(item => item._2 == relevantLabel)


    // Organize the scores in groups (so that all ties are together), and sorted by descending score
    val groupedByScore: Seq[(Double, Seq[(Double, L)])] = sorted.groupBy(_._1).toSeq.sortBy(- _._1) //.groupBy(_._1._1).toSeq.sortBy(- _._1)

    // For each tie group:
    for (i <- groupedByScore.indices) {
      // Retrieve the scored items in that tie group
      val itemsInGroup: Seq[(Double, L)] = groupedByScore(i)._2
      // Check to see if there are any relevant items in the group
      val numRelevantInGroup = itemsInGroup.count(scored => scored._2 == relevantLabel)
      if (numRelevantInGroup > 0) {
        // Make a slice
        val slice: Seq[(Double, L)] = groupedByScore.slice(0, i + 1).unzip._2.flatten
        val sliceSize:Double = slice.length
        val numRelevantInSlice:Double = slice.count(item => item._2 == relevantLabel)

        // Recall
        val sliceRecall:Double = numRelevantInSlice / numRelevant
        //println (s"Recall: $numRelevantInSlice / $numRelevant = $sliceRecall")

        // Precision
        val slicePrecision:Double = numRelevantInSlice / sliceSize
        //println (s"Precision: $numRelevantInSlice / $sliceSize = $slicePrecision")

        // Store the recall and precision after this item
        curvePoints.append((sliceRecall, slicePrecision))
      }

    }

//    for (i <- sorted.indices) {
//      val slice = sorted.slice(0, i+1)
//      val sliceSize:Double = slice.length
//      val numRelevantInSlice:Double = slice.count(item => item._2 == relevantLabel)
//
//      // Recall
//      val sliceRecall:Double = numRelevantInSlice / numRelevant
//
//      // Precision
//      val slicePrecision:Double = numRelevantInSlice / sliceSize
//
//      // Store the recall and precision after this item
//      curvePoints.append((sliceRecall, slicePrecision))
//    }

    curvePoints
  }

  // Takes the reverse SORTED labeled cosine similarities and returns values for a precision-yield curve, using thresholds
  // TIE-AWARE
  def precisionRecallCurveByThreshold[L](sorted:Seq[(Double, L)], relevantLabel:L, threshold:Double): Seq[(Double, Double, Double)] = {
    val curvePoints = new ArrayBuffer[(Double, Double, Double)]
    val numPairs:Double = sorted.length.toDouble
    val numRelevant = sorted.filter(p => p._2 == relevantLabel).length

    val cutoffs = -1.0 to 1.0 by threshold

    for (cutoff <- cutoffs.reverse) {

      // Keep items that are scored higher than the current cutoff
      val slice = sorted.filter(item => item._1 >= cutoff)
      //val slice = sorted.slice(0, cutoff)
      val sliceSize:Double = slice.length.toDouble
      //println (s"Number of items greater than $cutoff: $sliceSize")

      // Recall
      val relevantInSlice = slice.filter(p => p._2 == relevantLabel).length
      val sliceRecall:Double = relevantInSlice.toDouble / numRelevant.toDouble

      // Precision
      val slicePrecision:Double = relevantInSlice.toDouble / sliceSize


      // Store the recall and precision after this item
      curvePoints.append((cutoff, sliceRecall, slicePrecision))
    }

    curvePoints
  }

  def saveGraphAsTSV(points:Seq[(Double, Double)], fn:String): Unit = {
    val pw = new PrintWriter(fn)
    points.foreach(p => pw.println(s"${p._1}\t${p._2}"))
    pw.close()
  }


  def zipPointsAndSave(lines:Seq[Seq[(Double, Double)]], fn:String): Unit = {
    val numLines = lines.length
    val out = new ArrayBuffer[Seq[Double]]
    val header = new ArrayBuffer[String]
    header.append("Recall")
    for (lineIdx <- lines.indices) {
      header.append(s"Precision_$lineIdx")
      val line = lines(lineIdx)
      for (point <- line) {
        val pointOut = Array.fill[Double](numLines + 1)(-1.0)
        val recall = point._1
        val precision = point._2
        pointOut(0) = recall
        pointOut(lineIdx + 1) = precision
        out.append(pointOut)
      }
    }

    val pw = new PrintWriter(fn)
    pw.println(header.mkString("\t"))
    for (point <- out) {
      var sOut = new ArrayBuffer[String]
      for (elem <- point) {
        if (elem == -1.0) sOut.append("")
        else sOut.append(s"$elem")
      }
      pw.println(sOut.mkString("\t"))
    }

    pw.close()
  }

  def displayCounterDistribution(c:Counter[Int], pairs:Seq[(String, String, Int)], relevantLabel:Int): Unit = {
    // Count the number of pairs in each bin
    val unsortedDist = new ArrayBuffer[(Double, Int, Int)]   // Will be (numOccurrences, numItemsWithThatManyOccurrence, numItemsRelevant)
    val seq = c.toSeq

    // Find distribution info (i.e. bin and count)
    val grouped = seq.groupBy(_._2)
    for ((numOccurrences, elems) <- grouped) {
      val numRelevant = elems.count(e => pairs(e._1)._3 == relevantLabel)
      unsortedDist.append((numOccurrences, elems.length, numRelevant))
    }

    // Sort Distribution Info and Display
    val sortedDistByNumOccur = unsortedDist.sortBy(_._1)
    println ("Distribution of number of occurrences of test pairs in train data")
    for ((numOccur, nItems, numRel) <- sortedDistByNumOccur) {
      println(s"$numOccur occurrences: $nItems items ($numRel relevant)")
    }

  }

  def errorAnalysis(sorted:Seq[((Double, Int), Int)], pairs:Seq[(String, String, Int)], sliceSize:Double): Unit = {
    // Find the slice
    val slice = sorted.slice(0, math.round(sorted.length * sliceSize).toInt)
    for (i <- slice.indices) {
      val item = slice(i)
      val itemScore = item._1._1
      val origIndex = item._2
      val origPair = pairs(origIndex)
      val e1 = origPair._1
      val e2 = origPair._2
      val label = origPair._3
      val corr = if (label == TARGET_LABEL) "*" else ""
      println (s"$corr Item $i: ($e1, $e2) Label: $label ($itemScore)")
    }
  }

  // Save the SemEval pairs as indices (using the lexicons from the keras input)
  def saveForKeras(outPrefix:String, pairs:Seq[(String, String, Int)], srcLexFilename:String, dstLexFilename:String): Unit = {
    val srcLex = Lexicon.loadFrom[String](srcLexFilename)
    val dstLex = Lexicon.loadFrom[String](dstLexFilename)

    val srcOut = new PrintWriter(outPrefix + "_src.csv")
    val dstOut = new PrintWriter(outPrefix + "_dst.csv")
    val labOut = new PrintWriter(outPrefix + "_lab.csv")

    var oovTarget = 0
    var oovOther = 0
    val padding = Array.fill[Int](30)(0)
    // 866,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
    for ((e1, e2, lab) <- pairs) {
      val e1Index = srcLex.get(e1).getOrElse(0)
      val e2Index = dstLex.get(e2).getOrElse(0)
      if (e1Index == 0 || e2Index == 0) {
        if (lab == 1) oovTarget += 1
        if (lab == 0) oovOther += 1
      }
      srcOut.println((Array(e1Index) ++ padding).mkString(",")) // Add expected padding
      dstOut.println((Array(e2Index) ++ padding).mkString(","))
      labOut.println(lab)
    }

    // Housekeeping
    srcOut.close()
    dstOut.close()
    labOut.close()

    println (s"A total of ${pairs.length} SemEval pairs were processed and stored to: ")
    println (s"\t${outPrefix}_src.csv and ${outPrefix}_dst.csv")
    println (s" Of these, ${oovTarget + oovOther} (target: $oovTarget, other: $oovOther) were OOV!")
  }

  def getKerasPredictions(fn:String):Seq[(Double, Int)] = {
    val preds = new ArrayBuffer[(Double, Int)]

    val source = scala.io.Source.fromFile(fn)
    val lines = source.getLines()
    for {
      line <- lines
      sp = line.split("\t")
    } preds.append((sp(0).toDouble, sp(1).toDouble.toInt))

    // Housekeeping
    source.close()

    preds
  }


  def main(args:Array[String]): Unit = {

    val props = StringUtils.argsToProperties(args)

    // Step 1: Loading
    // Step 1a: Load in the CUSTOMIZED vectors, target for e1 and context for e2
    val customTargetW2V = new Word2vec(props.getProperty("vectors.custom_target"))
    val customContextW2V = new Word2vec(props.getProperty("vectors.custom_context"))

    // BiDirectional Experiment: Load the vectors which were trained in the opposite direction, i.e. effect->cause
    val customE2CEffectW2V = new Word2vec(props.getProperty("vectors.custom_effect.e2c"))
    val customE2CCauseW2V = new Word2vec(props.getProperty("vectors.custom_cause.e2c"))

    // Step 1a: Load in the PMI vectors, target for e1 and context for e2
    val pmiTargetW2V = new Word2vec(props.getProperty("vectors.pmi_target"))
    val pmiContextW2V = new Word2vec(props.getProperty("vectors.pmi_context"))

    // BiDirectional Experiment: Load the vectors which were trained in the opposite direction, i.e. effect->cause
    val pmiE2CEffectW2V = new Word2vec(props.getProperty("vectors.pmi_effect.e2c"))
    val pmiE2CCauseW2V = new Word2vec(props.getProperty("vectors.pmi_cause.e2c"))


    // 1b: Load in the comparison vectors (i.e. vanilla w2v)
    val comparisonW2V = new Word2vec(props.getProperty("vectors.comparison"))

    // 1c: Load in an Alignment Matrix
    val matrixPrefix = props.getProperty("matrix.prefix")
    val translationMatrix = new TranslationMatrixLimited(s"$matrixPrefix.matrix", s"$matrixPrefix.priors")

    // 1c: Load in the Vanilla Alignment Matrix
    val matrixPrefixVanilla = props.getProperty("matrix.prefix.baseline")
    val translationMatrixVanilla = new TranslationMatrixLimited(s"$matrixPrefixVanilla.matrix", s"$matrixPrefixVanilla.priors")

    // 1d: Load the extracted pairs for finding the counting baseline
    val extractedPairsDir = props.getProperty("extracted_pairs.directory")
    val extractedPairs = CausalAlignment.loadData(extractedPairsDir, "argsC", lenThreshold = 4)

    // Step 2: Load in the pairs, keeping track of whether they are the target relation or not
    val targetPairs = loadPairs(props.getProperty("input.target"), label = TARGET_LABEL)
    val nTagrgt = targetPairs.length
    println (s"* $nTagrgt target pairs loaded.")
    val otherPairsAll = loadPairs(props.getProperty("input.other"), label = OTHER_LABEL)

    // Create dev and test partitions of target and other pairs
    val random = new Random
    random.setSeed(426)
    val targetShuffled = random.shuffle(targetPairs)
    val otherShuffled = random.shuffle(otherPairsAll)

    // Create a dev and test fold
    //val (targetDevZ, _) = targetShuffled.zipWithIndex.partition(e => e._2 % 2 == 0)
    val (_, targetTestZ) = targetShuffled.zipWithIndex.partition(e => e._2 % 2 == 0)
//    val targetDev = targetDevZ.unzip._1
//    val devLength = targetDev.length
//    println (s"There are $devLength dev pairs loaded! First pair: ${targetDev.head}")
    val targetTest = targetTestZ.unzip._1
    val testLength = targetTest.length
    println (s"There are $testLength TEST pairs loaded! First pair: ${targetTest.head}")

    // Step 3: Make a randomized slice of the other pairs of equal length to the target pairs
    //val (otherDevZ, _) = otherShuffled.zipWithIndex.partition(e => e._2 % 2 == 0)
    val (_, otherTestZ) = otherShuffled.zipWithIndex.partition(e => e._2 % 2 == 0)
//    val otherDev = otherDevZ.unzip._1.slice(0, devLength)
//    val otherDevLength = otherDev.length
//    println (s"There are $otherDevLength dev pairs loaded! First pair: ${otherDev.head}")
    val otherTest = otherTestZ.unzip._1.slice(0, testLength)
    val otherTestLength = otherTest.length
    println (s"There are $otherTestLength TEST pairs loaded! First pair: ${otherTest.head}")



    // Step 3b: Combine to make a direct eval set
    //val pairs = targetDev ++ otherDev
    val pairs = targetTest ++ otherTest
    println (s"* ${pairs.length} TOTAL pairs.")
    //OPTIONAL - save to the format needed to evaluate with Keras
//    val kerasDir = "/lhome/bsharp/keras/causal/"
//    saveForKeras(
//      kerasDir + "SemEvalPairs_keras_format_lenCutoff0_padTo31_TEST",
//      pairs,
//      kerasDir + "agiga_causal_mar30_lemmas.src.lex",
//      kerasDir + "agiga_causal_mar30_lemmas.dst.lex"
//    )
//    sys.exit()

    // Step 4a: Find the cosine similarity between e1 and e2 using CUSTOM vectors (with target and context vectors) AND comparison
    val cosinesCustom = calculateCosineSims(pairs, customTargetW2V, customContextW2V)
    val cosinesComparison = calculateCosineSims(pairs, comparisonW2V, comparisonW2V)
    println ("Cosine similarities calculated.")

    // Bidirectional Experiment: calculate the cosine similarities for the other direction
    val cosinesE2C = calculateCosineSims(pairs, customE2CCauseW2V, customE2CEffectW2V)
    // Also, create a linearly-interpolated mix of the two directions:
    val cosinesCustomBidir = interpolate(cosinesCustom, cosinesE2C, lambda = 0.5)

    // PMI vectors
    val cosinesPMI = calculateCosineSims(pairs, pmiTargetW2V, pmiContextW2V)
    val cosinesE2CPMI = calculateCosineSims(pairs, pmiE2CCauseW2V, pmiE2CEffectW2V)
    val cosinesPMIBidir = interpolate(cosinesPMI, cosinesE2CPMI, lambda = 0.5)

    // Step 4b: Find the causal translation probability for each pair
    val lambda: Double = 0.5
    val assns = calculateAlignments(pairs, translationMatrix, lambda)
    val OOV = assns.filter(scored => scored._1 == Double.NegativeInfinity)
    val numOOV: Double = OOV.length
    val numOOVRelevant: Double = OOV.count(scored => scored._2 == TARGET_LABEL)
    println(s"numOOV: $numOOV\tnumOOVRelevant: $numOOVRelevant (${numOOVRelevant / numOOV}) ")
    println("Alignments calculated.")

    // Step 4c: Find the vanilla translation probability for each pair
    //val lambda: Double = 0.5
    val assnsComparison = calculateAlignments(pairs, translationMatrixVanilla, lambda)
    //val OOV = assns.filter(scored => scored._1 == Double.NegativeInfinity)
    //val numOOV: Double = OOV.length
    //val numOOVRelevant: Double = OOV.count(scored => scored._2 == TARGET_LABEL)
    //println(s"numOOV: $numOOV\tnumOOVRelevant: $numOOVRelevant (${numOOVRelevant / numOOV}) ")
    println("Vanilla Alignments calculated.")

    // 4d: Find the binned counts for each pair
    val counterOut = extractedPairsDir + "/causal.counter.TEST"
    // Making and saving a new counter
//    val pairCounter = makeBinnedCounts(extractedPairs._1, extractedPairs._2, pairs)
//    val pairPW = new PrintWriter(counterOut)
//    pairCounter.saveTo(pairPW)
//    pairPW.close()
//    println ("Binning complete. Counter saved to " + counterOut)

    // Loading a counter
    val reader = new BufferedReader(new FileReader(counterOut))
    val pairCounter = Counter.loadFrom[Int](reader)
    println ("Counter loaded from " + counterOut)

    displayCounterDistribution(pairCounter, pairs, relevantLabel = TARGET_LABEL)
    val matches = scoreByMatching(pairCounter, pairs)
    val numberOOVCausal = matches.filter(_._1 == 0.0).count(_._2 == TARGET_LABEL)
    val totalOOV = matches.count(_._1 == 0.0)
    println ("Lookup Baseline -- nOOV causal: " + numberOOVCausal)
    println ("Lookup Baseline -- nOOV Total: " + totalOOV)


    // Step 4e: Load the CNN scores for each pair
    val kerasPredictionsFile = props.getProperty("keras_predictions_file")
    val kerasPredictions = getKerasPredictions(kerasPredictionsFile)
    // Check the order...
    for (i <- kerasPredictions.indices) {
      assert (kerasPredictions(i)._2 == pairs(i)._3)
    }

    // Step 5: Sort
    // W2V
    val sortedCustom = cosinesCustom.zipWithIndex.sortBy(- _._1._1)
    val customLabels = sortedCustom.unzip._1.unzip._2
    val customScores = sortedCustom.unzip._1.unzip._1
    println("customLabels: " + customLabels.mkString(", "))
    println("CustomScores: " + customScores.mkString(", "))
    // ERROR ANALYSIS
    println ("-------------------------")
    println ("Error Anaysis: CUSTOM")
    errorAnalysis(sortedCustom, pairs, 0.2)

    //sys.exit()
    // Bidirectional Experiment:
    val sortedE2C = cosinesE2C.zipWithIndex.sortBy(- _._1._1)
    val e2cLabels = sortedE2C.unzip._1.unzip._2
    val e2cScores = sortedE2C.unzip._1.unzip._1
    println("e2cLabels: " + e2cLabels.mkString(", "))
    println("e2cScores: " + e2cScores.mkString(", "))
    // ERROR ANALYSIS
    println ("-------------------------")
    println ("Error Anaysis: E2C")
    errorAnalysis(sortedE2C, pairs, 0.2)

    val sortedBidir = cosinesCustomBidir.zipWithIndex.sortBy(- _._1._1)
    val bidirLabels = sortedBidir.unzip._1.unzip._2
    val bidirScores = sortedBidir.unzip._1.unzip._1
    println("bidirLabels: " + bidirLabels.mkString(", "))
    println("bidirScores: " + bidirScores.mkString(", "))
    // ERROR ANALYSIS
    println ("-------------------------")
    println ("Error Anaysis: BIDIR")
    errorAnalysis(sortedBidir, pairs, 0.2)


    // PMI
    val sortedPMI = cosinesPMI.zipWithIndex.sortBy(- _._1._1)
    val sortedE2CPMI = cosinesE2CPMI.zipWithIndex.sortBy(- _._1._1)
    val sortedBidirPMI = cosinesPMIBidir.zipWithIndex.sortBy(- _._1._1)

    // Vanilla
    val sortedComparison = cosinesComparison.zipWithIndex.sortBy(- _._1._1)
    val comparisonLabels = sortedComparison.unzip._1.unzip._2
    val comparisonScores = sortedComparison.unzip._1.unzip._1
    println("comparisonLabels: " + comparisonLabels.mkString(", "))
    println("ComparisonScores: " + comparisonScores.mkString(", "))

    // Alignment
    val sortedTrans = assns.zipWithIndex.sortBy(- _._1._1)
    val transLabels = sortedTrans.unzip._1.unzip._2
    val transScores = sortedTrans.unzip._1.unzip._1
    println("transLabels: " + transLabels.mkString(", "))
    println("transScores: " + transScores.mkString(", "))

    // Vanilla Alignment
    val sortedTransComparison = assnsComparison.zipWithIndex.sortBy(- _._1._1)

    // Exact Matches Baseline
    val sortedMatches = matches.zipWithIndex.sortBy(- _._1._1)

    // Sorted Keras
    val sortedKeras = kerasPredictions.zipWithIndex.sortBy(- _._1._1)

    // Random Baseline
    val randomized = assns.map(scored => (random.nextDouble(), scored._2)).zipWithIndex.sortBy(-_._1._1)

    // Step 6: Evaluate
    val customMAP = MAP[Int](sortedCustom, relevantLabel = TARGET_LABEL)
    val e2cMAP = MAP[Int](sortedE2C, relevantLabel = TARGET_LABEL)
    val bidirMAP = MAP[Int](sortedBidir, relevantLabel = TARGET_LABEL)
    val PMIMAP = MAP[Int](sortedPMI, relevantLabel = TARGET_LABEL)
    val e2cPMIMAP = MAP[Int](sortedE2CPMI, relevantLabel = TARGET_LABEL)
    val bidirPMIMAP = MAP[Int](sortedBidirPMI, relevantLabel = TARGET_LABEL)
    val comparisonMAP = MAP[Int](sortedComparison, relevantLabel = TARGET_LABEL)
    val transMAP = MAP[Int](sortedTrans, relevantLabel = TARGET_LABEL)
    val transMAPcomparison = MAP[Int](sortedTransComparison, relevantLabel = TARGET_LABEL)
    //val transMAPNoTies = MAPNotTieAware[Int](sortedTrans, relevantLabel = TARGET_LABEL)
    val matchesMAP = MAP[Int](sortedMatches, relevantLabel = TARGET_LABEL)
    val kerasMAP = MAP[Int](sortedKeras, relevantLabel = TARGET_LABEL)
    val randomMAP = MAP[Int](randomized, relevantLabel = TARGET_LABEL)
    println ("MAP for Custom Vectors: " + customMAP)
    println ("MAP for E2C Vectors: " + e2cMAP)
    println ("MAP for Bidir Vectors: " + bidirMAP)
    println ("MAP for PMI Vectors: " + PMIMAP)
    println ("MAP for E2C PMI Vectors: " + e2cPMIMAP)
    println ("MAP for Bidir PMI Vectors: " + bidirPMIMAP)
    println ("MAP for Comparison (Baseline) Vectors: " + comparisonMAP)
    println(s"MAP for Translation Model with lamda of $lambda : " + transMAP)
    println(s"MAP for Vanilla Translation Model with lamda of $lambda : " + transMAPcomparison)
    //println(s"MAPNoTies for Translation Model with lamda of $lambda : " + transMAPNoTies)
    println ("MAP for counting Matches: " + matchesMAP)
    println ("MAP for Keras: " + kerasMAP)
    println ("MAP for Random: " + randomMAP)


    val randomRPCurve = recallPrecisionCurve[Int](randomized.unzip._1, relevantLabel = TARGET_LABEL)
    val customRPCurve = recallPrecisionCurve[Int](sortedCustom.unzip._1, relevantLabel = TARGET_LABEL)
    val comparisonRPCurve = recallPrecisionCurve[Int](sortedComparison.unzip._1, relevantLabel = TARGET_LABEL)
    val transRPCurve = recallPrecisionCurve[Int](sortedTrans.unzip._1, relevantLabel = TARGET_LABEL)
    val matchingRPCurve = recallPrecisionCurve[Int](sortedMatches.unzip._1, relevantLabel = TARGET_LABEL)
    val e2cRPCurve = recallPrecisionCurve[Int](sortedE2C.unzip._1, relevantLabel = TARGET_LABEL)
    val bidirRPCurve = recallPrecisionCurve[Int](sortedBidir.unzip._1, relevantLabel = TARGET_LABEL)
    val kerasRPCurve = recallPrecisionCurve[Int](sortedKeras.unzip._1, relevantLabel = TARGET_LABEL)
    val PMIRPCurve = recallPrecisionCurve[Int](sortedPMI.unzip._1, relevantLabel = TARGET_LABEL)
    val e2cPMIRPCurve = recallPrecisionCurve[Int](sortedE2CPMI.unzip._1, relevantLabel = TARGET_LABEL)
    val bidirPMIRPCurve = recallPrecisionCurve[Int](sortedBidirPMI.unzip._1, relevantLabel = TARGET_LABEL)
    val transComparisonRPCurve = recallPrecisionCurve[Int](sortedTransComparison.unzip._1, relevantLabel = TARGET_LABEL)

    //saveGraphAsTSV(customRPCurve, "/home/bsharp/causal/customRPCurve.tsv")
    //saveGraphAsTSV(comparisonRPCurve, "/home/bsharp/causal/comparisonRPCurve.tsv")
    //zipPointsAndSave(Seq(customRPCurve, comparisonRPCurve, transRPCurve, randomRPCurve, matchingRPCurve, e2cRPCurve, bidirRPCurve, PMIRPCurve, e2cPMIRPCurve, bidirPMIRPCurve), "/lhome/bsharp/causal/allRPCurve5.tsv")
    //zipPointsAndSave(Seq(customRPCurve, comparisonRPCurve, bidirRPCurve, PMIRPCurve, bidirPMIRPCurve), "/lhome/bsharp/causal/allRPCurve6.tsv")
    //zipPointsAndSave(Seq(comparisonRPCurve, matchingRPCurve, customRPCurve, bidirRPCurve, PMIRPCurve, bidirPMIRPCurve, transRPCurve, kerasRPCurve), "/lhome/bsharp/causal/allRPCurve10Test.tsv")
    zipPointsAndSave(Seq(transRPCurve), "/lhome/bsharp/causal/transPRCurve_Test.tsv")
    zipPointsAndSave(Seq(randomRPCurve), "/lhome/bsharp/causal/randomPRCurve_Test.tsv")
//    zipPointsAndSave(Seq(customRPCurve, comparisonRPCurve, transRPCurve, randomRPCurve), "/home/bsharp/causal/allRPCurve_noMatches.tsv")

//    sys.exit()
    // Step 6b: Statistical Significance of the MAPs
    val n:Int = 10000
//    val pValueMAP = StatsUtils.runStatsMAP(cosinesCustom, cosinesComparison, n)
//    println ("---------------------------------------------------------------------------------------")
//    println ("\tcEmbed vs. Vanilla")
//    println (s"\tBOOTSTRAP RESAMPLING of MAPs with $n iterations: p-value = ${1.0 - pValueMAP}")
//    println ("---------------------------------------------------------------------------------------")
//
//    val pValueMAP2 = StatsUtils.runStatsMAP(cosinesCustomBidir, cosinesComparison, n)
//    println ("---------------------------------------------------------------------------------------")
//    println ("\tcEmbedBidir vs. Vanilla")
//    println (s"\tBOOTSTRAP RESAMPLING of MAPs with $n iterations: p-value = ${1.0 - pValueMAP2}")
//    println ("---------------------------------------------------------------------------------------")

//    val pValueMAP3 = StatsUtils.runStatsMAP(assns, cosinesComparison, n)
//    println ("---------------------------------------------------------------------------------------")
//    println ("\tcTrans vs. Vanilla")
//    println (s"\tBOOTSTRAP RESAMPLING of MAPs with $n iterations: p-value = ${1.0 - pValueMAP3}")
//    println ("---------------------------------------------------------------------------------------")
//
//    val pValueMAP4 = StatsUtils.runStatsMAP(matches, cosinesComparison, n)
//    println ("---------------------------------------------------------------------------------------")
//    println ("\tcLookup vs. Vanilla")
//    println (s"\tBOOTSTRAP RESAMPLING of MAPs with $n iterations: p-value = ${1.0 - pValueMAP4}")
//    println ("---------------------------------------------------------------------------------------")
//
//    val pValueMAP5 = StatsUtils.runStatsMAP(kerasPredictions, cosinesComparison, n)
//    println ("---------------------------------------------------------------------------------------")
//    println ("\tcCNN vs. Vanilla")
//    println (s"\tBOOTSTRAP RESAMPLING of MAPs with $n iterations: p-value = ${1.0 - pValueMAP5}")
//    println ("---------------------------------------------------------------------------------------")
//
//    val pValueMAP6 = StatsUtils.runStatsMAP(cosinesPMI, cosinesComparison, n)
//    println ("---------------------------------------------------------------------------------------")
//    println ("\tcEmbedNoise vs. Vanilla")
//    println (s"\tBOOTSTRAP RESAMPLING of MAPs with $n iterations: p-value = ${1.0 - pValueMAP6}")
//    println ("---------------------------------------------------------------------------------------")
//
//    val pValueMAP7 = StatsUtils.runStatsMAP(cosinesPMIBidir, cosinesComparison, n)
//    println ("---------------------------------------------------------------------------------------")
//    println ("\tcEmbedNoiseBidir vs. Vanilla")
//    println (s"\tBOOTSTRAP RESAMPLING of MAPs with $n iterations: p-value = ${1.0 - pValueMAP7}")
//    println ("---------------------------------------------------------------------------------------")

    //    val pValueMAP = StatsUtils.runStatsMAP(cosinesComparison, None, n)
//    println ("---------------------------------------------------------------------------------------")
//    println ("\tVanilla vs. Random")
//    println (s"\tBOOTSTRAP RESAMPLING of MAPs with $n iterations: p-value = ${1.0 - pValueMAP}")
//    println ("---------------------------------------------------------------------------------------")

//    val pValueMAP = StatsUtils.runStatsMAP(cosinesCustom, None, n)
//    println ("---------------------------------------------------------------------------------------")
//    println ("\tCustom vs. Random")
//    println (s"\tBOOTSTRAP RESAMPLING of MAPs with $n iterations: p-value = ${1.0 - pValueMAP}")
//    println ("---------------------------------------------------------------------------------------")

//    val pValueMAP8 = StatsUtils.runStatsMAP(assnsComparison, cosinesComparison, n)
//    println ("---------------------------------------------------------------------------------------")
//    println ("\tcTrans vs. Vanilla")
//    println (s"\tBOOTSTRAP RESAMPLING of MAPs with $n iterations: p-value = ${1.0 - pValueMAP3}")
//    println ("---------------------------------------------------------------------------------------")


  }


}
