package extractionUtils

import edu.arizona.sista.embeddings.DirectEval

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

/**
  * Created by bsharp on 4/1/16.
  */
object StatsUtils {

  // Run stats using bootstrap resampling to determine the p-value
  def runStats(a:Seq[Double], b:Seq[Double], nSamples:Int, randomSeed:Int = 426):Double = {
    val randA = new Random(randomSeed)
    val randB = new Random(randomSeed + 1)
    val nItems = a.length
    assert (b.length == nItems)

    var aBetter:Double = 0.0

    for (i <- 0 until nSamples) {
      var aSample:Double = 0.0
      var bSample:Double = 0.0

      // Generate a sample from both inputs
      for (j <- 0 until nItems) {
        val aIndex = randA.nextInt(nItems)
        val bIndex = randB.nextInt(nItems)
        aSample += a(aIndex)
        bSample += b(bIndex)
      }

      // Determine which is higher
      if (aSample > bSample) aBetter += 1.0
      if (aSample == bSample) {
        println ("Eek! a tie in the sampling!")
        aBetter += 0.5
      }
    }

    aBetter / nSamples.toDouble
  }

  // Run stats using bootstrap resampling to determine the p-value
  def runStatsAvgMRR(a:Seq[((Double, Int), Int)], b:Seq[((Double, Int), Int)], nSamples:Int, randomSeed:Int = 426):Double = {
    val randA = new Random(randomSeed)
    val randB = new Random(randomSeed + 1)
    val nItems = a.length
    assert (b.length == nItems)

    var aBetter:Double = 0.0

    for (i <- 0 until nSamples) {
      val aSample = new ArrayBuffer[((Double, Int), Int)]
      val bSample = new ArrayBuffer[((Double, Int), Int)]

      // Generate a sample from both inputs
      for (j <- 0 until nItems) {
        val aIndex = randA.nextInt(nItems)
        val bIndex = randB.nextInt(nItems)
        aSample.append(a(aIndex))
        bSample.append(b(bIndex))
      }

      // Sort each
      val sortedA = aSample.sortBy(- _._1._1)
      val sortedB = bSample.sortBy(- _._1._1)

      // Find the avgMRR of each
      val (avgMRRA, _) = DirectEval.avgMRR[Int](sortedA, relevantLabel = DirectEval.TARGET_LABEL)
      val (avgMRRB, _) = DirectEval.avgMRR[Int](sortedB, relevantLabel = DirectEval.TARGET_LABEL)

      // Determine which is higher
      if (avgMRRA > avgMRRB) aBetter += 1.0
      if (avgMRRA == avgMRRB) {
        println ("Eek! a tie in the sampling!")
        aBetter += 0.5
      }
    }

    aBetter / nSamples.toDouble
  }

  // each sequence is of (score, label)
  def runStatsMAP(a:Seq[(Double, Int)], b:Seq[(Double, Int)], nSamples:Int, randomSeed:Int = 426): Double = {
    val randA = new Random(randomSeed)
    val randB = new Random(randomSeed + 1)
    val nItems = a.length

    assert (b.length == nItems)

    var aBetter:Double = 0.0

    for (i <- 0 until nSamples) {
      val aSample = new ArrayBuffer[(Double, Int)]
      val bSample = new ArrayBuffer[(Double, Int)]

      // Generate a sample from both inputs
      for (j <- 0 until nItems) {
        val aIndex = randA.nextInt(nItems)
        val bIndex = randB.nextInt(nItems)
        aSample.append(a(aIndex))
        bSample.append(b(bIndex))
      }

      // Sort each
      val sortedA = aSample.zipWithIndex.sortBy(- _._1._1)
      val sortedB = bSample.zipWithIndex.sortBy(- _._1._1)

      // Find the MAP of each
      val avgMRRA = DirectEval.MAP[Int](sortedA, relevantLabel = DirectEval.TARGET_LABEL)
      val avgMRRB = DirectEval.MAP[Int](sortedB, relevantLabel = DirectEval.TARGET_LABEL)

      // Determine which is higher
      if (avgMRRA > avgMRRB) aBetter += 1.0
      if (avgMRRA == avgMRRB) {
        println ("Eek! a tie in the sampling!")
        aBetter += 0.5
      }
    }

    aBetter / nSamples.toDouble
  }



}
