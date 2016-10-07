package edu.arizona.sista.embeddings.postprocessing

import java.io.{BufferedReader, FileReader, PrintWriter}

import edu.arizona.sista.struct.Counter

import scala.collection.Set
import scala.collection.mutable.ArrayBuffer
import scala.io.Source

/**
  * Created by bsharp on 5/14/16.
  */
object CalculatePMI {

  def splitArgs(ce:String):(String, String) = {
    val split = ce.split("__")
    if (split.length != 2) println (ce)
    val c = split(0).toLowerCase
    val e = split(1).toLowerCase
    (c,e)
  }

  def makeString(a:String, b:String):String = s"${a}__$b"

  def makeSetBidirectional(in:Set[String]):Set[String] = {
    val switched = new ArrayBuffer[String]
    for {
      ce <- in
      (c,e) = splitArgs(ce)
      ec = makeString(e,c)
    } {
      if (ce == "allegation__tension") {
        println ("When making set: ce=" + ce + " and ec=" + ec)
      }
      switched.append(ec)
    }

    switched.toSet ++ in
  }

  def loadBigCounter(fn:String, filterBy:Set[String]): Counter[String] = {
    val counter = new Counter[String]
    val filterSet = makeSetBidirectional(filterBy)
    var lineCounter = 0
    for(line <- Source.fromFile(fn).getLines()){
      //println(line)
      if (lineCounter > 1) {
        //println (line)
        val fields = line.split(" ")
        val count = fields(0).toDouble
        val ce = fields(1).trim()
        val (c,e) = splitArgs(ce)
        val ec = makeString(e,c)
        if (filterSet.contains(ce) || filterSet.contains(ec)) {
          counter.setCount(ce, count)
        }
      }
      lineCounter += 1
    }

    counter
  }

  def convertCounterKeysToLowerCase(c:Counter[String]): Counter[String] = {

    val out = new Counter[String]
    val items = c.toSeq
    for {
      (k,v) <- items
    } out.incrementCount(k.toLowerCase(), v)
    out
  }

  // Returns the MAX value for each of the bins, such that each bin will have an equal number of data points
  def findQuantileValues(c:Counter[String], nBins:Int):Seq[Double] = {
    val sortedItems = c.toSeq.sortBy(_._2)
    val nItems = sortedItems.length
    val nInBin = Math.ceil(nItems.toDouble / nBins.toDouble).toInt   // overflow will go in last bin
    val maxes = for {
      i <- 0 until nBins
      start = i*nInBin
      slice = sortedItems.slice(start, start + nInBin)
    } yield slice.last._2
    println (maxes.mkString(", "))
    maxes.toSeq
  }

  def main (args:Array[String]): Unit = {

    val dir = "/lhome/bsharp/causal/pmi/"

    // Step 1: Load in the counter from the causal pairs
    val causalFile = dir + "causalCounter_mar30_new.txt"
    val causalCounter0 = Counter.loadFrom[String](new BufferedReader(new FileReader(causalFile)))
    val causalCounter = convertCounterKeysToLowerCase(causalCounter0)

    println ("Loaded counter from " + causalFile + " with " + causalCounter.size + " keys")

    // Step 2: Load in the counter from agiga
    val agigaFile = dir + "agigaCounter_new.txt"
    val agigaCounter0 = Counter.loadFrom[String](new BufferedReader(new FileReader(agigaFile)))
    val agigaCounter = convertCounterKeysToLowerCase(agigaCounter0)
    val N = agigaCounter.getTotal
    println ("Loaded counter from " + agigaFile + " with " + agigaCounter.size + " keys")

    // Step 3: Find PMI
    // log(p(y|x)/p(y)) -- that is the log (p(word-word pair | causal) / p(word-word pair))
    val pmiCounter = new Counter[String]
    var errors = 0
    // For each c,e pair in causalCounter, check for both (c,e) and (e,c) in agigaCounter
    println(causalCounter.keySet.head)
    for (ce <- causalCounter.keySet) {
      // Get the causal count
      val causalCount = causalCounter.getCount(ce)
      // Get the general count
      val (c, e) = splitArgs(ce)
      val ec = makeString(e, c)
      val ceCount = agigaCounter.getCount(ce)
      val ecCount = agigaCounter.getCount(ec)
      val freq = ceCount + ecCount
      if (freq > 0) {
        // This handles an uncommon error
        if (freq < causalCount) {
          val ratio = 1.0
          pmiCounter.setCount(ce, Math.log(ratio * N) * Math.log(1.0 + freq))
          //pmiCounter.setCount(ce, (ratio) * Math.log(1.0 + freq))
        }
        else {
          val ratio = causalCount / (freq)
          pmiCounter.setCount(ce, Math.log(ratio * N) * Math.log(1.0 + freq))
          //pmiCounter.setCount(ce, Math.log(ratio * causalCount) * Math.log(1.0 + freq))
          //pmiCounter.setCount(ce, (ratio) * Math.log(1.0 + freq))
        }

      }
      else {
        errors += 1
        //        println ("Error with ce: " + ce)
      }
    }
    println(s"There were $errors errors found!")

    // Save:
    val pw = new PrintWriter("/lhome/bsharp/causal/pmi/pmiCounter_N.txt")
    pmiCounter.saveTo(pw)
    pw.close()

    // Examine:
    println ("PMI of (hurricane, flooding): " + pmiCounter.getCount("hurricane__flooding"))
    println ("PMI of (flooding, hurricane): " + pmiCounter.getCount("flooding__hurricane"))
    println ("PMI of (increase, flooding): " + pmiCounter.getCount("increase__flooding"))
    println ("PMI of (situation, talk): " + pmiCounter.getCount("situation__talk"))
    println ("PMI of (friction, stoppage): " + pmiCounter.getCount("friction__stoppage"))

    val values = pmiCounter.values
    val items = pmiCounter.toSeq
    val sorted = items.sortBy(- _._2)
    val valuesSum = values.sum
    val nValues = values.length
    val mean = valuesSum / nValues.toDouble
    println (s"Average pmi: $mean")
    println (s"Max pmi: ${sorted.slice(0,40)}")
    println (s"Min pmi: ${sorted.reverse.slice(0,40)}")
    println (s"Median pmi: ${sorted(values.length / 2)}")

    println ("Pmi with 3 bins:")
    findQuantileValues(pmiCounter, 3)
    println ("Pmi with 5 bins:")
    findQuantileValues(pmiCounter, 5)

    }





}
