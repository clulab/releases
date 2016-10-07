package edu.arizona.sista.qa.scorer

import collection.mutable.ArrayBuffer
import java.io.PrintWriter
import edu.arizona.sista.struct.Counter
import Histogram.constWidthString

/**
 * Class to compute and display a histogram
 * User: peter
 * Date: 7/31/13
 */
class Histogram(var name:String) {
  val histBins = Array[Double](-1.01, -0.90, -0.50, -0.25, -0.10, -0.01, 0, 0.01, 0.10, 0.25, 0.50, 0.90, 1.01)
  val histData = new Array[Int](histBins.size)                          // Counts for each bin
  val histNums = new Array[ArrayBuffer[String]](histBins.size)          // Tags for each bin (ie. the question numbers in that bin)

  // Constructor -- Initialize data structures
  for (i <- 0 until histBins.size) {
    histData(i) = 0
    histNums(i) = new ArrayBuffer[String]
  }

  /*
   * Add a datapoint to the historgram
   * score: Data to add (-1 to 1)
   * tag: An optional string associated with the datapoint (e.g. "Question #10") that will be displayed with the histrogram
   */
  def addData(score:Double, tag:String) {
    // Histogram computation and binning
    for (j <- 0 until ((histBins.size) - 1)) {
      if ((score >= histBins(j)) && (score < histBins(j + 1))) {
        histData(j) += 1
        histNums(j).append(tag)
      }
    }
  }

  /*
   * Display the histogram
   */
  def display(pw:PrintWriter) {
    pw.println( name + " Histogram")
    for (j <- 0 until ((histBins.size) - 1)) {
      var binText = (histBins(j) * 100).formatted("%3.1f") + "% to " + (histBins(j + 1) * 100).formatted("%3.1f") + "%"
      binText = constWidthString(binText, 20)     // constant width formatting

      pw.print(binText + " : " + histData(j) + "   [" + histData(histData.size - 2 - j) + "]  ")
      pw.print("Qs{")
      val histQs = histNums(j)
      for (k <- 0 until histQs.size) {
        pw.print(histQs(k) + ", ")
      }
      pw.println("}")
    }
    pw.println("")
  }

}


// Histogram class for histograms whose bins are discrete strings rather than numeric values
class HistogramString(var name:String) {
  val data = new Counter[String]
  var datums:Double = 0.0

  def addData(bin:String) {
    data.incrementCount(bin)
    datums += 1
  }

  def display(pw:PrintWriter) {
    pw.println (name + " Histogram")
    println (name + " Histogram")
    val keys = data.keySet.toArray
    val sortedKeys = keys.sorted

    for (key <- sortedKeys) {
      var count = data.getCount(key)
      pw.println (constWidthString(key, 20) + "\t" + count + " (" + (count / datums).formatted("%3.3f") + ")" )
      println (constWidthString(key, 20) + "\t" + count + " (" + (count / datums).formatted("%3.3f") + ")" )
    }

    pw.println ("")
    println ("")
  }

}

object Histogram {

  def constWidthString(text:String, width:Int):String = {
    var out = text
    for (a <- 0 until (width - out.length)) out += " " // Constant width formatting
    out
  }
}