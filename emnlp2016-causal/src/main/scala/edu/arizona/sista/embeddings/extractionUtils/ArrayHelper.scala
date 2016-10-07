package extractionUtils

import scala.collection.mutable.ArrayBuffer
import scala.util.control.Breaks._

/**
 * Created by peter on 2/4/16.
 */

object ArrayHelper {

  // Removes the duplicates from an array of integers
  def removeDuplicates(in:Array[Int]):Array[Int] = {
    val out = new ArrayBuffer[Int]()
    for (i <- 0 until in.size) {
      var found:Boolean = false
      for (j <- 0 until out.size) {
        if (in(i) == out(j)) found = true
      }

      if (!found) out.append( in(i) )
    }
    out.toArray
  }

  def findIntersection(in1:Array[Int], in2:Array[Int]):Array[Int] = {
    val out = new ArrayBuffer[Int]
    for (a <- in1) {
      for (b <- in2) {
        if (a == b) {
          out.append( a )
        }
      }
    }
    out.toArray
  }

  def findFirstOccurrenceInt(in:Array[Int], searchFor:Int):Int = {
    for (i <- 0 until in.size) {
      if (in(i) == searchFor) return i
    }
    return -1
  }

  def removeDuplicates(in:Array[String]):Array[String] = {
    val out = new ArrayBuffer[String]()
    for (i <- 0 until in.size) {
      var found:Boolean = false
      for (j <- 0 until out.size) {
        if (in(i) == out(j)) found = true
      }

      if (!found) out.append( in(i) )
    }
    out.toArray
  }

  def removeOccurrences(in:Array[Int], removeThese:Array[Int]):Array[Int] = {
    val out = new ArrayBuffer[Int]()
    for (i <- 0 until in.size) {
      breakable {
        for (j <- 0 until removeThese.size) {
          if (in(i) == removeThese(j)) break()
        }
        // Not found
        out.append( in(i) )
      }
    }
    out.toArray
  }


}
