package preprocessing.SemEval

import java.io.PrintWriter

import scala.collection.mutable.ArrayBuffer

/**
  * Created by bsharp on 3/8/16.
  */
object CollapseFoundPairs extends App {

  def getPotentialSingular(s:String):Option[String] = {
    if (!s.endsWith("s")) return None
    else return Some(s.split("").slice(0, s.length - 1).mkString(""))
  }

  // Returns a set of pairs which don't include the plural version of existing pairs
  def collapsePlurals(set:Set[(String, String)]):Set[(String, String)] = {
    val exclude = new ArrayBuffer[(String, String)]

    for (item <- set) {
      // Get the singular form of the Cause
      val cause = item._1
      val singCause = getPotentialSingular(cause)
      // Get the singular form of the Effect
      val effect = item._2
      val singEffect = getPotentialSingular(effect)

      // If both have a singular form and the (sing, sing) pair exists, add to the exclude list
      if (singCause.isDefined && singEffect.isDefined) {
        if (set.contains((singCause.get, singEffect.get))) exclude.append((cause, effect))
      } else if (singCause.isDefined) {
        // OR.... If only the cause has a singular form and the (sing, regular) pair exists, add to the exclude list
        if (set.contains((singCause.get, effect))) exclude.append((cause, effect))
      } else if (singEffect.isDefined) {
        // OR.... If only the effect has a singular form and the (regular, sing) pair exists, add to the exclude list
        if (set.contains((cause, singEffect.get))) exclude.append((cause, effect))
      }
    }
    // Return the NON-excluded pairs
    set.diff(exclude.toSet)
  }

  val fnIn = "/home/bsharp/causal/SEMEVAL_NON_cause_effect_pairs.txt"
  val fnOut = fnIn + ".collapsed.tsv"
  val out = new PrintWriter(fnOut)

  val lines = scala.io.Source.fromFile(fnIn, "UTF-8").getLines()
  // Example line: #reactors (Cause) ==> electricity (Effect)

  val pairs = new ArrayBuffer[(String, String)]

  for (line <- lines) {
    val parts = line.split(" ==> ")
    val cause = parts(0).split("\\(.*\\)")(0).trim.split("#").last.toLowerCase
    val effect = parts(1).split("\\(.*\\)")(0).trim.toLowerCase

    val pair = (cause, effect)
    pairs.append(pair)

  }

  // All the pairs
  val collapsedPairs1 = pairs.toSet
  val collapsedPairs2 = collapsePlurals(collapsedPairs1) // remove the plurals duplicates

  // Exclude pairs with multi-word expressions
  var counter:Int = 0
  for ((e1,e2) <- collapsedPairs2) {
    val e1Tokens = e1.split(" ")
    if (e1Tokens.length > 1) {
      println("Multiword E1: " + e1)
    }
    val e2Tokens = e2.split(" ")
    if (e2Tokens.length > 1) {
      println("Multiword E2: " + e2)
    }
    if (e1Tokens.length > 1 || e2Tokens.length > 1) {
      counter += 1
    } else {
      // If doesn't contain a multiword entity, PRINT
      out.println (e1 + "\t" + e2)
    }
  }
  println ("number excluded due to multiword e1 or e2: " + counter)

//  // DEBUG: Display what was removed
//  val diff = collapsedPairs1.diff(collapsedPairs2)
//  diff.foreach(println(_))
//  println ("========================================")

//  // DEBUG: this essentially is a hacky way of sorting and displaying to make sure above filtering code worked
//  val collapsedPairs = collapsedPairs2.toSeq.groupBy(_._1).groupBy(_._1(0))
//
//  var counter = 1
//  for (character <- collapsedPairs.keySet) {
//    val wordSet = collapsedPairs.get(character).get
//    for (k <- wordSet.keySet){
//      val vs = wordSet.get(k).get
//      for (v <- vs) {
//        println (s"Pair $counter: " + v)
//        counter += 1
//      }
//    }
//  }


  // Housekeeping
  out.close()

}
