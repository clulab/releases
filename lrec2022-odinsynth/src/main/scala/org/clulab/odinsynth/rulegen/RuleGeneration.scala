package org.clulab.odinsynth.rulegen

import scala.util.Random
import scala.collection.mutable.ArrayBuffer
import util.control.Breaks._

/* Generate rules from a list of annotatios
 *
 * @constructor create a new rule generator with a sequence of annotations
 * @param annotations a list of annotations with the part of the sentece that will be used as source
 * @param length number of tokens in the span
 */
class RuleGenerator (annotations: Map[String, Seq[String]], length: Int) {
  /** Returns the rule generated */
  def run: String = {
    // generate rule
    val rule = generateRule()
    // return rule
    rule
  }
   
  /** Add quotes to a string to escape special characters */
  def maybeQuote(string: String): String = { 
    val allowedChar = "(^[A-Za-z]*$)".r
    string match {
      case allowedChar(s) => s
      case default => '"' + default + '"'
    }
  }
  
  /** Add [<terminal> = something] */
  private def addTerminal(v: Map[String, Seq[String]], brackets: Boolean = true) = {
    // when adding a terminal, v must have only one 
    assert(v.size == 1)
    // get type
    val key = v.keys.head
    // get value
    val value = v(key)
    // make sure val is as expected
    assert(value.size == 1)
    // add final string
    if(brackets)
      s"[${key}=${maybeQuote(value.head)}]"
    else 
      s"${key}=${maybeQuote(value.head)}"
  }
  
  /** Add  [<terminal1> = something ('|' or '&') <terminal1> something] */
  private def addLogical(v: Map[String, Seq[String]], tokenId: Int, logicalValue: String): String = {
    val annotations_cp = annotations.toMap
    val rulePart = new StringBuilder()
    // controll what to randomly select
    var possibilities = ArrayBuffer("word", "tag", "entity")
    // if the entity annotation is not available, do not include it
    if (annotations("entity") == "O") possibilities -= "entity"
    //
    rulePart ++= "["
    // generate one for tag
    if (annotations("entity")(tokenId) == "O") {
      rulePart ++= addTerminal( Map("word" -> Seq(annotations("word")(tokenId) )), false)
    } else {
      rulePart ++= addTerminal( Map("entity" -> Seq(annotations("entity")(tokenId) )), false)
    }
    // add or
    rulePart ++= s" $logicalValue "
    // generate one for word
    rulePart ++= addTerminal( Map("tag" -> Seq(annotations("tag")(tokenId) )), false) 
    // close
    rulePart ++= "]"
    // return the rules
    rulePart.toString
  }
  
  /** Return a list of repetitions if any is found in the annotation list
   *
   *  for example:
   *  ["RB", "NN", "NN"] will return Seq(Pair(1, 2))
   *
   *  @param annotations the list of annotations extracted from the span used as source to generate the rule
   *  @return sequence of pair of spans with the repetitions
   */
  def getRepetitions(annotations: Seq[String]): Seq[Pair[Int, Int]] = {
    val repetitions: ArrayBuffer[Pair[Int, Int]] = new ArrayBuffer
    var start: Int = -1
    for(i <- 0 until annotations.length){
      // deal with the last element
      val next = if(i < annotations.length-1) annotations(i+1) else ""
      // if the current element is equal to the next
      // and start variable is not set
      if(annotations(i).take(2) == next.take(2) & start == -1) {
        start = i
      // if the elements are differt
      } else if(annotations(i).take(2) == next.take(2)) {
      } else {
        if(start != -1) {
            repetitions += Pair(start, i)
            start = -1
        }
      }
    }
    // return the sequence
    repetitions.toSeq
  }
  
  /** returns a rule generated based on the annotations received by this class */
  def generateRule(): String = {
    // generate a range to foreach tokens
    val tokenRange = 0 to length-1
    
    // detece repetitions to add */+
    val repetitions = getRepetitions(annotations("tag"))
    val selectedRepetition = if(repetitions.nonEmpty) repetitions.head else Pair(-1, -1)
    //
    // expected output List[Map[String,Seq[Pair[Int, Int]]]]
    // print all words using the token range
    val rule = tokenRange.map(tid => {
      // check if the current tid is part of a repetition
      if(selectedRepetition._1 to selectedRepetition._2 contains tid) {
        // print the repetition only once
        if(tid == selectedRepetition._1){
          // TODO: this should add only the first 2 elements of the string for part of spech (tag)
          Random.nextInt(100) match {
            case v if 0 until 30 contains v =>
              addTerminal( Map("tag" -> Seq(annotations("tag")(tid) )) ) + "*"
            case v if 30 until 100 contains v =>
              addTerminal( Map("tag" -> Seq(annotations("tag")(tid) )) ) + "+"
          }
        } else {
          // after the first element, just print empty
          ""
        }
      } else {
        // otherwise do the regular stuff
        Random.nextInt(100) match {
          // TODO: add entities
          case v if 0 until 25 contains v =>
            addLogical(annotations, tid, "&") 
          case v if 25 until 35 contains v =>
            addLogical(annotations, tid, "|") 
          case v if 35 until 50 contains v =>
            addTerminal( Map("word" -> Seq(annotations("word")(tid) )) ) 
          case v if 50 until 100 contains v =>
            // entities and tags are together in this thing
            Random.nextInt(50) match {
              case v if 0 until 40 contains v =>
                if( annotations("entity")(tid) != "O"){
                  addTerminal( Map("entity" -> Seq(annotations("entity")(tid) )) ) 
                } else {
                  addTerminal( Map("tag" -> Seq(annotations("tag")(tid) )) ) 
                }
              case v if 40 until 50 contains v =>
                  addTerminal( Map("tag" -> Seq(annotations("tag")(tid) )) ) 
            }
        }
      }
    }).mkString(" ")
    //
    rule
  }
}
