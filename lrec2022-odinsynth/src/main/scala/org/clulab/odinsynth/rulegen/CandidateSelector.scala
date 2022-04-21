package org.clulab.odinsynth.rulegen

import scala.util.Random
import scala.collection.mutable.ArrayBuffer
import util.control.Breaks._

/** Receives annotations and returns a list of span identifying the candidates
 *
 * @constructor create a new candidate selector with a list of annotations
 * @param annotations a list of annotations from the sentece
 */
class CandidateSelector(annotations: Map[String, Seq[String]])   {
  /** Returns a list of candidates based on chunks
   *
   * @param chunks chunk annotations
   * @return sequence of spans with candidates
   */
  def getCandidates(chunks: Seq[String]): Seq[(Int, Int)] = {
    val candidates: ArrayBuffer[Pair[Int, Int]] = new ArrayBuffer
    // generate candidates for chunks
    var start = 0
    for((c, i) <- chunks.zipWithIndex) {
      val info = c.split("-").toList
      breakable {
        // processing an actiual O
        if(info.length == 2){
          if(info(0) == "B"){
            start=i
          }
          if(info(0) == "I"){
            candidates += Pair(start, i)
          }
        }
      }
    }
    candidates
  }.groupBy( _._1).map(l => (l._1, l._2.map(_._2).max) ).toSeq
  
  /** Returns the span of a single candidate acording to some strategu
   *
   * @param candidates sequence of spans of candidates 
   * @param strategy how the candidate is going to be selected
   * @return a range from the beginning of the span of the selected candidate to the end of it
   */
  def getTokenIds(candidates: Seq[(Int, Int)], strategy: String = "larg"): Range = {
    val selected = strategy match {
      case "larg" => candidates.map(pp => pp._1 - pp._2)
                      .zip(candidates)
                      .sortWith((s1, s2) => s1._1 < s2._1)
                      .head._2
      case _ => Random.shuffle(candidates).head
    }
    // select the annotations only for selected._1 & selected._2
    selected._1 to selected._2
  }
  
  /** Returns only the annotations as a [[scala.collection.Map]]
   *
   * @param tokenIds range from the beginning of the span of the selected candidate to the end of it
   * @return a filtered annotations list, with only what was requested
   */
  def getSelectedAnnotation(tokenIds: Range): Map[String, Seq[String]]= {
    // slect only annotations used to generate the rule
    annotations
      // remove stuff that is not tokenSeq
      .filter(ann => ann._2.size > 0)
      // select only what is needed
      .map(ann => ann._1 -> tokenIds.map(id => ann._2(id) ).toSeq)
    // print the options
    // from this line up -> separate class //
  }
} 
