//package org.clulab.clint
//
//import org.scalatest._
//import scala.math.{ exp, log }
//import scala.collection.mutable.{ HashMap, HashSet, ArrayBuffer }
//import BootstrapGupta.{ softPMIpattern, Index, Counts, Counts2 }
//
//class TestSoftPmi extends FlatSpec with Matchers {
//
//  val promotedEntities = HashMap(
//    "category 1" -> HashSet(1, 2, 3, 4),
//    "category 2" -> HashSet(5, 6, 7, 8),
//    "category 3" -> HashSet(9)
//  )
//
//  val patternToEntities = new Index(Map(
//    1 -> Set(1, 3, 4, 5),
//    2 -> Set(6, 7, 8),
//    3 -> Set(9)
//  ))
//
//  val entityPatternCount = new Counts2(Map(
//    (1, 1) -> 5,
//    (3, 1) -> 2,
//    (4, 1) -> 8,
//    (5, 1) -> 3,
//    (6, 2) -> 6,
//    (7, 2) -> 3,
//    (8, 2) -> 1,
//    (9, 3) -> 1159
//  ))
//
//  val entityCounts = new Counts(Map(
//    1 -> 15,
//    2 -> 20,
//    3 -> 11,
//    4 -> 10,
//    5 -> 33,
//    6 -> 75,
//    7 -> 92,
//    8 -> 33,
//    9 -> 1311
//  ))
//
//  val patternCounts = new Counts(Map(
//    1 -> 18,
//    2 -> 10,
//    3 -> 1938
//  ))
//
//  val epsilon = 1e-10
//
//  "softPMIpattern" should "return the correct value" in {
//
//    // we are using the first row in the table at
//    // https://en.wikipedia.org/wiki/Pointwise_mutual_information#Applications
//
//    val p = 3
//    val cat = "category 3"
//    val expandedEntities: HashSet[(Int, Double)] = HashSet.empty
//
//    val score = softPMIpattern(p, cat, expandedEntities, promotedEntities,
//      patternToEntities, entityPatternCount, entityCounts, patternCounts)
//
//    // we need to convert our score to a real PMI
//    val pmi = log(exp(score) * 50000952)
//    // check we got it right withing some range
//    pmi should equal (10.0349081703 +- epsilon)
//  }
//
//}
