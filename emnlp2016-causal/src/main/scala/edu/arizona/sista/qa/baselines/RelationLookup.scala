package edu.arizona.sista.qa.baselines

import java.util.Properties

import edu.arizona.sista.qa.translation.CausalAlignment
import edu.arizona.sista.struct.Counter
import edu.arizona.sista.utils.StringUtils
import org.slf4j.LoggerFactory
import RelationLookup.logger



/**
  * Created by bsharp on 4/27/16.
  */
class RelationLookup (directory:String, extension:String, lenCutoff:Int = 0) {

  def this (props:Properties) = {
    this (props.getProperty("relation_lookup.input_directory"),
      props.getProperty("relation_lookup.input_extension"),
      StringUtils.getInt(props, "relation_lookup.len_threshold", 0))
  }

  // Currently I am implementing this as a word:word lookup, can be extended to phrase:phrase
  val (targetMap, contextMap) = loadMaps(directory, extension:String, lenCutoff)

  // Load the target and context maps from a directory
  private def loadMaps(dir:String, extension:String, lenCutoff:Int): (Map[String, Counter[String]], Map[String, Counter[String]]) = {
    val tMap = new collection.mutable.HashMap[String, Counter[String]]()
    val cMap = new collection.mutable.HashMap[String, Counter[String]]()

    // Retrieve the relevant files
    val (srcPhrases, dstPhrases) = CausalAlignment.loadData(dir, extension, lenThreshold = lenCutoff)
    assert(srcPhrases.length == dstPhrases.length)

    // generate the maps
    for (i <- srcPhrases.indices) {
      // Retrieve the phrases
      val srcTxt = srcPhrases(i)
      val dstTxt = dstPhrases(i)

      // Split on whitespace
      val srcTokens = srcTxt.split(" ")
      val dstTokens = dstTxt.split(" ")

      // Process
      for (st <- srcTokens) {
        // If this target/source word hasn't been previously seen, add to the target map
        if (!tMap.contains(st)) tMap.put(st, new Counter[String])
        for (dt <- dstTokens) {
          // If this context/destination word hasn't been previously seen, add to the context map
          if (!cMap.contains(dt)) cMap.put(dt, new Counter[String])

          // Add the words to the appropriate counters
          tMap(st).incrementCount(dt)
          cMap(dt).incrementCount(st)
        }
      }
    }

     logger.debug(s"Finished loading the target and context maps from files in $dir with extension $extension and an argument lenThreshold of $lenCutoff")
    (tMap.toMap, cMap.toMap)
  }

  // Returns the total lookup matches between two texts
  // TODO: Do I want this? It will be quite biased towards longer answers!
  def textMatches(t1:Array[String], t2:Array[String], m: Map[String, Counter[String]]): Double = {

    var totalMatches:Double = 0.0

    for (e1 <- t1) {
      val counter = m.get(e1)
      if (counter.isDefined) {
        for (e2 <- t2) {
          val matches = counter.get.getCount(e2)
          totalMatches += matches
        }
      }
    }

    totalMatches
  }

  // Returns the average matches for all pairs successfully looked up
  def averageMatches(t1:Array[String], t2:Array[String], m: Map[String, Counter[String]]): Double = {

    var totalMatches:Double = 0.0
    var lookupCounter:Double = 0.0

    for (e1 <- t1) {
      val counter = m.get(e1)
      if (counter.isDefined) {
        for (e2 <- t2) {
          val matches = counter.get.getCount(e2)
          if (matches > 0.0) {
            totalMatches += matches
            lookupCounter += 1.0
          }
        }
      }
    }

    totalMatches / lookupCounter
  }

  // Returns the highest number of matches found for any given pair in t1 and t2
  def maxMatches(t1:Array[String], t2:Array[String], m: Map[String, Counter[String]]): Double = {

    var maxMatches:Double = 0.0

    for (e1 <- t1) {
      val counter = m.get(e1)
      if (counter.isDefined) {
        for (e2 <- t2) {
          val matches = counter.get.getCount(e2)
          if (matches > maxMatches) {
            maxMatches = matches
          }
        }
      }
    }

    maxMatches
  }

}

object RelationLookup {
  val logger = LoggerFactory.getLogger(classOf[RelationLookup])
}
