package edu.arizona.sista.qa.ranking

import org.slf4j.LoggerFactory
import scala.collection.mutable.{ListBuffer, ArrayBuffer}
import edu.arizona.sista.utils.StringUtils
import scala._
//import edu.arizona.sista.qa.discourse.{DiscourseModelNGram, DiscourseModelCusp}
import scala.Predef._
import edu.arizona.sista.qa.QA
import java.io.{FileInputStream, BufferedInputStream}
import java.util.Properties

/**
 * Main entry point for running a ranker-based experiment specified using a properties file
 * User: peter
 * Date: 7/31/2013
 */
object RankerEntryPoint {
  val logger = LoggerFactory.getLogger(classOf[Ranker])
  val VERY_SMALL_SCORE_DIFF = 0.01


  def main(args:Array[String]) {
    println(s"heap size: ${Runtime.getRuntime.maxMemory / (1024 * 1024)}")
    val props = StringUtils.argsToProperties(args)
    if (props.getProperty("workunit_id") == null) props.setProperty("workunit_id", util.Random.nextLong().toString) // set it so it gets saved
    val workunitID = props.getProperty("workunit_id", "workunit_undefined")


    // Select ranker training method
    val trainMethod = props.getProperty("ranker.train_method", "voting").toLowerCase
    println(trainMethod)
    var ranker:Ranker = null
    if (trainMethod == "normal") {
      ranker = new RankerSimple(props)
    } else if (trainMethod == "voting") {
      ranker = new RankerVoting(props)
    } else if (trainMethod == "combined") {
      ranker = new RankerCombined(props)
    } else {
      throw new RuntimeException("ERROR: unknown ranker type: " + trainMethod)
    }

    // Perform training and evaluation procedure
    ranker.doRun(workunitID)

    // Explicit call to exit -- otherwise JForest hangs
    logger.info ("Exiting...")
    //exit(0)
  }

}
