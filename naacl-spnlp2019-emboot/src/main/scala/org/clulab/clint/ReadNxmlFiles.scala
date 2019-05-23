package org.clulab.clint

import java.io._
import com.typesafe.scalalogging.LazyLogging
import com.typesafe.config.ConfigFactory
import ai.lum.common.ConfigUtils._
import ai.lum.nxmlreader.NxmlReader

object ReadNxmlFiles extends App with LazyLogging {

  val config = ConfigFactory.load()
  val nxmlDir = config[File]("clint.nxml-dir")
  val textDir = config[File]("clint.text-dir")
  val sectionsToIgnore = config[List[String]]("clint.sections-to-ignore")
  val ignoreFloats = config[Boolean]("clint.ignore-floats")

  val nxmlreader = new NxmlReader(sectionsToIgnore.toSet, ignoreFloats)

  // extract text from nxml files
  for (f <- nxmlDir.listFiles() if f.getName().endsWith(".nxml")) {
    logger.info(s"Reading ${f.getName()}")
    val targetFile = new File(textDir, f.getName().dropRight(5) + ".txt")
    val text = nxmlreader.read(f).text
    writeTextFile(targetFile, text)
  }

  // helper function to write text into a file
  def writeTextFile(file: File, text: String): Unit = {
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write(text)
    bw.close()
  }

}
