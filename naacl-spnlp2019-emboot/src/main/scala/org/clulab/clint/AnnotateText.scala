package org.clulab.clint

import java.io.File
import scala.io.Source
import org.clulab.processors.Document
import org.clulab.processors.bionlp.BioNLPProcessor
import com.typesafe.scalalogging.LazyLogging
import com.typesafe.config.ConfigFactory
import ai.lum.common.ConfigUtils._
import ai.lum.common.FileUtils._
import ai.lum.common.IteratorUtils._
import ai.lum.common.Serializer

object AnnotateText extends App with LazyLogging {

  val config = ConfigFactory.load()
  val textDir = config[File]("clint.text-dir")
  val docsDir = config[File]("clint.docs-dir")

  val processor = new BioNLPProcessor

  // annotate all text files in text directory
  for (f <- textDir.listFilesByWildcard("*.txt").par) {
    val docFile = new File(docsDir, f.getBaseName() + ".ser")
    if (docFile.exists()) {
      logger.info(s"${docFile.getName()} already exists")
    } else {
      logger.info(s"Annotating ${f.getName()}")
      val text = f.readString()
      val doc = processor.annotate(text)
      doc.id = Some(f.getBaseName())
      Serializer.serialize(doc, docFile)
    }
  }

}
