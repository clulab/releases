package org.clulab.clint

import java.io._
import java.util.zip._
import scala.collection.parallel.ForkJoinTaskSupport
import scala.concurrent.forkjoin.ForkJoinPool
import com.typesafe.config._
import com.typesafe.scalalogging.LazyLogging
import org.clulab.processors.bionlp.BioNLPProcessor
import org.clulab.embeddings.word2vec.Word2Vec.sanitizeWord
import ai.lum.nxmlreader.NxmlReader
import ai.lum.common.ConfigUtils._
import ai.lum.common.IteratorUtils._
import ai.lum.common.FileUtils._

object PrepareW2V extends App with LazyLogging {

  val config = ConfigFactory.load()
  val openAccessDir = config[File]("clint.open-access-dir")
  val w2vDir = config[File]("clint.w2v-dir")
  val sectionsToIgnore = config[List[String]]("clint.sections-to-ignore")
  val ignoreFloats = config[Boolean]("clint.ignore-floats")

  val openAccessDirName = openAccessDir.getCanonicalPath()

  val nxmlreader = new NxmlReader(sectionsToIgnore.toSet, ignoreFloats)
  val processor = new BioNLPProcessor

  val errorLog = new File(w2vDir, "ERRORS.txt")

  logger.info("reading nxml files")
  val nxmlFiles = openAccessDir.listFilesByWildcard("*.nxml", recursive = true).par
  nxmlFiles.tasksupport = new ForkJoinTaskSupport(new ForkJoinPool(24))
  for (f <- nxmlFiles) {
    val targetFileName = f.getCanonicalPath().drop(openAccessDirName.length).dropRight(5)
    val targetFile = new File(w2vDir, targetFileName + ".txt.gz")
    if (!targetFile.exists()) {
      targetFile.touch()
      logger.info(s"reading ${f.getName()}")
      try {
        val text = processFile(f)
        writeCompressed(targetFile, text)
      } catch {
        case e: Exception =>
          logger.error(s"failed reading ${f.getCanonicalPath()}")
          error(e, f.getCanonicalPath())
      }
    }
  }

  logger.info("concatenating")
  val concatFile = new File(w2vDir, "open-access.txt.gz")
  val concatWriter = new OutputStreamWriter(new GZIPOutputStream(new FileOutputStream(concatFile)), "UTF-8")
  for (f <- w2vDir.listFilesByWildcard("*.txt.gz", recursive = true) if f.getName() != "open-access.txt.gz") {
    if (f.size == 0) {
      logger.info(s"${f.getName()} is empty. Skipping.")
    } else {
      logger.info(s"reading ${f.getName()}")
      val text = readCompressed(f)
      concatWriter.write(text)
    }
  }
  concatWriter.close()


  def processFile(f: File): String = {
    val text = nxmlreader.read(f).text
    val doc = processor.mkDocument(text)
    doc.sentences.map(_.words.map(sanitizeWord(_)).mkString(" ")).mkString("\n")
  }

  def writeCompressed(file: File, text: String): Unit = {
    val writer = new OutputStreamWriter(new GZIPOutputStream(new FileOutputStream(file)), "UTF-8")
    writer.write(text)
    writer.close()
  }

  def readCompressed(file: File): String = {
    val br = new BufferedReader(new InputStreamReader(new GZIPInputStream(new FileInputStream(file)), "UTF-8"))
    var line: String = br.readLine()
    var text: String = ""
    while (line != null) {
      text += s"$line\n"
      line = br.readLine()
    }
    br.close()
    text
  }

  def error(e: Exception, filename: String): Unit = synchronized {
    errorLog.writeString(s"file: $filename\nerror: ${e.getMessage()}\n\n", append = true)
  }

}
