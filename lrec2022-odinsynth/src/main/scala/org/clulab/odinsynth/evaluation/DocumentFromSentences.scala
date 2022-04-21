package org.clulab.odinsynth.evaluation

import org.clulab.odinsynth.using

import org.clulab.processors.Processor
import org.clulab.processors.fastnlp.FastNLPProcessor

import ai.lum.odinson.Document
import ai.lum.odinson.extra.ProcessorsUtils

import scala.io.Source

import java.io.File
import java.io.PrintWriter
import java.security.MessageDigest
import java.nio.file.Files
import scala.util.Try

object DocumentFromSentences {

  /**
    * Convert a sequence of sentences (Seq[Seq[String]]) to an odinson document
    * Abstracting this code here helps enforcing uniformity
    *
    * @param sentences - the sentences of interest
    * @param processor - the processor used to construct an odinson document
    * @return an odinson document
    */
  def documentFromSentences(sentences: Seq[Seq[String]], processor: FastNLPProcessor): Document = {
    ProcessorsUtils.convertDocument(processor.annotateFromTokens(sentences.map(_.map(_.toLowerCase()))))
  }

  def documentFromSentencesKeepCase(sentences: Seq[Seq[String]], processor: FastNLPProcessor): Document = {
    ProcessorsUtils.convertDocument(processor.annotateFromTokens(sentences))
  }
  
  /**
    * Convert a sequence of sentences (Seq[Seq[String]]) to an odinson document
    * an saves the result. If the sentences were already processed, loads the result
    *
    * @param sentences - the sentences of interest
    * @param processor - the processor used to construct an odinson document
    * @return an odinson document
    */
  def documentFromSentencesAndCache(sentences: Seq[Seq[String]], processor: FastNLPProcessor, cacheFolder: String, saveToCache: Boolean = true): Document = {
    val lengths       = sentences.map(_.length).hashCode()
    val sentencesSize = sentences.size.hashCode()
    val sha           = MessageDigest.getInstance("SHA-256")
                           .digest(sentences.map(_.mkString(" ")).mkString("\n").getBytes("UTF-8"))
                           .map("%02x".format(_)).mkString

    val hash = f"${sha}_${lengths}_${sentencesSize}"
    
    val potentialFile = new File(f"$cacheFolder/$hash.json")

    // The file might be corrputed
    lazy val trying: Try[Document] = Try(using(Source.fromFile(potentialFile)) { it => Document.fromJson(it.getLines().toSeq.mkString("\n")) })

    // NOTE start
    // Note that trying is lazy. This is to avoid building an exception stacktrace in
    // case the file does not exist. If trying is computed, the file always exist
    // This is guaranteed because of the short-circuit evaluation
    // For example, if potentialFile.exists() returns false, then there is no way
    // that (potentialFile.exists() && trying.isSuccess) is true, hence trying.isSuccess
    // is not evaluated. Since the trying variable is lazy, it is not computed
    // NOTE end
    if (potentialFile.exists() && trying.isSuccess) {
      trying.get
    } else {
      val doc = ProcessorsUtils.convertDocument(processor.annotateFromTokens(sentences.map(_.map(_.toLowerCase()))))
      if (saveToCache) {
        using(new PrintWriter(potentialFile)) { it =>
          it.print(doc.toJson)
        }
      }
      doc
    }
  }
  
}
