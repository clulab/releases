package edu.arizona.sista.embeddings.postprocessing

import preprocessing.agiga.ProcessAgiga

import scala.collection.mutable.ArrayBuffer

/**
  * Created by bsharp on 3/21/16.
  */
object exampleFinder extends App {

  def prefix(fn:String):String = fn.split("").slice(0,3).mkString("")

  // Getting examples from Agiga files for paper
  val fn = "/data/nlp/corpora/agiga/data/xml/nyt_eng_199905.xml.gz"

  val doc = ProcessAgiga.agigaDocToDocument(fn)
  println (s"Document retrieved from $fn")

  val c = "fistfight"
  val e = "confrontational"

  var desiredSentences = new ArrayBuffer[String]
  val nSents = doc.sentences.length
    for (sId <- doc.sentences.indices) {
      if (sId % 1000 == 0) println (s"Processing sentence $sId of $nSents")
      val s = doc.sentences(sId)
      val txt = s.getSentenceText()
      if (txt.contains(c) && txt.contains(e)) {
        desiredSentences.append(txt)
      }
    }

  println ("\nDesired Sentences:")
  desiredSentences.foreach(println (_))




}
