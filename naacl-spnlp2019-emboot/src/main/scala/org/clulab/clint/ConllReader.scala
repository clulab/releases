package org.clulab.clint

import scala.collection.mutable.ArrayBuffer
import org.clulab.processors.{ Document, Sentence }
import scala.io.Source
import com.typesafe.scalalogging.LazyLogging
import com.typesafe.config.ConfigFactory
import ai.lum.common.ConfigUtils._
import java.io.File
import ai.lum.common.Serializer
import org.clulab.odin.ExtractorEngine
import scala.collection.mutable.HashMap
import scala.collection.mutable.HashSet
import java.io.ObjectOutputStream
import java.io.FileOutputStream
import java.io.BufferedWriter
import java.io.FileWriter

object ConllReader extends App with LazyLogging {

  val config = ConfigFactory.load()
  val docsDir = config[File]("clint.docs-dir")
  
  val conllDataFile = config[String]("clint.conllformatdataFile") // "/Users/ajaynagesh/Research/data/CoNLL/2003/eng.testa"
  val data = Source.fromFile("/Users/ajaynagesh/Research/data/CoNLL/2003/eng.testa").getLines.mkString("\n")
//  println("=========================================================================")
//  println(data)
//  println("=========================================================================")
  val documents = ConllReader.readAnnotations(data)
  
//  val goldLabels = HashMap.empty[String, HashSet[String]]
  val goldLabels = new HashMap[String,List[String]].withDefaultValue(Nil)
  
  val finder = new OracleCandidateFinder
  for(d <- documents) {
    val candidates = finder.findCandidates(d)
    for(c <- candidates){
      c.label match {
        case "PER" => goldLabels("PER") ::= c.text
        
        case "ORG" => goldLabels("ORG") ::= c.text
        
        case "LOC" => goldLabels("LOC") ::= c.text  
        
        case "MISC" => goldLabels("MISC") ::= c.text
        
        case _ => 
      
      }
//      println(c.text + "--" + c.label)
    }
  }
  
  val filename = "/Users/ajaynagesh/Research/code/research/clint/data/conll.dev.goldlabels"
  val fw = new BufferedWriter(new FileWriter(new File(filename)))
  
  for(lbl <- goldLabels.keys){
    val entities = goldLabels(lbl)
    println(lbl)
//    println("-------------")
    fw.write(s"${lbl}\t${entities.mkString("\t")}\n")
  }
  fw.close()
  
  println(s"File written to $filename")
  
  for((doc,idx) <- documents.zipWithIndex){
    val docFile = new File(docsDir, "conll03doc-"+ idx + ".ser")
    if (docFile.exists()) {
      logger.info(s"${docFile.getName()} already exists")
    } else {
      logger.info(s"Creating conll03doc-${idx}.ser")
      doc.id = Some(s"doc-${idx}")
      Serializer.serialize(doc, docFile)
    }
  }
  
//    for((doc,idx) <- documents.zipWithIndex) {
//      val docFile = new FileWriter(new File(docsDir + "conll03doc-"+ idx + ".txt") )
//      logger.info(s"Creating conll03doc-${idx}.txt")
//      docFile.write( doc.sentences.map(s => s.getSentenceText() ).mkString("\n") )
//      docFile.close()
//    }
  
//  println("Finished reading all the documents")
//  
//  		val rules = """
//    				|- name: OracleEntityCandidates
//    				|  label: Candidate
//    				|  type: token
//    				|  priority: 1
//    				|  pattern: |
//    				|    [entity=/^B-/]? [entity=/^I-/]+
//    				|""".stripMargin
//
//
//    		val extractor = ExtractorEngine(rules)
//  
//  println("Initialized the extractor engine")  		
//    		
//  documents.foreach { d =>
//    println("Start of the extractor function")
//    val mentions = extractor.extractFrom(d)
//    println("End of the extractor function");
////    for (s <- d.sentences) {
////      for ((tag,idx) <- s.entities.get.zipWithIndex) {
////        if(tag.startsWith("B-") || tag.startsWith("I-")){
////          
////        }
////      }
////    }
//    println("---------------------------------------<START>-----------------------------------------------")
//    println(d.sentences.map { s => s.words.mkString(" ") }.mkString("\n"))
//    println("---------------------------------------<END>-----------------------------------------------")
//
//    d.sentences.foreach { s =>  
//      println(s.entities.get.mkString(" "))  
//    }
//    
//    println("====================================================================================")
//    for(m <- mentions){
//      println(m.words.mkString(" "))
//    }
//    println("====================================================================================")
//
//  }    
  logger.info(s"Finished reading the conll document collection; Total number of documents :  + ${documents.length}")
  
  
  /** gets a string containing CoNLL 2002-2003 annotations
   *  and returns an array of processor's documents
   */
  def readAnnotations(data: String): Array[Document] = {
    // Data consists of many documents concatenated.
    // Each doc begins with the string "-DOCSTART- -X- -X- O"
    data.split("-DOCSTART- -X- -X- O")
      .tail // first string is empty
      .map(_.trim) // remove surrounding whitespace
      .map(mkDocument) // make a processor's document
  }

  def mkDocument(data: String): Document = {
    // sentences are separated by an empty line
    val sentences = data.split("\n\n").map(mkSentence)
    new Document(sentences)
  }

  def mkSentence(data: String): Sentence = {
    val wordsBuffer = new ArrayBuffer[String]
    val tagsBuffer = new ArrayBuffer[String]
    val chunksBuffer = new ArrayBuffer[String]
    val entitiesBuffer = new ArrayBuffer[String]
    // add strings to buffers
    for (line <- data.lines) {
      val Array(word, tag, chunk, entity) = line.split(" ")
      wordsBuffer += word
      tagsBuffer += tag
      chunksBuffer += chunk
      entitiesBuffer += entity
    }
    // make arrays from buffers
    val words = wordsBuffer.toArray
    val tags = tagsBuffer.toArray
    val chunks = chunksBuffer.toArray
    val entities = entitiesBuffer.toArray
    // make offsets
    val (startOffsets, endOffsets) = mkOffsets(words)
    // make sentence
    val sentence = new Sentence(words, startOffsets, endOffsets)
    // add annotations
    sentence.tags = Some(tags)
    sentence.chunks = Some(chunks)
    sentence.entities = Some(entities)
    // return resulting sentence
    sentence
  }

  /** generates character offsets for the given array of words */
  def mkOffsets(words: Array[String]): (Array[Int], Array[Int]) = {
    var start: Int = 0
    var end: Int = words.head.size
    val startOffsets = new ArrayBuffer[Int]
    val endOffsets = new ArrayBuffer[Int]
    // add first word
    startOffsets += start
    endOffsets += end
    // iterate over all words except first one
    for (w <- words.tail) {
      start = end + 1 // plus a space
      end = start + w.size
      startOffsets += start
      endOffsets += end
    }
    (startOffsets.toArray, endOffsets.toArray)
  }
  
}
