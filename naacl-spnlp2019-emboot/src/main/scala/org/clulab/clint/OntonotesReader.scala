package org.clulab.clint

import scala.collection.mutable.ArrayBuffer
import org.clulab.processors.{Document, Sentence}

import scala.io.Source
import com.typesafe.scalalogging.LazyLogging
import com.typesafe.config.ConfigFactory
import ai.lum.common.ConfigUtils._
import java.io.File

import ai.lum.common.Serializer

import scala.collection.mutable.HashMap
import java.io.BufferedWriter
import java.io.FileWriter

import org.clulab.clint.ConllReader.logger

/**
 * Original data in ontonotes-5.0
 * Expects the input files to be in CoNLL format. 
 * The conversion from Brat format to CoNLL format is carried
 * out by the utility in the brat tool called 'tools/anntoconll.py'
 * (https://github.com/nlplab/brat/issues/996)
 */
object OntonotesReader extends App with LazyLogging {
  
  val config = ConfigFactory.load()
  val docsDir = config[File]("clint.docs-dir") // "/Users/ajaynagesh/Research/code/research/clint/data/docs_ScienceIE/" //
  val goldLabelsFile = config[String]("clint.goldLabelsFile") // "/Users/ajaynagesh/Research/code/research/clint/data/ScienceIE.goldlabels"
  //val conllFormatDataFilesDir =  config[String]("clint.conllFormatDataFilesDir") // "/Users/ajaynagesh/Research/data/ScienceIE/training_data/train2.conll"
  val dataFile:String = config[String]("clint.conllFormatDataFile") // "/Users/ajaynagesh/Research/data/CoNLL/2003/eng.testa"

  val data = Source.fromFile(dataFile).getLines.mkString("\n")
  val documents = OntonotesReader.readAnnotations(data)

//  val conllFormatDataFiles = new java.io.File(conllFormatDataFilesDir).listFiles
//  val documents = for(file <- conllFormatDataFiles) yield {
//    val document = OntonotesReader.readAnnotations(file)
//    document
//  }
  
  val goldLabels = new HashMap[String,List[String]].withDefaultValue(Nil)
  
  val finder = new OracleCandidateFinderOntonotes
  for(d <- documents) {
    val candidates = finder.findCandidates(d)
    for(c <- candidates){
      c.label match {
        case "PERSON" => goldLabels("PERSON") ::= c.text
        case "NORP" => goldLabels("NORP") ::= c.text
        case "FAC" => goldLabels("FAC") ::= c.text
        case "ORG" => goldLabels("ORG") ::= c.text
        case "GPE" => goldLabels("GPE") ::= c.text
        case "LOC" => goldLabels("LOC") ::= c.text
        case "PRODUCT" => goldLabels("PRODUCT") ::= c.text
        case "EVENT" => goldLabels("EVENT") ::= c.text
        case "WORK_OF_ART" => goldLabels("WORK_OF_ART") ::= c.text
        case "LAW" => goldLabels("LAW") ::= c.text
        case "LANGUAGE" => goldLabels("LANGUAGE") ::= c.text
        case "DATE" => goldLabels("DATE") ::= c.text
        case "TIME" => goldLabels("TIME") ::= c.text
        case "PERCENT" => goldLabels("PERCENT") ::= c.text
        case "MONEY" => goldLabels("MONEY") ::= c.text
        case "QUANTITY" => goldLabels("QUANTITY") ::= c.text
        case "ORDINAL" => goldLabels("ORDINAL") ::= c.text
        case "CARDINAL" => goldLabels("CARDINAL") ::= c.text        
        case _ => 
      
      }
//      println(c.text + "--" + c.label)
    }
  }
  
  val fw = new BufferedWriter(new FileWriter(new File(goldLabelsFile)))
  
  for(lbl <- goldLabels.keys){
    val entities = goldLabels(lbl)
    println(lbl)
//    println("-------------")
    fw.write(s"${lbl}\t${entities.mkString("\t")}\n")
  }
  fw.close()
  
  println(s"Gold Labels File written to ${goldLabelsFile}")


  for((doc,idx) <- documents.zipWithIndex){
    val docFile = new File(docsDir, "ontonotes-5.0-train-"+ idx + ".ser")
    if (docFile.exists()) {
      logger.info(s"${docFile.getName()} already exists")
    } else {
      logger.info(s"Creating ontonotes-5.0-train-${idx}.ser")
      doc.id = Some(s"ontonotes-5.0-train-${idx}")
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
    val doc = Document(sentences)
    doc
  }

  def mkSentence(data: String): Sentence = {
    val wordsBuffer = new ArrayBuffer[String]
    val entitiesBuffer = new ArrayBuffer[String]
    val tagsBuffer = new ArrayBuffer[String]
    // add strings to buffers
    for (line <- data.lines) {
     // O       175     187     Subsequently
     // O       187     188     ,
     // B-Material      189     195     wafers
     // O       196     200     were
     // O       201     208     exposed
     // O       209     211     to
     // O       212     214     an
     // B-Process       215     217     Al
     // I-Process       218     222     flux
     // ......
      
      val Array(word, postag, entity) = line.split("\t")
      wordsBuffer += word
      entitiesBuffer += entity
      tagsBuffer += postag

    }
    // make arrays from buffers
    val words = wordsBuffer.toArray
    val entities = entitiesBuffer.toArray
    val postags = tagsBuffer.toArray
    // make offsets
    val (startOffsets, endOffsets) = mkOffsets(words)
    // make sentence
    val sentence = new Sentence(words, startOffsets, endOffsets)
    // add annotations
    sentence.entities = Some(entities)
    sentence.tags = Some(postags)
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
