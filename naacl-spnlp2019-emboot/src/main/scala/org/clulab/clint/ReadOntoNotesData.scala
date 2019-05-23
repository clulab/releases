package org.clulab.clint

import java.io._
import com.typesafe.scalalogging.LazyLogging
import com.typesafe.config.ConfigFactory
import ai.lum.common.ConfigUtils._
import scala.xml.XML
import scala.io._

/**
  * Input : Takes the directory containing conll formatted ontonotes data (converted to a simple conll 03 format (NOTE: need to verify how this conversion is done, but the conversion is faithful to the original dataset upon spot checking in multiple files))
  * Output: A single file which contains the entire contents of the files in the directory specified as input, separated by --DOCSTART--, and is in the simple conll'03 format which can be used to create datasets for our algorithms
  */

object ReadOntoNotesData extends App with LazyLogging {
  val config = ConfigFactory.load()
  val ontonotesInDir = config[File]("clint.ontonotes_inDir")
  val ontonotesOutFile = config[String]("clint.ontonotes_outFile")

  def getListOfFiles(dir: File): List[File] = dir.listFiles.filter(_.isFile).toList

  val toFilter = Array("DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL")

  val bw = new FileWriter(new File(ontonotesOutFile))

  val fileList = getListOfFiles(ontonotesInDir)
  for (file <- fileList) {
    logger.info(s"Processing file ${file.getName} ... ")
    bw.write("-DOCSTART- -X- -X- O\n\n")
    val lines = Source.fromFile(file).getLines().toList
    var inNEspan = false
    var currentNElabel: String = ""
    val newlines = for (line <- lines) yield {
      if (line.isEmpty) {
        ""
      }
      else {
        val fields = line.split("\t")
        val nelabel = fields(2)
        val newfield = if (nelabel.endsWith(")") && nelabel.startsWith("(")) {
          val nelabelname = nelabel.replace("(", "").replace(")", "").replace("*", "")
          val newnelabelname = if (!toFilter.contains(nelabelname))
            "B-" + nelabelname
          else
            "O"
          newnelabelname
        }
        else if (nelabel.startsWith("(")) {
          val nelabelname = nelabel.replace("(", "").replace(")", "").replace("*", "")
          val newnelabelname = if (!toFilter.contains(nelabelname)) {
            inNEspan = true
            currentNElabel = nelabelname
            "B-" + nelabelname
          }
          else {
            inNEspan = false
            "0"
          }
          newnelabelname
        }
        else if (inNEspan && nelabel.equals("*")) {
          "I-" + currentNElabel
        }
        else if (inNEspan && nelabel.equals("*)")) {
          inNEspan = false
          "I-" + currentNElabel
        }
        else {
          "O"
        }

//        println(fields(0) + "\t" + fields(1) + "\t" + newfield)
        fields(0) + "\t" + fields(1) + "\t" + newfield
      }
    }
    bw.write(newlines.mkString("\n") + "\n\n")

  }
  bw.close
}
//    val filename = "/Users/ajaynagesh/Research/data/ontonotes-release-5.0/conll-formatted-ontonotes-5.0/conll-formatted-ontonotes-5.0/data/train_simple.v1/bn_cnn_03_cnn_0327.gold_conll.simple"
//    val lines = Source.fromFile(filename).getLines().toList

//  val dirloc = config[String]("clint.dirloc") //"/Users/ajaynagesh/Research/data/ontonotes-release-5.0/data/files/data/english/annotations/"
//    val files = Source.fromFile(xmlFileList).getLines().toArray
//    println(files)
//    for(file <- files) {
//      val xmldocument = XML.loadFile(file)
//
//      val fileBaseName = file.replace(dirloc, "")
//                                       .replace("/","_").replace(".name", ".txt").tail
//
//      logger.info(s"Processing : ${file}")
//      logger.info(s"File basename : ${fileBaseName}")
//      //logger.info(xmldocument.toString())
//      //logger.info("-------")
//      //logger.info(xmldocument.text)
//
//      val documentText = xmldocument.text.tail
//      val namesAndTypesInDoc = xmldocument \ "ENAMEX"
//
//      // Create the ann document
//      val fileBaseAnnName = fileBaseName.replace(".txt", ".ann")
//      logger.info(s"Creating ann document : ${fileBaseAnnName}")
//      val outAnnFile = s"${txtAndAnnOutDir}/${fileBaseAnnName}"
//      val bwAnn = new FileWriter(new File(outAnnFile))
//
//      var id = 0
//      var prevIdx = 0
//      for(nameType <- namesAndTypesInDoc) {
//        val entityName = nameType.text
//        val entityType = nameType \ "@TYPE"
//
//        if (! toFilter.contains(entityType.toString())) {
//
//          id +=1 ;
//          val startIdx = documentText.indexOf(entityName, prevIdx)
//          val endIdx = startIdx + entityName.length
//
//          println(s"T${id}\t${entityType}\t${startIdx}\t${endIdx}\t${entityName}\t--\t${documentText.substring(startIdx, endIdx)}")
//          bwAnn.write(s"T${id}\t${entityType} ${startIdx} ${endIdx}\t${entityName}")
//          bwAnn.write("\n")
//          prevIdx = endIdx
//        }
//      }
//
//      // close ann file
//      bwAnn.close
//
//      // Create the text document
//      logger.info(s"Creating text document : ${fileBaseName}")
//      val outFile = s"${txtAndAnnOutDir}/${fileBaseName}"
//      val bw = new FileWriter(new File(outFile))
//      bw.write(documentText)
//      bw.close
//
//    }
//    for (f <- xmlDir.listFiles() if f.getName().endsWith(".name")) {
//      logger.info(s"Reading ${f.getName()}")
////      val targetFile = new File(textDir, f.getName().dropRight(5) + ".txt")
////      val text = nxmlreader.read(f).text
////      writeTextFile(targetFile, text)
//    }    
//}