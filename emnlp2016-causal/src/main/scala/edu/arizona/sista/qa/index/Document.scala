package edu.arizona.sista.qa.index

import java.io.{FilenameFilter, File}
import nu.xom.Builder
import collection.mutable.ListBuffer

/**
 * 
 * User: mihais
 * Date: 3/1/13
 */
class Document (val docid:String, val sentences: List[Sentence])

class Sentence (val text: String, val offset:Int, val paragraph:Int)

object Document {
  def parse(file:File):Document = {
    val b = new Builder()
    //println(file)
    val docid = getDocidFromFileName(file)
    val sb = new ListBuffer[Sentence]
    val doc = b.build(file)
    val root = doc.getRootElement
    val sentences = root.getChildElements("sentence")
    for(i <- 0 to sentences.size - 1) {
      val se = sentences.get(i)
      val offset = Integer.valueOf(se.getAttribute("offset").getValue)
      val paragraph = Integer.valueOf(se.getAttribute("par").getValue)
      val text = se.getChild(0).getValue
      //println("\t" + offset + " " + paragraph + " " + text)
      val sentence = new Sentence(text, offset, paragraph)
      sb += sentence
    }

    new Document(docid, sb.toList)
  }

  def getDocidFromFileName(file:File):String = {
    val n = file.getName
    if (n.endsWith(".xml"))
      return n.substring(0, n.length - 4)
    n
  }
}

class DocumentIterator(val collectionDir:String) extends Iterator[Document] {
  private val files:Array[File] = readFiles()
  private var offset:Int = 0

  def readFiles(): Array[File] = {
    val dir = new File(collectionDir)
    dir.listFiles(new FilenameFilter {
      def accept(dir: File, name: String): Boolean = name.endsWith(".xml")
    })
  }

  def hasNext:Boolean = {
    return offset < files.length
  }

  def next(): Document = {
    val doc = Document.parse(files(offset))
    offset += 1
    doc
  }
}