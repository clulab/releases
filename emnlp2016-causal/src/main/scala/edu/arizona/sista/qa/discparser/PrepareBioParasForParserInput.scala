package edu.arizona.sista.qa.discparser

import edu.arizona.sista.processors.corenlp.CoreNLPProcessor
import scala.io.Source
import java.io.{File, FileOutputStream, PrintStream}

/**
 * Prepares the paragraphs from Campbell's book to be the input of the discourse parser
 * User: mihais
 * Date: 8/30/13
 */
object PrepareBioParasForParserInput {
  def main(args:Array[String]) {
    val dir = args(1)
    var ab:StringBuilder = null
    var docid:String = null
    var count = 0
    var pipe = new CoreNLPProcessor()

    for(line <- Source.fromFile(args(0)).getLines()) {
      // println(line)
      if(line.startsWith("docid:")) {
        docid = line.substring(6).trim
        assert(docid.length > 0)
        ab = new StringBuilder
      }

      else if(docid != null) {
        if(line.equals("**********")) {
          // save ab
          val doc = pipe.mkDocument(ab.toString())
          var os = new PrintStream(new FileOutputStream(dir + File.separator + docid + ".txt"))
          var firstSent = true
          for(sent <- doc.sentences) {
            if(! firstSent) os.print("<s>")
            var firstToken = true
            for(w <- sent.words) {
              if(! firstToken) os.print(" ")
              os.print(w)
              firstToken = false
            }
            firstSent = false
          }
          os.println("<p>")
          os.close()

          count += 1
          docid = null
          ab = null
        } else {
          if(line.startsWith("text:")) {
            ab.append(line.substring(5).trim)
          } else {
            ab.append(line.trim)
          }
          ab.append(" ")
        }
      }
    }

    println(s"Processed $count answers.")
  }
}
