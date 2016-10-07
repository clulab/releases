package edu.arizona.sista.qa.discparser

import scala.io.Source
import java.io.{PrintStream, FileOutputStream, File}
import edu.arizona.sista.processors.corenlp.CoreNLPProcessor
import java.util.regex.Pattern

/**
 * Prepares the answer files that are the input of the discourse parser
 * User: mihais
 * Date: 8/5/13
 */
object PrepareDiscParserInput {
  val URL = Pattern.compile("^\"?(http:|www\\.)", Pattern.CASE_INSENSITIVE)
  val EMAIL = Pattern.compile("[A-Z0-9._%-]+@[A-Z0-9.-]+\\.[A-Z]{2,4}", Pattern.CASE_INSENSITIVE)

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

      else if(docid != null && ! line.startsWith("question:")) {
        if(line.equals("**********")) {
          // save ab
          val doc = pipe.mkDocument(ab.toString())
          var os = new PrintStream(new FileOutputStream(dir + File.separator + docid + ".txt"))
          var firstSent = true
          for(sent <- doc.sentences) {
            if(! firstSent) os.print("<s>")
            var firstToken = true
            for(w <- sent.words) {
              var nw = w
              if(URL.matcher(w).find()) {
                println("FOUND URL: " + w)
                nw = "XURLX"
              } else if(EMAIL.matcher(w).matches()) {
                println("FOUND EMAIL: " + w)
                nw = "XEMAILX"
              }
              if(! firstToken) os.print(" ")
              os.print(nw)
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
          if(line.startsWith("answer:")) {
            ab.append(line.substring(7).trim)
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
