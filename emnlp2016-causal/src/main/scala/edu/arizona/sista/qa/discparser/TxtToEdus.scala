package edu.arizona.sista.qa.discparser

import java.io.{FileOutputStream, PrintStream, FileReader, BufferedReader}

/**
 * Creates one EDU per sentence
 * Useful when the discourse parser crashes
 * User: mihais
 * Date: 8/14/13
 */
object TxtToEdus {
  def main(args:Array[String]) {
    val is = new BufferedReader(new FileReader(args(0)))
    val text = is.readLine()
    val sents = text.split("<s>|<p>")
    // for(s <- sents) println(s)
    is.close()

    val os = new PrintStream(new FileOutputStream(args(0) + ".edus"))
    for(s <- sents) {
      os.println(s"<edu>$s</edu>")
    }
    os.close()
  }
}
