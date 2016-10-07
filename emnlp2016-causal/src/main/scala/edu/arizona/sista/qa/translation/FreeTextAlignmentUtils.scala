package edu.arizona.sista.qa.translation

import edu.arizona.sista.processors.{Processor, Document, Sentence}
import edu.arizona.sista.discourse.rstparser.TokenOffset
import java.io.{FilenameFilter, File, PrintWriter}
import edu.arizona.sista.qa.index.TermFilter
import edu.arizona.sista.struct.Counter
import org.slf4j.{Logger, LoggerFactory}
import edu.arizona.sista.processors.fastnlp.FastNLPProcessor
import scala.sys.process._
/**
 * Created by peter on 3/13/15.
 */
class FreeTextAlignmentUtils {

}

object FreeTextAlignmentUtils {
  val termFilter = new TermFilter()
  val logger = LoggerFactory.getLogger(classOf[FreeTextAlignmentUtils])
  lazy val queryProcessor: Processor = new FastNLPProcessor()



  val NTEMP = "nuclei_temp"
  val STEMP = "satellites_temp"
  val NTEMPBL = "nuclei_temp_baseline"
  val STEMPBL = "satellites_temp_baseline"
  val NTEMPRAND = "nuclei_temp_random"
  val STEMPRAND = "satellites_temp_random"
  val NTEMP_TRUERAND = "nuc_temp_truerand"
  val STEMP_TRUERAND = "sat_temp_truerand"

  /*
   *    Methods for Making Translation Matrices from Free Text
   */



  // Wrapper for when given Strings only
  def makePriorsFromStrings(text:Seq[String], wdir:String, filePrefix:String): Unit = {
    val counts = new Counter[String]
    for {
      line <- text
      words = line.split(" ")
      w <- words
    } counts.incrementCount(w)
    val total = counts.values.sum.toInt

    logger.debug("Priors computed for a collection of size " + total)
    MakeTranslationMatrix.savePriors(counts, total, wdir + "/" + filePrefix + ".priors")
  }

//  // Wrapper for when given Array of Sentences, saves the priors to the specified location
//  def makePriorsFromSentences(sentences:Array[Sentence], wdir:String, filePrefix:String) {
//
//    val counts = new Counter[String]
//    val total = makePriorsFromSentencesHelper(sentences, counts)
//
//    logger.debug("Priors computed for a collection of size " + total)
//    MakeTranslationMatrix.savePriors(counts, total, wdir + filePrefix + ".priors")
//  }
//
//  // Does the actual priors computation, updates the Counter[String] and returns the totals from this set of Sentences
//  def makePriorsFromSentencesHelper(sentences:Array[Sentence], counts:Counter[String]):Int = {
//    var total:Int = 0
//
//    for (sentIdx <- 0 until sentences.size) {
//      for (tokenIdx <- 0 until sentences(sentIdx).words.size) {
//        val word = sentences(sentIdx).words(tokenIdx)
//        val lemma = sentences(sentIdx).lemmas.get(tokenIdx)
//        val tag = sentences(sentIdx).tags.get(tokenIdx)
//
//        if (termFilter.validToken(word, lemma) && TermFilter.isContentTag(tag)) {
//          //add tokens to counter
//          counts.incrementCount(word)
//          total += 1
//
//        }
//      }
//    }
//
//    total
//  }
//
//

  /*
   *  Miscellaneous Helper Methods
   */

//  // make a partial annotation, does NOT include discourse
//  def mkPartialAnnotation(text:String):Document = {
//    val doc = queryProcessor.mkDocument(text)
//    queryProcessor.tagPartsOfSpeech(doc)
//    queryProcessor.lemmatize(doc)
//    queryProcessor.parse(doc)
//    doc.clear()
//    doc
//  }

  // Make a set of features based on desired view (e.g. lemmas, dependencies, words_content, etc.)
  def mkView(text:String, viewType:String, sanitizeForWord2Vec:Boolean = true): TransView = {
    val san = sanitizeString(text)
    // dependencies are cheap to produce via FastNLP
    val doc = queryProcessor.annotate(san)
    val view = new TransView(doc, sanitizeForW2V = sanitizeForWord2Vec)
    view.makeView(viewType)
    view
  }

  // Input: e.g. "/var/www/text.txt"
  // Output ("/var/www", "text.txt")
  def extractDir(fn:String):(String, String) = {
    val end = fn.lastIndexOf(File.separator)
    if(end == -1) return (".", fn)
    (fn.substring(0, end), fn.substring(end + 1))
  }


  // Get all files in a directory ending with a given extension
  def findFiles(collectionDir: String, fileExtension:String): Array[File] = {
    val dir = new File(collectionDir)
    dir.listFiles(new FilenameFilter {
      def accept(dir: File, name: String): Boolean = name.endsWith(fileExtension)
    })
  }




  // Makes a discourse annotated document from a text file
//  def makeDocFromPlainText(filename:String, lengthLimit:Int):Document = {
//    val lines = Source.fromFile(filename, "ISO-8859-1").getLines().toArray
//
//    // Check that the file isn't too long
//    if (lines.size > lengthLimit) {
//      println (s"Error: File $filename exceeds the current length limit (fileLength:${lines.size}/lengthLimit:$lengthLimit})")
//      System.exit(1)
//    }
//
//    val sb = new mutable.StringBuilder()
//    for (line <- lines) {
//      sb.append(line.trim() + " ")
//    }
//    val text = sb.mkString
//    val doc = discourseExplorer.mkDiscourseDoc(text)
//    doc
//  }



  // Command line call
  def exe(cmd:String) {
    logger.debug("Running command " + cmd)
    val exitCode = cmd.!
    if(exitCode != 0)
      throw new RuntimeException("ERROR: failed to execute command " + cmd)
  }

  // Removes line breaks and replaces other spaces with a true space
  def sanitizeString(in:String):String = {
    var os:String = in
    os = os.replaceAll ("[\r\n]+", "")       // line break(s)
    os = os.replaceAll ("\\s+", " ")         // any other spaces
    os
  }
}
