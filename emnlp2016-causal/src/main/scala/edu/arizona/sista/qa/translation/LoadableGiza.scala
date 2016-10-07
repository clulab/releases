package edu.arizona.sista.qa.translation

import scala.collection.mutable.{ArrayBuffer, ListBuffer}
import edu.arizona.sista.struct.Lexicon
import LoadableGiza._
import edu.arizona.sista.qa.linearalg.SparseMatrix
import java.io.PrintWriter

/**
 *
 * User: mihais
 * Date: 1/29/14
 */


trait LoadableGiza {
  def loadGizaTransProbs(filename:String, keepNulls:Boolean = false):(SparseMatrix[Double], Lexicon[String]) = {
    val lexicon = new Lexicon[String]
    val matrix = new SparseMatrix[Double](0.0)

    // each tuple stores a (dst, src, prob)
    println("-- Making a SparseMatrix from Giza file...")
    for(line <- io.Source.fromFile(filename).getLines()) {
      val tokens = line.split("\\s+")
      if(tokens.size != 3)
        throw new RuntimeException("ERROR: invalid line in translation file: " + line)

      if(keepNulls || (tokens(0) != NIL && tokens(1) != NIL)) {
        val src = lexicon.add(tokens(0))
        val dst = lexicon.add(tokens(1))
        val prob = tokens(2).toDouble
        matrix.set(src, dst, prob)
      }
    }

    println("-- Renormalizing after having discarded NULLs...")
    matrix.normalizeWithinRow()

    (matrix, lexicon)
  }

  /**
   * Loads the precomputed giza prior probabilities
   * Needs to run after loadGizaTransProbs
   * @param priorFile File with prior probs
   * @return
   */
  def loadGizaPriors(priorFile:String, lexicon:Lexicon[String]):Array[Double] = {
    // for debugging
    println(s"-- Loading Giza Priors file: $priorFile")
    val lines = new ListBuffer[(Int, Double)]
    for(line <- io.Source.fromFile(priorFile).getLines()) {
      val tokens = line.split("\\s+")
      if(tokens.length != 2)
        throw new RuntimeException("ERROR: invalid line in priors file: " + line)
      val id = lexicon.add(tokens(0))
      val p = tokens(1).toDouble
      lines += new Tuple2(id, p)
    }

    val ps = new Array[Double](lexicon.size)
    for(line <- lines) ps(line._1) = line._2
    ps
  }

  /**
   * Loads the priors array to a file that can be loaded with the loadGizaPriors method
   * @param priors Array of prior probs
   * @param lexicon a Lexicon that serves as a word-to-index hashmap
   * @param filename the filename for saving
   * @return
   */
  def saveGizaPriors(priors:Array[Double], lexicon:Lexicon[String], filename:String) {
    // Saves the priors file
    val pw = new PrintWriter(filename)

    for (i <- 0 until priors.size) {
      val word = lexicon.get(i)
      val weight = priors(i)
      pw.println(word + " " + weight)
    }
    pw.close()
  }

}








object LoadableGiza {
  val NIL = "NULL"
}