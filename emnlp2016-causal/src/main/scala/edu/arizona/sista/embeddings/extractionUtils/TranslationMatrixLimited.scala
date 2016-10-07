package extractionUtils

import java.io.PrintWriter

import edu.arizona.sista.struct.Lexicon

import scala.collection.mutable.{ListBuffer, ArrayBuffer}

/**
  * Created by bsharp on 4/4/16.
  */
class TranslationMatrixLimited (var transProbs: Array[SrcRow],
                                var lexicon: Lexicon[String],
                                var priors: Array[Double]) {

  // Load from files
  def this (filenameMatrix: String, filenamePriors:String) = {
    this (new Array[SrcRow](0), new Lexicon[String], new Array[Double](0))
    val data = loadGizaTransProbs(filenameMatrix)
    transProbs = data._1
    lexicon = data._2
    priors = loadGizaPriors(filenamePriors, lexicon)
  }

  private def loadGizaTransProbs(filename:String, keepNulls:Boolean = false):(Array[SrcRow], Lexicon[String]) = {
    val lexicon = new Lexicon[String]
    val matrix = new ArrayBuffer[SrcRow]

    // each tuple stores a (dst, src, prob)
    println("-- Making a SparseMatrix from Giza file...")
    for(line <- io.Source.fromFile(filename).getLines()) {
      val tokens = line.split("\\s+")
      if(tokens.size != 3)
        throw new RuntimeException("ERROR: invalid line in translation file: " + line)

      if(keepNulls || (tokens(0) != TranslationMatrixLimited.NIL && tokens(1) != TranslationMatrixLimited.NIL)) {
        val src = lexicon.add(tokens(0))
        val dst = lexicon.add(tokens(1))
        val prob = tokens(2).toDouble
        //matrix.set(src, dst, prob)

        // Add more rows if necessary
        if (src >= matrix.size) {
          for (i <- 0 until ((src - matrix.size) + 1) ) {
            matrix.append(new SrcRow())
          }
        }

        // Store the translation Probability
        matrix(src).add(dst, prob)
      }
    }

    println("-- Renormalizing after having discarded NULLs...")
    matrix.map(row => row.sumToOne())

    (matrix.toArray, lexicon)
  }

  private def loadGizaPriors(priorFile:String, lexicon:Lexicon[String]):Array[Double] = {
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

  def getPrior(id:Int):Double = {
    // a word unseen in the prior collection
    if(id >= priors.length) return Double.MinPositiveValue
    val p = priors(id).toDouble
    if(p == 0.0) return Double.MinPositiveValue
    p
  }

  def getTransProb(dst:String, src:String): Double = {
    val dstIndex = lexicon.get(dst)
    val srcIndex = lexicon.get(src)
    // If not defined...
    if (dstIndex.isEmpty || srcIndex.isEmpty) return 0.0
    // Otherwise...
    getTransProb(dstIndex.get, srcIndex.get)
  }

  def getTransProb(dst:Int, src:Int):Double = {
    if (src >= transProbs.length) return 0.0
    val row = transProbs(src)
    val location = row.dstIndices.indexOf(dst)
    if (location == -1) return 0.0

    row.dstValues(location)
  }

  def prob(qFeatures:Array[String], aFeatures:Array[String], lambda:Double, errorOut:PrintWriter = null):Double = {
    var logP:Double = 0

    // convert strings to lexicon indices
    val qids = toIndices(qFeatures)
    val aids = toIndices(aFeatures)

    // Compute P(Q|A)
    for (qid <- qids) {
      // Compute P(q|A)
      val p = scala.math.log10( smoothedWordProb(qid, aids, lambda, errorOut) )
      //logger.debug (s"log P(q|A) [${lexicon.get(qid)}] = $p")
      logP += p
      if (errorOut != null) errorOut.println (s"log P(q|A) [${lexicon.get(qid)}] = $p   (logP running total = $logP)")
    }

    if (errorOut != null) errorOut.println (s"Final log P(Q|A) = $logP")
    logP
  }
  def smoothedWordProb(qid:Int, aids:Array[Int], lambda:Double, errorOut:PrintWriter = null):Double = {
    // The unsmoothed probability Pml(q|A)
    val p = wordProb(qid, aids, errorOut)

    // the prior probability Pml(q|C), where C is the collection of all answers
    val prior = getPrior(qid)

    // The smoothed probability P(q|A) = (1-L)*Pml(q|A) + L*Pml(q|C)
    val smoothed = ((1-lambda) * p) + (lambda * prior)

    //logger.debug(s"\tunsmoothed [${lexicon.get(qid)}] = $p, prior = $prior")
    if (errorOut != null) errorOut.println(s"\tunsmoothed [${lexicon.get(qid)}] = $p, prior = $prior")
    smoothed
  }

  /** Computes the unsmoothed probability Pml(q|A) */
  def wordProb(qid:Int, aids:Array[Int], errorOut:PrintWriter = null):Double = {
    var prob:Double = 0
    val pml:Double = 1.0 / aids.size.toDouble // Pml(a|A)

    //logger.debug (s"\t\tPml(a|A) = $pml")
    if (errorOut != null) errorOut.println (s"\t\tPml(a|A) = $pml")

    for (aid <- aids) {
      // Gus: maybe add in count function to use count as numerator
      val trans = getTransProb(qid, aid) // T(q|a)
      prob += (trans * pml)

      //logger.debug(s"\t\ttrans [dst = ${lexicon.get(qid)}, src = ${lexicon.get(aid)}] = $trans")
      if (errorOut != null) errorOut.println(s"\t\ttrans [dst = ${lexicon.get(qid)}, src = ${lexicon.get(aid)}] = $trans")
    }
    prob
  }



  def toIndices(words:Array[String]):Array[Int] = {
    this.synchronized {
      val ids = new ArrayBuffer[Int]()
      for(word <- words) {
        // add new words to the lexicon
        // this is important for two reasons:
        //   all words get some small probability through the prior
        //   if words in question and answer are identical (even if new) they get very high prob
        val id = lexicon.add(word)
        ids += id
      }
      ids.toArray
    }
  }



}

object TranslationMatrixLimited {
  val NIL = "NULL"
}


class SrcRow (val dstIndices: ArrayBuffer[Int], val dstValues: ArrayBuffer[Double]) {
  def this() = {
    this (new ArrayBuffer[Int], new ArrayBuffer[Double])
  }

  def add(dst:Int, value:Double): Unit = {
    dstIndices.append(dst)
    dstValues.append(value)
  }

  def scalarMultiply(factor:Double) {
    for (i <- 0 until dstValues.size) {
      dstValues(i) *= factor
    }
  }

  // Scales the vector such that the sum of all non-zero elements is one (excluding the defaultValue)
  def sumToOne() {
    if (dstValues.size == 0) return
    val sum = this.sum
    if (sum != 0.0) {
      scalarMultiply(1.0 / sum)
    }
  }

  // Sum all non-default values of the vector
  def sum:Double = dstValues.sum
}


