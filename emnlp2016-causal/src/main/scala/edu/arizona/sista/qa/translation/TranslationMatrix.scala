package edu.arizona.sista.qa.translation

import scala.collection.mutable.ArrayBuffer
import scala.language.postfixOps
import org.slf4j.LoggerFactory
import TranslationMatrix._
import edu.arizona.sista.qa.linearalg._
import scala.Predef._
import edu.arizona.sista.struct.{Lexicon,Counter}
import java.io.PrintWriter
import edu.arizona.sista.qa.word2vec.Word2vec
import edu.arizona.sista.utils.MathUtils.softmax
import edu.arizona.sista.utils.{Serialization, MathUtils, Profiler}
import scala.collection.parallel.ForkJoinTaskSupport
import scala.util.control.Breaks._
import scala.collection.immutable.IndexedSeq
import java.util.concurrent.atomic.AtomicInteger
import scala.reflect.ClassTag

/**
 * Load and handle queries to the translation matrix
 * User: peter, mihai
 * Date: 8/2/13
 */

@SerialVersionUID(2L)
class TranslationMatrix[B](var transProbs: Matrix[B],
                           var lexicon: Lexicon[String],
                           var priors: Array[B],
                         //val matrixFile:String,
                         //val priorFile:String,
                         val selfAssociationProb:B)(implicit num: Fractional[B], tag: ClassTag[B]) extends LoadableGiza with Serializable {
  /**
   * Stores the translation probabilities computed by Giza and its lexicon for both source and destination
   * Columns are destination words, rows are source words (since the probabilites are P(dst | src) )
   * Summing up the values in a row should be 1 (a probability distribution)
   */

  // this is used in the construction of page rank matrices, and is set as a variable because the implementation
  // changes depending on whether we're using a sparse or dense matrix

  var rowEmpty: Int => Boolean = { i => this.transProbs.asInstanceOf[SparseMatrix[B]].rowVecs(i).labels.isEmpty }
  // mainly for backward compatibility, try to use the constructor that takes a matrix filename instead,
  // or call importGiza directly after calling this
  def this(selfAssociationProb: B)(implicit num: Fractional[B], tag: ClassTag[B]) = {
    this(new SparseMatrix[B](num.zero), new Lexicon[String], new Array[B](0), selfAssociationProb)
  }

  def this(matrixFile: String, priorFile: String, selfAssociationProb:B)(implicit num: Fractional[B], tag: ClassTag[B]) = {
    this(selfAssociationProb)
    importGiza(matrixFile, priorFile, 0)
  }

  // load a giza matrix
  def importGiza(matrixFile:String, priorFile:String, topN:Int, makeDense: Boolean = false){
    //(transProbs, lexicon) = loadGizaTransProbs(matrixFile)
    val (tp: SparseMatrix[Double], lex: Lexicon[String]) = loadGizaTransProbs(matrixFile)
    if (makeDense) {
      val (transProbs, rowEmpty) = tp.toDense(Some(lex.size), Some(lex.size))
      // arghhh this is terrible sorry! will probably crash if B is not Double
      this.transProbs = transProbs.asInstanceOf[Matrix[B]]
      this.rowEmpty = i => rowEmpty(i)
      lexicon = lex
      priors = loadGizaPriors(priorFile, lexicon).asInstanceOf[Array[B]]
    } else {
      this.transProbs = tp.asInstanceOf[Matrix[B]]
      this.lexicon = lex
      priors = loadGizaPriors(priorFile, lexicon).asInstanceOf[Array[B]]
      rowEmpty = i => tp.rowVecs(i).labels.isEmpty

      // Prune matrix, such that each word will only retain it's top N associations.
      // This greatly speeds the matrix multiplication, while removing weights that are many orders of magnitude too
      // small to make a difference.
      if (topN > 0) {
        logger.info ("* Pruning all but top " + topN + " weights in translation matrix (for each word)... ")
        prune(topN)
      }
    }
  }

  // destructively prune the matrix contained within this instance
  def prune(topN: Int) = {
    transProbs.prune(topN)
    transProbs.normalizeWithinRow
  }


  private def sparseLookupWord(sparseVector: SparseVector[B], sparseIndex: Int): String = {
    lexicon.get(sparseVector.labels(sparseIndex))
  }

  def getTopAssociatesAndProbabilities(index:Int,
                                       topN:Int,
                                       filterPredicate: Option[String => Boolean] = None): List[(String, B)] =
    transProbs match {
      case tp : DenseMatrix => {
        import num._
        val row = transProbs.getRowVec(index)
        val indicesToSearch = filterPredicate match {
          case None => (0 until row.size)
          case Some(predicate) => (0 until row.size).filter(ix => predicate(lexicon.get(ix)))
        }

        val lexiconIndices = MathUtils.nBest[Int](ix => row(ix).toDouble)(indicesToSearch.toIterable, topN)
        lexiconIndices.map(pair => (lexicon.get(pair._1), row(pair._1)))
      }
      case tp : SparseMatrix[B] => {
        import num._
        val sparseVector = transProbs.getSparseRowVec(index)
        require(sparseVector.values.size == sparseVector.labels.size, "values and labels must be same size")
        val indicesToSearch = filterPredicate match {
          case None => (0 until sparseVector.size)
          case Some(predicate) => (0 until sparseVector.size).filter(ix => predicate(sparseLookupWord(sparseVector, ix)))
        }
        // return the best indices into the sparse matrix

        val sparseIndicesAndProbs: List[(Int, Double)] = MathUtils.nBest[Int](ix => sparseVector.values(ix).toDouble)(indicesToSearch.toIterable, topN)
        sparseIndicesAndProbs.map(pair => (sparseLookupWord(sparseVector, pair._1), sparseVector.values(pair._1)))
      }
  }

  def getTopAssociates(index: Int, topN: Int): List[String] =
    getTopAssociatesAndProbabilities(index, topN).map(_._1)

  // Given a cue word (e.g. "apple"), returns the topN words associated with that cue word (e.g. "pear", "tree")
  def getTopAssociates(cueWord:String, topN:Int): List[String] =
    lexicon.get(cueWord).map({
      index =>  getTopAssociates(index, topN)
    }).getOrElse(List.empty[String])

  def save(filenamePrefix:String, binaryFormat: Boolean = false){
    if (binaryFormat) {
      Serialization.serialize(this, binaryPath(filenamePrefix))
    } else {
      transProbs.save(filenamePrefix)
      lexicon.saveTo(filenamePrefix + ".lexicon")
      import num._
      saveGizaPriors(priors.map(_.toDouble), lexicon, filenamePrefix + ".priors")
    }
  }

  //non-Giza
  def load(filenamePrefix:String, dense:Boolean = false){
    transProbs = if (dense)
      DenseMatrix.load(filenamePrefix).asInstanceOf[Matrix[B]]
    else
      SparseMatrix.loadDouble(filenamePrefix).asInstanceOf[Matrix[B]]
    lexicon = Lexicon.loadFrom(filenamePrefix + ".lexicon")
    priors = loadGizaPriors(filenamePrefix + ".priors", lexicon).asInstanceOf[Array[B]]
  }

  def getPrior(id:Int):Double = {
    import num._
    // a word unseen in the prior collection
    if(id >= priors.length) return FUNCTIONAL_ZERO
    val p = priors(id).toDouble
    if(p == 0.0) return FUNCTIONAL_ZERO
    p
  }

  def getTransProb(dst:Int, src:Int):B = {
    transProbs.get(src, dst)
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


  /**
   * Entry point for this class: P(Q|A)
   * @param qFeatures Tokens in question (destination)
   * @param aFeatures Tokens in answer (source)
   * @param lambda Interpolation hyperparam
   * @return log(P(Q|A))
   */
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
      val trans = transProb (qid, aid) // T(q|a)
      prob += (trans * pml)

      //logger.debug(s"\t\ttrans [dst = ${lexicon.get(qid)}, src = ${lexicon.get(aid)}] = $trans")
      if (errorOut != null) errorOut.println(s"\t\ttrans [dst = ${lexicon.get(qid)}, src = ${lexicon.get(aid)}] = $trans")
    }
    prob
  }

  def transProb(qid:Int, aid:Int):Double = {
    import num._
    // Check if the source and destination words are the same
    // p(w|w) is defined as 1.0 (which is 0.5 after normalization)
    if (qid == aid && selfAssociationProb != 0.0) return selfAssociationProb.toDouble

    val self = getTransProb(qid, qid).toDouble
    if (self == 1.0) return 0.0 // prevents divide by zero below for cases where the self-association is 1.0

    val w = getTransProb(qid, aid).toDouble
    val normalizedW = (w / (1.0 - self)) * (1 - selfAssociationProb.toDouble)

    normalizedW
  }

  def *(other: TranslationMatrix[B]) = {
    val prod = transProbs * other.transProbs
    new TranslationMatrix[B](prod, lexicon, priors, selfAssociationProb)
  }

  private def parallelIndices(nThreads: Option[Int] = None) = {
    val pc = (0 until transProbs.numRows).par
    if (nThreads.isDefined) {
      pc.tasksupport = new ForkJoinTaskSupport(new scala.concurrent.forkjoin.ForkJoinPool(nThreads.get))
    }
    pc
  }

  // construct a new matrix with the same lexicon as this matrix, by repeatedly calling a function that takes the
  // index of a word in the lexicon and returns a row (represented as a sparse vector),
  // aggregating the resulting sparse vectors into a new translation matrix. Used for creating higher-order matrices.
  private def matrixFromRowFunction(rowConstructor: Int => SparseVector[B], nThreads: Option[Int] = None): TranslationMatrix[B] = {
    val count: AtomicInteger = new AtomicInteger(0)
    val rows: Seq[SparseVector[B]] = (for {
      rowIdx <- parallelIndices(nThreads)
      row = rowConstructor(rowIdx)
      _ = {count.incrementAndGet; if (count.get % 1000 == 0) println(s"count = $count")}
    } yield row).seq
    val rowBuffer = new ArrayBuffer[SparseVector[B]](rows.size)
    for (row <- rows) {
      rowBuffer.append(row)
    }
    val newMatrix = new SparseMatrix[B](num.zero, rowBuffer)
    new TranslationMatrix[B](newMatrix, lexicon, priors, selfAssociationProb)
  }

  // add together the distributions corresponding
  private def interpolate(wordsAndWeights: Iterable[(String, B)]) = {
    import num._
    val vecs = for {
      (w, p) <- wordsAndWeights
      rowIdx <- lexicon.get(w)
      row = transProbs.getSparseRowVec(rowIdx)
    } yield row * p
    val v = if (vecs.isEmpty) new SparseVector[B](num.zero) else vecs.reduce(_ + _)
    val s = v.sum
    if (s != num.zero) v / s else v
  }

  private def mkSecondOrderVectorUsingTopAssociates(N_closest: Int)(index: Int): SparseVector[B] = {
    val associatesAndWeights: List[(String, B)] = getTopAssociatesAndProbabilities(index, N_closest)
    interpolate(associatesAndWeights)
  }

  /**
   * create a new matrix where the row for each word, w, is the weighted sum of each of w's top associates in this
   * matrix
   */
  def secondOrderMatrix(N_closest: Int, nThreads: Option[Int] = None): TranslationMatrix[B] =
    matrixFromRowFunction(mkSecondOrderVectorUsingTopAssociates(N_closest), nThreads)

  def lookup(word: String): Option[SparseVector[B]] = for {
    index <- lexicon.get(word)
    if (index < transProbs.numRows)
  } yield transProbs.getSparseRowVec(index)

  def denseLookup(word: String): Option[Array[Float]] = for {
    index <- lexicon.get(word)
    if (index < transProbs.numRows)
  } yield transProbs.asInstanceOf[DenseMatrix].accessRow(index)

  def distances(xs: Traversable[String], ys: Traversable[String]): Traversable[Double] = {
    import num._
    if (transProbs.isInstanceOf[DenseMatrix]) {
      for {
        x <- xs.flatMap(denseLookup)
        y <- ys.flatMap(denseLookup)
      } yield DenseMatrix.jsDistance(x.map(_.toDouble), y.map(_.toDouble))
    } else {
      for {
        x <- xs.flatMap(lookup)
        y <- ys.flatMap(lookup)
      } yield x.jsDistance(y)
    }
  }

  def avgDistance(xs: Traversable[String], ys: Traversable[String]): Option[Double] = {
    avgDistance(distances(xs, ys).toSeq)
  }

  def avgDistance(distances: Seq[Double]) = {
    if (distances.size != 0) Some(distances.sum / distances.size) else None
  }

  def minDistance(xs: Traversable[String], ys: Traversable[String]): Option[Double] = {
    minDistance(distances(xs, ys))
  }

  def minDistance(distances: Traversable[Double]) =
    if (distances.nonEmpty)
      Some(distances.reduce(math.min(_, _)))
    else
      None

  def maxDistance(xs: Traversable[String], ys: Traversable[String]): Option[Double] = {
    maxDistance(distances(xs, ys))
  }

  def maxDistance(distances: Traversable[Double]) =
    if (distances.nonEmpty)
      Some(distances.reduce(math.max(_, _)))
    else
      None

  // add up the distributions corresponding to each word, then normalize
 def compose(xs: Traversable[String]): Option[SparseVector[B]] = {
    val vecs = xs.flatMap(lookup)
    if (vecs.nonEmpty) {
      val s = vecs.reduce(_ + _)
      val ss = s.sum
      Some(if (ss != 0.0) s / ss else s)
    }
    else None
  }

  def denseCompose(xs: Traversable[String]): Option[Array[Double]] = {
    import num._
    val N = transProbs.numCols
    val x = new Array[Double](N)
    var foundSome: Boolean = false
    for (s <- xs) {
      denseLookup(s).foreach ( y => {
        foundSome = true
        for (i <- 0 until N)
          x(i) += y(i).toDouble
      })
    }
    if (foundSome) {
      val ss = x.sum
      Some(if (ss != 0.0) x.map(_ / ss) else x)
    } else 
      None
  }

  def textDistance(xs: Traversable[String], ys: Traversable[String]): Option[Double] = {
    if (transProbs.isInstanceOf[DenseMatrix]) {
      for {
        x <- denseCompose(xs)
        y <- denseCompose(ys)
      } yield DenseMatrix.jsDistance(x, y)
    } else {
      for {
        x <- compose(xs)
        y <- compose(ys)
      } yield x.jsDistance(y)
    }
  }

  def sparseRows: IndexedSeq[SparseVector[B]] = for {
    i <- 0 until transProbs.numRows
  } yield transProbs.getSparseRowVec(i)

  def distributionAssociateFrequencies = {
    val c = new Counter[Integer]
    for (row <- sparseRows; label <- row.labels) {
      c.incrementCount(label)
    }
    c
  }

  def destinationProbabilities = {
    import num._
    val c = new Counter[Integer]
    // p(d) = \sum_{s} p(d,s) = \sum_{s} p(d|s) p(s)
    for ((row, rowIdx) <- sparseRows.zipWithIndex; (label, dstGivenSrc) <- (row.labels zip row.values)) {
      c.incrementCount(label, dstGivenSrc.toDouble * priors(rowIdx).toDouble)
    }
    c
  }

  def convertToFloat: TranslationMatrix[Float] = {
    import num._
    transProbs match {
      case sparseMat : SparseMatrix[_] => new TranslationMatrix[Float](sparseMat.convertToFloat, lexicon,
        priors.map(_.toFloat), selfAssociationProb.toFloat)
      case _ => sys.error("dense float matrix not implemented")
    }
  }

  def convertToDouble: TranslationMatrix[Double] = {
    import num._
    transProbs match {
      case sparseMat : SparseMatrix[_] => new TranslationMatrix[Double](sparseMat.convertToDouble, lexicon,
        priors.map(_.toDouble), selfAssociationProb.toDouble)
      case _ => sys.error("this is a dense matrix, so it should already be double")
    }
  }

  def filter(wordsToKeep: Set[String]): TranslationMatrix[B] = {
    val wordsAndIndices: Seq[(String, Int)] = wordsToKeep.toSeq.map(word => (word, lexicon.get(word).get)).sortBy(_._2)
    val newLexicon = new Lexicon[String]
    val oldToNewIndices = (for {
      (word, index) <- wordsAndIndices
    } yield (index -> newLexicon.add(word))).toMap
    val oldIndicesToKeep: Set[Int] = wordsAndIndices.map(_._2).toSet
    def filterRow(v: SparseVector[B]) = {
      val (indices, values) = (for {
        (oldIndex, value) <- v.labels zip v.values
        if oldIndicesToKeep.contains(oldIndex)
      } yield (oldToNewIndices(oldIndex), value)).unzip
      new SparseVector[B](v.defaultValue, indices.toArray, values.toArray)
    }
    val rows = for {
      index <- oldIndicesToKeep.toSeq.sorted
    } yield filterRow(transProbs.getSparseRowVec(index))
    val newTransProbs = new SparseMatrix[B](transProbs.asInstanceOf[SparseMatrix[B]].defaultValue, rows.to[collection.mutable.ArrayBuffer])
    newTransProbs.setMatrixSize(newLexicon.size, newLexicon.size)
    val newPriors = priors.zipWithIndex.filter(p => oldIndicesToKeep.contains(p._2)).map(_._1)

    new TranslationMatrix[B](newTransProbs, newLexicon, newPriors, selfAssociationProb)
  }

  def filter(counter: Counter[String], K: Int): TranslationMatrix[B] = {
    val wordsToKeep: Set[String] = lexicon.keySet.toSeq.map(word => (word, counter.getCount(word))).sortBy(p => - p._2).take(K).map(_._1).toSet
    filter(wordsToKeep)
  }
}

object TranslationMatrix {
  val logger = LoggerFactory.getLogger(classOf[TranslationMatrix[Double]])
  val FUNCTIONAL_ZERO = 1e-9
  val NIL = "NULL"

  private def binaryPath(prefix: String) = prefix + ".TranslationMatrix.dat"

  // allow passing default selfProb arguments
  def smartLoadDouble(filenamePrefix: String, selfProb: Double = 0.5, isDense: Boolean = false) =
    smartLoad[Double](filenamePrefix, selfProb, isDense)

  def smartLoadFloat(filenamePrefix: String, selfProb: Float = 0.5f, isDense: Boolean = false) =
    smartLoad[Float](filenamePrefix, selfProb, isDense)

  def smartLoad[B](filenamePrefix: String, selfProb: B, isDense: Boolean = false)
                  (implicit num: Fractional[B], tag: ClassTag[B]): TranslationMatrix[B] = {
    def pathExists(path: String) = new java.io.File(path).exists

    if (pathExists(binaryPath(filenamePrefix))) { // binary serialized
      Serialization.deserialize[TranslationMatrix[B]](binaryPath(filenamePrefix))
    } else if (pathExists(filenamePrefix + ".lexicon") &&
                pathExists(filenamePrefix + ".matrix") &&
                pathExists(filenamePrefix + ".priors")) { // ascii serialized
      val matrix = new TranslationMatrix[B](selfProb)
      matrix.load(filenamePrefix, isDense)
      matrix
    } else if (pathExists(filenamePrefix + ".matrix") &&
                pathExists(filenamePrefix + ".priors")) { // giza format
      val matrix = new TranslationMatrix[B](selfProb)
      matrix.importGiza(filenamePrefix + ".matrix", filenamePrefix + ".priors", 0, isDense)
      matrix
    } else throw new Exception(s"could not find a valid giza or serialized translation matrix with prefix $filenamePrefix")
  }
}
