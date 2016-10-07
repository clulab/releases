package edu.arizona.sista.qa.word2vec

import java.io.PrintWriter

import org.slf4j.LoggerFactory

import scala.collection.mutable.ArrayBuffer
import scala.io.Source

/**
  * Created by bsharp on 4/25/16.
  */
class Word2VecRelation (matrixFilenameTarget:String,
                        matrixFilenameContext:String,
                        wordsToUse: Option[Set[String]] = None) {

  // Load the target and context matrices for the relation
  val (matrixTarget, _) = Word2VecRelation.loadMatrix(matrixFilenameTarget, wordsToUse)
  val (matrixContext, _) = Word2VecRelation.loadMatrix(matrixFilenameContext, wordsToUse)

  val targetDimensions = matrixTarget.values.head.length
  val contextDimensions = matrixContext.values.head.length


  def saveMatrices(prefix: String) {
    // Save the target matrix
    val pwt = new PrintWriter(prefix + ".target")
    pwt.println(s"${matrixTarget.size}, $targetDimensions")
    for ((word, vec) <- matrixTarget) {
      val strRep = vec.map(_.formatted("%.6f")).mkString(" ")
      pwt.println(s"$word $strRep")
    }
    pwt.close()
    // Save the context matrix
    val pwc = new PrintWriter(prefix + ".context")
    pwc.println(s"${matrixContext.size}, $contextDimensions")
    for ((word, vec) <- matrixContext) {
      val strRep = vec.map(_.formatted("%.6f")).mkString(" ")
      pwc.println(s"$word $strRep")
    }
    pwc.close()
  }


}

object Word2VecRelation {
  val logger = LoggerFactory.getLogger(classOf[Word2VecRelation])

  private def loadMatrix(mf:String, wordsToUse: Option[Set[String]]):(Map[String, Array[Double]], Int) = {
    logger.debug("Started to load word2vec matrix from file " + mf + "...")
    val m = new collection.mutable.HashMap[String, Array[Double]]()
    var first = true
    var dims = 0
    for((line, index) <- Source.fromFile(mf, "iso-8859-1").getLines().zipWithIndex) {
      val bits = line.split("\\s+")
      if(first) {
        dims = bits(1).toInt
        first = false
      } else {
        if (bits.length != dims + 1) {
          println(s"${bits.length} != ${dims + 1} found on line ${index + 1}")
        }
        assert(bits.length == dims + 1)
        val w = bits(0)
        if (wordsToUse.isEmpty || wordsToUse.get.contains(w)) {
          val weights = new Array[Double](dims)
          var i = 0
          while(i < dims) {
            weights(i) = bits(i + 1).toDouble
            i += 1
          }
          norm(weights)
          m.put(w, weights)
        }
      }
    }
    logger.debug("Completed matrix loading.")
    (m.toMap, dims)
  }


  /** Normalizes this vector to length 1, in place */
  private def norm(weights:Array[Double]) {
    var i = 0
    var len = 0.0
    while(i < weights.length) {
      len += weights(i) * weights(i)
      i += 1
    }
    len = math.sqrt(len)
    i = 0
    if (len != 0) {
      while(i < weights.length) {
        weights(i) /= len
        i += 1
      }
    }
  }

  /**
    * Computes the cosine similarity between two texts, according to the word2vec matrix
    * IMPORTANT: t1, t2 must be arrays of words, not lemmas!
    */
  def textSimilarity(t1:Iterable[String], t2:Iterable[String],
                     m1: Map[String, Array[Double]], m2:Map[String, Array[Double]],
                     d1: Int, d2: Int):Double = {
    val st1 = new ArrayBuffer[String]()
    t1.foreach(st1 += Word2vec.sanitizeWord(_))
    val st2 = new ArrayBuffer[String]()
    t2.foreach(st2 += Word2vec.sanitizeWord(_))
    sanitizedTextSimilarity(st1, st2, m1, m2, d1, d2)
  }

  def bidirTextSimilarity(t1:Iterable[String], t2:Iterable[String],
                          m1: Map[String, Array[Double]], m2:Map[String, Array[Double]],
                          d1: Int, d2: Int,
                          m1b: Map[String, Array[Double]], m2b:Map[String, Array[Double]],
                          d1b: Int, d2b: Int):Double = {
    val st1 = new ArrayBuffer[String]()
    t1.foreach(st1 += Word2vec.sanitizeWord(_))
    val st2 = new ArrayBuffer[String]()
    t2.foreach(st2 += Word2vec.sanitizeWord(_))
    sanitizedBidirTextSimilarity(st1, st2, m1, m2, d1, d2, m1b, m2b, d1b, d2b)
  }

  /**
    * Computes the cosine similarity between two texts, the first uses the target matrix,
    * the second uses the context matrix.
    * IMPORTANT: words here must already be normalized using Word2vec.sanitizeWord()!
    * IMPORTANT: Directionality matters!
    */
  def sanitizedTextSimilarity(st1:Iterable[String], st2:Iterable[String],
                              m1: Map[String, Array[Double]], m2:Map[String, Array[Double]],
                              d1:Int, d2:Int):Double = {
    val v1 = makeCompositeVector(st1, m1, d1)
    val v2 = makeCompositeVector(st2, m2, d2)
    dotProduct(v1, v2)
  }

  def sanitizedBidirTextSimilarity(st1:Iterable[String], st2:Iterable[String],
                                   m1: Map[String, Array[Double]], m2:Map[String, Array[Double]],
                                   d1:Int, d2:Int,
                                   m1b: Map[String, Array[Double]], m2b:Map[String, Array[Double]],
                                   d1b:Int, d2b:Int):Double = {
    val v1 = makeCompositeVector(st1, m1, d1)
    val v2 = makeCompositeVector(st2, m2, d2)

    val v1b = makeCompositeVector(st1, m1b, d1b)
    val v2b = makeCompositeVector(st2, m2b, d2b)

    (0.5 * dotProduct(v1, v2)) + (0.5 * dotProduct(v1b, v2b))
  }


  private def makeCompositeVector(t:Iterable[String], matrix:Map[String, Array[Double]], dim:Int):Array[Double] = {
    val vTotal = new Array[Double](dim)
    for(s <- t) {
      val v = matrix.get(s)
      if(v.isDefined) add(vTotal, v.get, dim)
    }
    Word2VecRelation.norm(vTotal)
    vTotal
  }


  /**
    * Finds the average word2vec similarity between any two words in these two texts
    * IMPORTANT: words here must be words not lemmas!
    */
  def avgSimilarity(t1:Iterable[String], t2:Iterable[String],
                    m1: Map[String, Array[Double]], m2:Map[String, Array[Double]]):Double = {
    val st1 = new ArrayBuffer[String]()
    t1.foreach(st1 += Word2vec.sanitizeWord(_))
    val st2 = new ArrayBuffer[String]()
    t2.foreach(st2 += Word2vec.sanitizeWord(_))
    val (score, pairs) = sanitizedAvgSimilarity(st1, st2, m1, m2)

    score
  }

  def bidirAvgSimilarity(t1:Iterable[String], t2:Iterable[String],
                         m1: Map[String, Array[Double]], m2:Map[String, Array[Double]],
                         m1b: Map[String, Array[Double]], m2b:Map[String, Array[Double]]):Double = {
    val st1 = new ArrayBuffer[String]()
    t1.foreach(st1 += Word2vec.sanitizeWord(_))
    val st2 = new ArrayBuffer[String]()
    t2.foreach(st2 += Word2vec.sanitizeWord(_))
    val (score, pairs) = sanitizedBidirAvgSimilarity(st1, st2, m1, m2, m1b, m2b)

    score
  }

  /**
    * Finds the average word2vec similarity between any two words in these two texts
    * IMPORTANT: words here must already be normalized using Word2vec.sanitizeWord()!
    * Changelog: (Peter/June 4/2014) Now returns words list of pairwise scores, for optional answer justification.
    */
  def sanitizedAvgSimilarity(st1:Iterable[String], st2:Iterable[String],
                             m1: Map[String, Array[Double]], m2:Map[String, Array[Double]]):(Double, ArrayBuffer[(Double, String, String)]) = {
    // Top words
    var pairs = new ArrayBuffer[(Double, String, String)]

    var avg = 0.0
    var count = 0
    for(s1 <- st1) {
      val v1 = m1.get(s1)
      if(v1.isDefined) {
        for(s2 <- st2) {
          val v2 = m2.get(s2)
          if(v2.isDefined) {
            val s = dotProduct(v1.get, v2.get)
            avg += s
            count += 1

            // Top Words
            pairs.append ( (s, s1, s2) )
          }
        }
      }
    }
    if(count != 0) (avg / count, pairs)
    else (0, pairs)
  }

  def sanitizedBidirAvgSimilarity(st1:Iterable[String], st2:Iterable[String],
                                  m1: Map[String, Array[Double]], m2:Map[String, Array[Double]],
                                  m1b: Map[String, Array[Double]], m2b:Map[String, Array[Double]]):(Double, ArrayBuffer[(Double, String, String)]) = {
    // Top words
    var pairs = new ArrayBuffer[(Double, String, String)]

    var avg = 0.0
    var count = 0
    for(s1 <- st1) {
      val v1 = m1.get(s1)
      val v1b = m1b.get(s1)
      // FIXME: right now, is shouldn't ever be the case that a word is in one vocab and not the other, but this
      // should be made much more general!
      if(v1.isDefined) {
        for(s2 <- st2) {
          val v2 = m2.get(s2)
          val v2b = m2b.get(s2)
          if(v2.isDefined) {
            var s = dotProduct(v1.get, v2.get)
            if (v1b.isDefined && v2b.isDefined) {
              val s2 = dotProduct(v1b.get, v2b.get)
              s = (0.5 * s) + (0.5 * s2)
            }
            avg += s
            count += 1

            // Top Words
            pairs.append ( (s, s1, s2) )
          }
        }
      }
    }
    if(count != 0) (avg / count, pairs)
    else (0, pairs)
  }

  def bidirSimComp(t1:Iterable[String], t2:Iterable[String],
                    m1: Map[String, Array[Double]], m2:Map[String, Array[Double]],
                    m1b: Map[String, Array[Double]], m2b:Map[String, Array[Double]]): Seq[(String, Seq[(String, String, Double)])] = {

    val st1 = new ArrayBuffer[String]()
    t1.foreach(st1 += Word2vec.sanitizeWord(_))
    val st2 = new ArrayBuffer[String]()
    t2.foreach(st2 += Word2vec.sanitizeWord(_))
    val sims = sanitizedBidirSimComp(st1, st2, m1, m2, m1b, m2b)

    sims
  }

  def sanitizedBidirSimComp(st1:Iterable[String], st2:Iterable[String],
                            m1: Map[String, Array[Double]], m2:Map[String, Array[Double]],
                            m1b: Map[String, Array[Double]], m2b:Map[String, Array[Double]]):Seq[(String, Seq[(String, String, Double)])] = {
    // Top words
    val questionRows = new ArrayBuffer[(String, Seq[(String, String, Double)])]

    for (s1 <- st1) {
      val currRow = new ArrayBuffer[(String, String, Double)]

      val v1 = m1.get(s1)
      val v1b = m1b.get(s1)
      // FIXME: right now, is shouldn't ever be the case that a word is in one vocab and not the other, but this
      // should be made much more general!
      var maxSim = -99.0
      var maxWord = ""
      var minSim = 100.0
      var minWord = ""
      var sumWords = 0.0
      var nWordsSeen = 0.0
      var avgSim = -99.9
      if (v1.isDefined) {

        for (s2 <- st2) {
          val v2 = m2.get(s2)
          val v2b = m2b.get(s2)
          if (v2.isDefined) {
            var s = dotProduct(v1.get, v2.get)
            if (v1b.isDefined && v2b.isDefined) {
              val score2 = dotProduct(v1b.get, v2b.get)
              s = (0.5 * s) + (0.5 * score2)
              //Max
              if (s > maxSim) {
                maxSim = s
                maxWord = s2
              }
              //Min
              if (s < minSim) {
                minSim = s
                minWord = s2
              }
              //Avg
              sumWords += s
              nWordsSeen += 1.0
            }
          }
        }
        avgSim = sumWords / nWordsSeen
        //max
        currRow.append(("WORD2VEC_REL_BIDIR_causal_mar30_1hop_MAXSIM", maxWord, maxSim))
        //min
        currRow.append(("WORD2VEC_REL_BIDIR_causal_mar30_1hop_MINSIM", minWord, minSim))
        //avg
        currRow.append(("WORD2VEC_REL_BIDIR_causal_mar30_1hop_AVGSIM", "n/a", avgSim))

      }

      questionRows.append((s1, currRow))


    }
    questionRows
  }


  def bidirAllSimilarities(t1:Iterable[String], t2:Iterable[String],
                           m1: Map[String, Array[Double]], m2:Map[String, Array[Double]],
                           m1b: Map[String, Array[Double]], m2b:Map[String, Array[Double]]): Seq[Seq[Double]] = {

    val st1 = new ArrayBuffer[String]()
    t1.foreach(st1 += Word2vec.sanitizeWord(_))
    val st2 = new ArrayBuffer[String]()
    t2.foreach(st2 += Word2vec.sanitizeWord(_))
    val sims = sanitizedBidirAllSimilarities(st1, st2, m1, m2, m1b, m2b)

    sims
  }

  def sanitizedBidirAllSimilarities(st1:Iterable[String], st2:Iterable[String],
                                  m1: Map[String, Array[Double]], m2:Map[String, Array[Double]],
                                  m1b: Map[String, Array[Double]], m2b:Map[String, Array[Double]]):Seq[Seq[Double]] = {
    // Top words
    val questionRows = new ArrayBuffer[Seq[Double]]

    for(s1 <- st1) {
      val currRow = new ArrayBuffer[Double]
      val v1 = m1.get(s1)
      val v1b = m1b.get(s1)
      // FIXME: right now, is shouldn't ever be the case that a word is in one vocab and not the other, but this
      // should be made much more general!
      if(v1.isDefined) {
        for(s2 <- st2) {
          val v2 = m2.get(s2)
          val v2b = m2b.get(s2)
          if(v2.isDefined) {
            var s = dotProduct(v1.get, v2.get)
            if (v1b.isDefined && v2b.isDefined) {
              val s2 = dotProduct(v1b.get, v2b.get)
              s = (0.5 * s) + (0.5 * s2)
            }
            currRow.append (s)
          } else currRow.append(-99.0)
        }
        questionRows.append(currRow.toArray)
      } else questionRows.append(new Array[Double](0))
    }

    questionRows
  }

  /**
    * Finds the maximum word2vec similarity between any two words in these two texts
    * IMPORTANT: IMPORTANT: t1, t2 must be arrays of words, not lemmas!
    */
  def maxSimilarity(t1:Iterable[String], t2:Iterable[String],
                    m1: Map[String, Array[Double]], m2:Map[String, Array[Double]]):Double = {
    val st1 = new ArrayBuffer[String]()
    t1.foreach(st1 += Word2vec.sanitizeWord(_))
    val st2 = new ArrayBuffer[String]()
    t2.foreach(st2 += Word2vec.sanitizeWord(_))
    sanitizedMaxSimilarity(st1, st2, m1, m2)
  }

  def bidirMaxSimilarity(t1:Iterable[String], t2:Iterable[String],
                         m1: Map[String, Array[Double]], m2:Map[String, Array[Double]],
                         m1b: Map[String, Array[Double]], m2b:Map[String, Array[Double]]):Double = {
    val st1 = new ArrayBuffer[String]()
    t1.foreach(st1 += Word2vec.sanitizeWord(_))
    val st2 = new ArrayBuffer[String]()
    t2.foreach(st2 += Word2vec.sanitizeWord(_))
    sanitizedBidirMaxSimilarity(st1, st2, m1, m2, m1b, m2b)
  }

  def minSimilarity(t1: Iterable[String], t2: Iterable[String],
                    m1: Map[String, Array[Double]], m2:Map[String, Array[Double]]): Double = {
    val st1 = t1.map(Word2vec.sanitizeWord(_))
    val st2 = t2.map(Word2vec.sanitizeWord(_))
    sanitizedMinSimilarity(st1, st2, m2, m2)
  }

  def bidirMinSimilarity(t1: Iterable[String], t2: Iterable[String],
                         m1: Map[String, Array[Double]], m2:Map[String, Array[Double]],
                         m1b: Map[String, Array[Double]], m2b:Map[String, Array[Double]]): Double = {
    val st1 = t1.map(Word2vec.sanitizeWord(_))
    val st2 = t2.map(Word2vec.sanitizeWord(_))
    sanitizedBidirMinSimilarity(st1, st2, m2, m2, m1b, m2b)
  }

  /**
    * Finds the maximum word2vec similarity between any two words in these two texts
    * IMPORTANT: words here must already be normalized using Word2vec.sanitizeWord()!
    */
  def sanitizedMaxSimilarity(t1:Iterable[String], t2:Iterable[String],
                             m1: Map[String, Array[Double]], m2:Map[String, Array[Double]]):Double = {
    var max = Double.MinValue
    for(s1 <- t1) {
      val v1 = m1.get(s1)
      if(v1.isDefined) {
        for(s2 <- t2) {
          val v2 = m2.get(s2)
          if(v2.isDefined) {
            val s = dotProduct(v1.get, v2.get)
            if(s > max) max = s
          }
        }
      }
    }
    max
  }

  def sanitizedBidirMaxSimilarity(t1:Iterable[String], t2:Iterable[String],
                                  m1: Map[String, Array[Double]], m2:Map[String, Array[Double]],
                                  m1b: Map[String, Array[Double]], m2b:Map[String, Array[Double]]):Double = {
    var max = Double.MinValue
    for(s1 <- t1) {
      val v1 = m1.get(s1)
      val v1b = m1b.get(s1)
      if(v1.isDefined) {
        for(s2 <- t2) {
          val v2 = m2.get(s2)
          val v2b = m2b.get(s2)
          if(v2.isDefined) {
            var s = dotProduct(v1.get, v2.get)
            if (v1b.isDefined && v2b.isDefined) {
              val s2 = dotProduct(v1b.get, v2b.get)
              s = (0.5 * s) + (0.5 * s2)
            }
            if(s > max) max = s
          }
        }
      }
    }
    max
  }

  /**
    * Finds the minimum word2vec similarity between any two words in these two texts
    * IMPORTANT: words here must already be normalized using Word2vec.sanitizeWord()!
    */
  def sanitizedMinSimilarity(t1:Iterable[String], t2:Iterable[String],
                             m1: Map[String, Array[Double]], m2:Map[String, Array[Double]]):Double = {
    var min = Double.MaxValue
    for(s1 <- t1) {
      val v1 = m1.get(s1)
      if(v1.isDefined) {
        for(s2 <- t2) {
          val v2 = m2.get(s2)
          if(v2.isDefined) {
            val s = dotProduct(v1.get, v2.get)
            if(s < min) min = s
          }
        }
      }
    }
    min
  }

  def sanitizedBidirMinSimilarity(t1:Iterable[String], t2:Iterable[String],
                                  m1: Map[String, Array[Double]], m2:Map[String, Array[Double]],
                                  m1b: Map[String, Array[Double]], m2b:Map[String, Array[Double]]):Double = {
    var min = Double.MaxValue
    for(s1 <- t1) {
      val v1 = m1.get(s1)
      val v1b = m1b.get(s1)
      if(v1.isDefined) {
        for(s2 <- t2) {
          val v2 = m2.get(s2)
          val v2b = m2b.get(s2)
          if(v2.isDefined) {
            var s = dotProduct(v1.get, v2.get)
            if (v1b.isDefined && v2b.isDefined) {
              val s2 = dotProduct(v1b.get, v2b.get)
              s = (0.5 * s) + (0.5 * s2)
            }
            if(s < min) min = s
          }
        }
      }
    }
    min
  }


  /** Adds the content of src to dest, in place */
  private def add(dest:Array[Double], src:Array[Double], dimensions:Int) {
    var i = 0
    while(i < dimensions) {
      dest(i) += src(i)
      i += 1
    }
  }

  private def dotProduct(v1:Array[Double], v2:Array[Double]):Double = {
    assert(v1.length == v2.length) //should we always assume that v2 is longer? perhaps set shorter to length of longer...
    var sum = 0.0
    var i = 0
    while(i < v1.length) {
      sum += v1(i) * v2(i)
      i += 1
    }
    sum
  }

}
