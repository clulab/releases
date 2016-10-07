package edu.arizona.sista.qa.linearalg

import scala.Array
import collection.mutable.ArrayBuffer
import java.util
import Numeric.Implicits._
import scala.reflect.ClassTag

/**
 * A memory-efficient storage class for sparse vectors
 * User: peter
 * Date: 2/4/14
 */

@SerialVersionUID(2L)
class SparseVector[B](val defaultValue:B,
                      var labels: Array[Int],
                      var values: Array[B])
                              (implicit num: Fractional[B], tag: ClassTag[B]) extends Traversable[(Int,
  B)] with Serializable{

  def this(defaultValue:B)(implicit num: Fractional[B], tag: ClassTag[B]) =
    this(defaultValue, new Array[Int](0), new Array[B](0))

  def labelToIdx(label:Int):Option[Int] = {
    val ix = util.Arrays.binarySearch(labels, label)
    // ix is the insertion index, where the label should be inserted to maintain a sorted array
    if (ix >= 0)
      Some(ix)
    else
      None
  }

  def set(label:Int, value:B) {
    // Bound checking
    if (label < 0) throw new RuntimeException("* SparseVector.set(): Index label cannot be less than 0")

    val labelIdx = labelToIdx(label)
    if (value == defaultValue) {
      // Check if we're overwriting an existing value
      if (! labelIdx.isDefined) return
      // Remove existing value
      val l = new Array[Int](labels.size - 1)
      val v = new Array[B](values.size - 1)
      if (labelIdx.get > 0) {
        System.arraycopy(labels, 0, l, 0, labelIdx.get)
        System.arraycopy(values, 0, v, 0, labelIdx.get)
      }
      if (labels.size - labelIdx.get > 1) {
        System.arraycopy(labels, labelIdx.get + 1, l, labelIdx.get, labels.size - labelIdx.get - 1)
        System.arraycopy(values, labelIdx.get + 1, v, labelIdx.get, values.size - labelIdx.get - 1)
      }
      labels = l
      values = v
      return
    }

    // Case 1: Empty case
    if (labels.size == 0) {
      labels = Array(label)
      values = Array(value)
    }

    // Case 2: Non-empty case: Locate the value if it already exists, or if it's between two existing indicies

    // i will be either the location or, if not present, (- (insertion point) - 1)
    val ix = util.Arrays.binarySearch(labels, label)
    val ip = - (ix + 1)
    if (ix >= 0) {
      values(ix) = value                 // swap in new value
    } else {
      // Case 2A: ip is the insertion index, so splice it in
      val l = new Array[Int](labels.size + 1)
      System.arraycopy(labels, 0, l, 0, ip)
      System.arraycopy(labels, ip, l, ip + 1, labels.size - ip)
      l(ip) = label
      labels = l
      val v = new Array[B](values.size + 1)
      System.arraycopy(values, 0, v, 0, ip)
      System.arraycopy(values, ip, v, ip + 1, values.size - ip)
      v(ip) = value
      values = v
    }
  }

  def get(label:Int):B = {
    // Bound checking
    if (label < 0) throw new RuntimeException("* SparseVector.get(): Index cannot be less than 0")
    labelToIdx(label) match {
      case Some(idx) => values(idx)
      case None => defaultValue
    }
  }

  def clear() {
    labels = Array.empty[Int]
    values = Array.empty[B]
  }

  /*
   * Import function
   */

  def populateFromDenseArray(denseIn:Array[B]) {
    val newLabels = new ArrayBuffer[Int]
    val newValues = new ArrayBuffer[B]

    for (i <- 0 until denseIn.size) {
      if ( denseIn(i) != defaultValue ) {
        newLabels.append( i )
        newValues.append( denseIn(i) )
      }
    }

    labels = newLabels.toArray
    values = newValues.toArray
  }


  /*
   * Helper functions
   */

  def display() {
    println ("Display Vector:")
    for (idx <- 0 until labels.size) {
      print ("(" + labels(idx) + ", " + values(idx) + ") \t")
    }
    println ("")
  }

  override def toString:String = mkString("\t")

  // take the top threshold percentage of the values, by mass
  def pruneMass(threshold: Double) = {
    import num._
    assert(threshold > 0 && threshold <= 1.0)
    val sorted: Array[(Int, B)] = (labels zip values).sortBy(- _._2)
    // calculate the cumulative sum of the sorted probability values
    val cmf: Array[Double] = sorted.map(_._2).scanLeft(0.0)(_ + _.toDouble).tail
    val cutoff: Double = cmf.last * threshold
    // traverse up until the cutoff
    clear()
    for ((c, i) <- cmf.takeWhile(_ <= cutoff).zipWithIndex) {
      set(sorted(i)._1, sorted(i)._2)
    }
  }

  //keep only values with specified indices
  def pruneIndices(indicesToKeep:Array[Int]){

    // If there are fewer than topN values stored in the array, then it already meets the postcondition and there's no work to do
    if (labels.size <= indicesToKeep.size) return

    //temporarily store the values being kept, in same order as the Array of indices
    val tempRowValues = new ArrayBuffer[B]
    for (index <- indicesToKeep) {
      tempRowValues += values(labels.indexOf(index))
    }

    // Clear this vector
    clear()

    // Repopulate with only the kept values
    for (i <- 0 until indicesToKeep.size) {
      set(indicesToKeep(i), tempRowValues(i))
    }

  }

  // Prunes all but the top N values. Useful for reducing the load in multiplication when there are many small weights.
  def pruneTopN(topN:Int) {
    import num._
    // If there are fewer than topN values stored in the array, then it already meets the postcondition and there's no work to do
    if (labels.size < topN) return

    // Sort the values
    val unsorted = new Array[(Int, B)](labels.size)
    for (i <- 0 until labels.size) {
      unsorted(i) = (labels(i), values(i))
    }
    val sorted = unsorted.sortBy(-_._2.toDouble)

    // Clear this vector
    clear()

    // Repopulate with only the highest N values
    for (i <- 0 until topN) {
      set(sorted(i)._1, sorted(i)._2)
    }
  }

  def pruneGap(gapRatio: Double, cutoffRatio: Double) {
    import num._
    // Sort the values
    val unsorted = new Array[(Int, B)](labels.size)
    for (i <- 0 until labels.size) {
      unsorted(i) = (labels(i), values(i))
    }
    val sorted = unsorted.sortBy(-_._2.toDouble)

    // Clear this vector
    clear()

    // Repopulate with only the highest N values
    val topValue = sorted(0)._2.toDouble
    val cutoff = topValue * cutoffRatio
    val maxGap = topValue * gapRatio
    var prevValue = topValue
    for ((l, v) <- sorted) {
      while (v.toDouble >= cutoff) {
        if (prevValue - v.toDouble > maxGap) {
          set(l, v)
          prevValue = v.toDouble
        }
      }
    }
  }

  def zipWithUnion[A](fn: (B, B) => A)(that:SparseVector[B])(implicit num: Fractional[A],
                                                                      tag: ClassTag[A]) = {
    val N = this.size + that.size
    val labels = new Array[Int](N)
    val values = new Array[A](N)

    var iThis:Int = 0
    var iThat:Int = 0
    var iNew:Int = 0

    while ((iThis < this.size) && (iThat < that.size)) {
      if (this.labels(iThis) == that.labels(iThat)) {
        labels(iNew) = this.labels(iThis)
        values(iNew) = fn(this.values(iThis), that.values(iThat))
        iThis += 1
        iThat += 1
        iNew += 1
      } else if (this.labels(iThis) < that.labels(iThat)) {
        labels(iNew) = this.labels(iThis)
        values(iNew) = fn(this.values(iThis), that.defaultValue)
        iThis += 1
        iNew += 1
      } else if (this.labels(iThis) > that.labels(iThat)) {
        labels(iNew) = that.labels(iThat)
        values(iNew) = fn(this.defaultValue, that.values(iThat))
        iThat += 1
        iNew += 1
      }
    }

    while (iThis < this.size) {
      labels(iNew) = this.labels(iThis)
      values(iNew) = fn(this.values(iThis), that.defaultValue)
      iThis += 1
      iNew += 1
    }

    while (iThat < that.size) {
      labels(iNew) = that.labels(iThat)
      values(iNew) = fn(this.defaultValue, that.values(iThat))
      iThat += 1
      iNew += 1
    }

    new SparseVector[A](fn(this.defaultValue, that.defaultValue), labels.slice(0, iNew), values.slice(0, iNew))
  }

  def zipWithIntersection[A](fn: (B, B) => A)(that:SparseVector[B])(implicit num: Fractional[A], tag: ClassTag[A]) = {
    val N = this.size + that.size
    val labels = new Array[Int](N)
    val values = new Array[A](N)

    var iThis:Int = 0
    var iThat:Int = 0
    var iNew:Int = 0

    while ((iThis < this.size) && (iThat < that.size)) {
      if (this.labels(iThis) == that.labels(iThat)) {
        labels(iNew) = this.labels(iThis)
        values(iNew) = fn(this.values(iThis), that.values(iThat))
        iThis += 1
        iThat += 1
        iNew += 1
      } else if (this.labels(iThis) < that.labels(iThat)) {
        iThis += 1
      } else if (this.labels(iThis) > that.labels(iThat)) {
        iThat += 1
      }
    }

    new SparseVector(fn(this.defaultValue, that.defaultValue), labels.slice(0, iNew), values.slice(0, iNew))
  }

  /**
   * Kullback-Leibler divergence, log base 2. This and other should be probability distributions
   */
  def klDivergence(other: SparseVector[B]): Double = {
    import num._
    zipWithIntersection((p, q) => math.log(p.toDouble / q.toDouble) * p.toDouble)(other).sum / math.log(2)
  }

  /**
   * Jensen-Shannon divergence, log base 2, a symmetric (and finite) form of Kullback-Leibler. This and other should be
   * probability distributions. Bounded between 0 and 1.
   */
  def jsDivergence(other: SparseVector[B]): Double = {
    val M = (this + other) / num.fromInt(2)
    math.max((this.klDivergence(M) + other.klDivergence(M) ) / 2, 0) // avoid numerical instability
  }

  def jsDistance(other: SparseVector[B]): Double = math.sqrt(jsDivergence(other))

  /*
   * Mathematical functions
   */
  def dot(in:SparseVector[B]):B = {
    import num._
    var out:B = num.zero
    var idxThis:Int = 0
    var idxIn:Int = 0

    // Perform sparse dot product
    while ((idxThis < this.labels.size) && (idxIn < in.labels.size)) {
      if (this.labels(idxThis) == in.labels(idxIn)) {
        out += this.values(idxThis) * in.values(idxIn)
        idxThis += 1
        idxIn += 1
      } else if (this.labels(idxThis) < in.labels(idxIn)) {
        idxThis += 1
      } else if (this.labels(idxThis) > in.labels(idxIn)) {
        idxIn += 1
      }
    }

    out
  }

  def scalarMultiply(factor:B) {
    for (i <- 0 until values.size) {
      values(i) *= factor
    }
  }

  // Scales the vector such that the sum of all non-zero elements is one (excluding the defaultValue)
  def sumToOne() {
    if (values.size == 0) return
    val sum = this.sum
    if (sum != 0.0) {
      scalarMultiply(num.div(num.fromInt(1), sum))
    }
  }

  // Sum all non-default values of the vector
  def sum:B = values.sum

  // Compares two SparseVectors, and returns a new SparseVector with non-defaultValue values on the dimensions where both vectors contained the same value
  def intersection(in:SparseVector[B]):SparseVector[B] = {
    val out = new SparseVector[B](defaultValue)
    for (i <- 0 until labels.size) {
      val label = labels(i)
      if (in.get(label) == values(i)) {
        out.set(label, values(i))
      }
    }
    out
  }



  def foreach[U](f: ((Int, B)) => U): Unit = {
    (labels zip values).foreach[U](f)
  }

  override def size: Int = labels.size

  def +(that:SparseVector[B]) = {
    require(this.defaultValue == 0.0 && that.defaultValue == 0.0, "vectors must have default value of 0")
    this.zipWithUnion(_ + _)(that)
  }

  def -(that:SparseVector[B]) = {
    require(this.defaultValue == 0.0 && that.defaultValue == 0.0, "vectors must have default value of 0")
    this.zipWithUnion(_ - _)(that)
  }

  def mapValues[A](fn: B => A)(implicit num: Fractional[A], tag: ClassTag[A]) =
    new SparseVector[A](fn(defaultValue), labels, values map fn)

  def map(fn: ((Int, B)) => B): SparseVector[B] =
    new SparseVector[B](defaultValue, labels, (labels zip values) map fn)

  def *(x:B) = mapValues(_ * x)

  def /(x:B) = {
    import num._
    mapValues(_ / x)
  }

}
