package edu.arizona.sista.qa.linearalg

import java.io._
import edu.arizona.sista.utils.MathUtils
import scala.collection.JavaConversions._

import edu.arizona.sista.jama._

import scala.reflect.ClassTag

/**
 * Created by dfried on 5/24/14.
 */

@SerialVersionUID(2L)
class DenseMatrix(var defaultValue: Double, var M: FloatMatrix) extends Matrix[Double] {

  var sparseRowVecs = new collection.mutable.HashMap[Int, SparseVector[Double]]

  def this(defaultValue: Double) = this(defaultValue, null)

  def this(defaultValue: Double, rows: Int, cols: Int) = {
    this(defaultValue, new FloatMatrix(rows, cols, 0.0f))
  }

  // Accessor methods
  def get(rowIdx: Int, colIdx: Int): Double = {
    if (rowIdx < 0 || rowIdx >= numRows || colIdx < 0 || colIdx >= numCols)
      defaultValue
    else
      M.get(rowIdx, colIdx).toDouble
  }

  def getFloat(rowIdx: Int, colIdx: Int): Float = M.get(rowIdx, colIdx)

  def set(rowIdx: Int, colIdx: Int, value: Float): Unit = M.set(rowIdx, colIdx, value)

  def set(rowIdx: Int, colIdx: Int, value: Double): Unit = M.set(rowIdx, colIdx, value.toFloat)

  def scalarMultiply(factor: Double): Unit = scalarMultiply(factor.toFloat)

  def numRows: Int = M.getRowDimension

  def numCols: Int = M.getColumnDimension

  def setRowVec(row: Int, vector: Array[Float]) {
    require(vector.size == numCols, s"vector size (${vector.size}) must match numCols ($numCols)")
    M.getArray(row) = vector
  }

  override def getRowVec(row: Int)(implicit tag: ClassTag[Double]): Array[Double] = accessRow(row).map(_.toDouble)

  def save(filenamePrefix: String) = {
    val fileOut = new FileOutputStream(filenamePrefix + ".DenseMatrix.dat")
    val out = new ObjectOutputStream(fileOut)
    out.writeObject(M)
    out.close
    fileOut.close
  }

  def load(filenamePrefix: String) {
    val fileIn = new FileInputStream(filenamePrefix + ".denseMatrix.dat")
    val in  = new ObjectInputStream(fileIn)
    M = in.readObject.asInstanceOf[FloatMatrix]
    in.close
    fileIn.close
  }

  // for override purposes
  def load(filenamePrefix: String, ignoreMe: String => Double) {
    load(filenamePrefix)
  }

  // direct memory access to the row of probabilities, not a view!
  def accessRow(row: Int): Array[Float] = M.getArray(row)

  // Multiplication
  /*
  def *(in: Matrix): Matrix = in match {
    case in:DenseMatrix => this * in
    case _ => throw new Exception("can currently only multiply dense by dense")
  }
  */

  def *(in:Matrix[Double]):Matrix[Double] = {
    in match {
      case m:DenseMatrix => *(in.asInstanceOf[DenseMatrix])
      case _ => throw new RuntimeException ("DenseMatrix.*: Cannot multiply sparse matrix with non-sparse matrix")
    }
  }

  def *(in:DenseMatrix): DenseMatrix = new DenseMatrix(this.defaultValue, this.M parallelTimes in.M)

  def findMinMax: (Double, Double) = {
    var minValue = Double.PositiveInfinity
    var maxValue = Double.NegativeInfinity

    for (r <- 0 until numRows) {
      for (c <- 0 until numCols) {
        val v = get(r, c)
        if (v > maxValue) maxValue = v.toDouble
        if (v < minValue) minValue = v.toDouble
      }
    }

    (minValue, maxValue)
  }

  def scalarMultiply(factor: Float): Unit = {
    M timesEquals(factor)
  }

  // prune within each row to take the topN values (destructive)
  def prune(topN: Int): Unit = throw new Exception("prune not implemented for dense matrix")

  // prune within each row to take the values corresponding to topN w2v scores(destructive)
  def pruneW2VTopN(rowIndex:Int, indicesToKeep:Array[Int]): Unit = throw new Exception("pruneW@VTopN not implemented for dense matrix")


  def normalizeWithinRow: Unit = {
    val A = M.getArray
    for (r <- 0 until numRows) {
      val sum: Float = A(r).sum
      if (sum != 0.0f) for (c <- 0 until numCols) A(r)(c) /= sum
    }
  }

  def scaleRow(rowIdx: Int, factor: Double): Unit = {
    val r = accessRow(rowIdx)
    val ff = factor.toFloat
    for (c <- 0 until r.size) {
      r(c) *= ff
    }
  }

  /*
  def addRow(rowIdx: Int, values: Array[Double]): Unit = {
    val r = accessRow(rowIdx)
    require(r.size == values.size, "DenseMatrix: row in matrix and row to add must be same size!")
    for (c <- 0 until r.size) {
      r(c) += values(c).toFloat
    }
  }
  */

  def populateRowFromDenseArray(rowIdx: Int, values: Array[Double]): Unit = {
    M.getArray(rowIdx) = values.map(_.toFloat)
  }

  def softmaxRow(rowIdx: Int): Unit = {
    M.getArray(rowIdx) = MathUtils.denseSoftmaxFloat(M.getArray(rowIdx))
    // M.getArray(rowIdx) = MathUtils.softmax(M.getArray(rowIdx)).toArray
  }

  // return indices, values for non-zero values in the row
  def getSparseRowVec(row: Int) = this.synchronized({
    if (sparseRowVecs == null)
      sparseRowVecs = new collection.mutable.HashMap[Int, SparseVector[Double]]
    sparseRowVecs.get(row).getOrElse {
      val (values, indices) = accessRow(row).zipWithIndex.filter(_._1 != defaultValue).unzip
      val v = new SparseVector[Double](defaultValue, indices.toArray, values.map(_.toDouble).toArray)
      sparseRowVecs.put(row, v)
      v
    }
  })

  def toSparse:SparseMatrix[Double] = {
    val rows = for  {
      ix <- (0 until numRows)
    } yield getSparseRowVec(ix)
    new SparseMatrix[Double](defaultValue.toDouble, rows.to[collection.mutable.ArrayBuffer])
  }
}

object DenseMatrix {
  def load(filenamePrefix: String, defaultValue: Double = 0.0): DenseMatrix = {
    val dm = new DenseMatrix(defaultValue, null)
    dm.load(filenamePrefix)
    dm
  }

  def main(args: Array[String]) = {
    // load a matrix and print the size
    val filenamePrefix = args(0)
    println(s"loading matrix w/ prefix $filenamePrefix")
    val dm = load(filenamePrefix)
    println(s"rows: ${dm.numRows}")
    println(s"cols: ${dm.numCols}")
  }

  def klDivergence(xs: Array[Double], ys: Array[Double]): Double = {
    var sum = 0.0
    for ((x, y) <- xs zip ys) {
      if (x != 0 && y != 0)
        sum += x * (math.log(x) - math.log(y))
    }
    sum / math.log(2)
  }

  def jsDivergence(xs: Array[Double], ys: Array[Double]): Double = {
    var left = 0.0
    var right = 0.0
    for ((x, y) <- xs zip ys) {
      val m = (x + y) / 2
      if (x != 0 && m != 0)
        left += x * (math.log(x) - math.log(m))
      if (y != 0 && m != 0)
        right += y * (math.log(y) - math.log(m))
    }
    (left + right) / (2 * math.log(2))
  }

  def jsDistance(xs: Array[Double], ys: Array[Double]) = math.sqrt(jsDivergence(xs, ys))
}
