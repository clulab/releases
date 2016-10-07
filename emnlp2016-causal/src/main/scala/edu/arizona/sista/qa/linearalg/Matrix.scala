package edu.arizona.sista.qa.linearalg

import scala.reflect.ClassTag

/**
 * Trait for a generic (sparse or dense) matrix, as well as basic matrix operations
 * User: peter
 * Date: 2/4/14
 */
trait Matrix[B] extends Serializable {

  // Accessor methods
  def get(rowIdx:Int, colIdx:Int):B
  def set(rowIdx:Int, colIdx:Int, value:B)

  def numRows:Int
  def numCols:Int

  def save(filenamePrefix: String): Unit
  def load(filenamePrefix: String, parseFn: String => B): Unit


  /*
   * Basic mathematical operations
   */

  // In-place Addition
  def +=(in:Matrix[B])(implicit num: Fractional[B]) {
    import num._
    // Determine maximum size
    var maxRows = this.numRows
    if (in.numRows > maxRows) maxRows = in.numRows
    var maxCols = this.numCols
    if (in.numCols > maxCols) maxCols = in.numCols

    // Perform addition
    for (i <- 0 until maxRows) {
      for (j <- 0 until maxCols) {
        this.set(i, j, this.get(i, j) + in.get(i, j) )
      }
    }
  }


  // linear algebra multiplication, not destructive
  def *(in:Matrix[B]):Matrix[B]

  // prune within each row to take the topN values (destructive)
  def prune(topN: Int): Unit

  // prune within each row to take the values corresponding to topN w2v scores (destructive)
  def pruneW2VTopN(rowIndex:Int, indicesToKeep:Array[Int]): Unit

  def dot(v1:Array[B], v2:Array[B])(implicit num: Fractional[B]):B = {
    import num._
    var out:B = num.zero
    if (v1.size != v2.size) throw new RuntimeException("* Matrix.dot(): dot product requires vectors to have the same length. (v1.size = " + v1.size + ", v2.size = " + v2.size + ")")

    for (i <- 0 until v1.size) {
      out += v1(i) * v2(i)
    }

    out
  }


  /*
   * Helper methods
   */
  def getRowVec(row:Int)(implicit tag: ClassTag[B]):Array[B] = {
    // Note: Returns dense array
    val out = new Array[B](numCols)
    for (i <- 0 until numCols) {
      out(i) = get(row, i)
    }
    out
  }

  // return sparse vector of the non-default values in the row
  def getSparseRowVec(row:Int): SparseVector[B]

  def getColVec(col:Int)(implicit tag: ClassTag[B]):Array[B] = {
    // Note: Returns dense array
    val out = new Array[B](numRows)
    for (i <- 0 until numRows) {
      out(i) = get(i, col)
    }
    out
  }

  def scalarMultiply(factor:B)

  def normalizeWithinRow: Unit

  def scaleRow(rowIdx: Int, factor: B): Unit

  def populateRowFromDenseArray(rowIdx: Int, values: Array[B]): Unit

}
