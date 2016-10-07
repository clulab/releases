package edu.arizona.sista.qa.linearalg

//import edu.arizona.sista.mc.tessellatedgraph.structures.Schema
import edu.arizona.sista.struct.Lexicon

import collection.mutable.ArrayBuffer
import java.io._
import org.slf4j.LoggerFactory
import scala.util.control.Breaks._
import SparseMatrix.logger
import edu.arizona.sista.utils.Profiler
import edu.arizona.sista.utils.MathUtils
import edu.arizona.sista.qa.word2vec.Word2vec
import Numeric.Implicits._
import scala.reflect.ClassTag

/**
 * A memory-efficient storage class for sparse matrices.  Optimized for fast matrix multiplication of very large sparse matricies
 * User: peter
 * Date: 2/4/14
 */

@SerialVersionUID(2L)
class SparseMatrix[B](var defaultValue:B, var rowVecs: ArrayBuffer[SparseVector[B]])(implicit num: Fractional[B],
                                                                                     tag: ClassTag[B]) extends Matrix[B] {

  def this(defaultValue:B)(implicit num: Fractional[B], tag: ClassTag[B]) = {
    this(defaultValue, new ArrayBuffer[SparseVector[B]])
  }

  var maxColumnAccessed:Int = 0

  def set(rowIdx:Int, colIdx:Int, value:B) {
    doExpansion(rowIdx)

    // Bound checking
    if ((colIdx < 0) || (rowIdx < 0)) throw new RuntimeException("* SparseMatrix.set(): colIdx/rowIdx cannot be less than 0")

    // Keep track of the largest column index that's been accessed (for rough size calculations)
    if (colIdx > maxColumnAccessed) maxColumnAccessed = colIdx

    rowVecs(rowIdx).set(colIdx, value)

  }

  override def get(rowIdx:Int, colIdx:Int):B = {
    // Bound checking
    if ((colIdx < 0) || (rowIdx < 0)) throw new RuntimeException("* SparseMatrix.get(): colIdx/rowIdx cannot be less than 0")
    if (rowIdx >= numRows) return defaultValue

    rowVecs(rowIdx).get(colIdx)

  }

  // Checks to see if there are enough rows -- if there aren't, it pushes new rows with empty columns on
  def doExpansion(row:Int) {
    // TODO: expand exponentially like ArrayBuffer
    if (row >= rowVecs.size) {
      for (i <- 0 until ((row - rowVecs.size) + 1) ) {
        rowVecs.append(new SparseVector[B](defaultValue))
      }
    }
  }


  /*
   * Import functions
   */

  def populateRowFromDenseArray(rowIdx:Int, denseIn:Array[B]) {
    doExpansion(rowIdx)
    rowVecs(rowIdx).populateFromDenseArray(denseIn)
  }

  /*
  * Helper functions
  */
  def numRows:Int = rowVecs.size
  def numCols:Int = maxColumnAccessed + 1

  // Sparsity (0 empty, 1 full)
  def sparsity:Double = {
    var sizeSum:Int = 0
    for (row <- rowVecs) {
      sizeSum += row.labels.size
    }
    val sizesq:Double = numRows.toDouble * numCols.toDouble     // the square is very large
    (sizeSum.toDouble / sizesq)
  }


  def setMatrixSize(rows:Int, cols:Int) {
    // Ensure matrix is seen as proper size
    doExpansion(rows-1)
    maxColumnAccessed = cols-1
  }


  // Returns a row from the sparse matrix as a sparse vector.  This is a much less expensive operation than
  // returning a column, and is used (in concert with transpose) for sparse multiplication.
  def getRowVecSparse(row:Int):SparseVector[B] = {
    if (row < 0) throw new RuntimeException("* SparseMatrix.getRowVecSparse(): rowIdx cannot be less than 0")
    if (row >= rowVecs.size) return new SparseVector[B](defaultValue)
    rowVecs(row)
  }

  // Keeps only the top N weights in a given row.  Useful for pruning small weights, and reducing the computational load of matrix multiplication.
  def prune(topN:Int) {
    for (row <- rowVecs) {
      row.pruneTopN(topN)
    }
  }

  // Keeps only the specified elements in the specified row.  Used with pruning based on external scores.
  def pruneW2VTopN(rowIndex:Int, indicesToKeep:Array[Int]) {
    rowVecs(rowIndex).pruneIndices(indicesToKeep)
  }

  // Prunes each row by looking for a gap in values, backoff to below a ratio of the highest value
  def pruneBinning(gapRatio:Double, cutoffRatio:Double) {
    for (row <- rowVecs) {
      row.pruneGap(gapRatio, cutoffRatio)
    }
  }

  // Normalizes a row, such that all the values in the row sum to one
  def normalizeWithinRow() {
    for (row <- rowVecs) {
      row.sumToOne()
    }
  }

  def clearRow(idx:Int) {
    if (idx < 0) throw new RuntimeException("* SparseMatrix.clearRow(): rowIdx cannot be less than 0")
    if (idx >= rowVecs.size) return
    rowVecs(idx) = new SparseVector[B](defaultValue)
  }

  def clearCol(idx:Int) {
    if (idx < 0) throw new RuntimeException("* SparseMatrix.clearCol(): colIdx cannot be less than 0")
    for (i <- 0 until numRows) {
      set(i, idx, defaultValue)
    }
  }

  // debug
  def display() {
    println ("Display Matrix:")
    println ("numRows: " + numRows + "\tnumColumns: " + numCols)
    for (row <- 0 until rowVecs.size) {
      println ("Row (" + row + "): \t" + rowVecs(row).toString)
    }
  }


  /*
   * Mathematical functions
   */

  def *(in:Matrix[B]):Matrix[B] = {
    in match {
      case m:SparseMatrix[B] => *(in.asInstanceOf[SparseMatrix[B]])
      case _ => throw new RuntimeException ("SparseMatrix.*: Cannot multiply sparse matrix with non-sparse matrix")
    }
  }

  def *(in:SparseMatrix[B]):SparseMatrix[B] = {
    val out = new SparseMatrix[B](this.defaultValue)

    println ("1st matrix size: " + this.numRows + " x " + this.numCols)
    println ("2nd matrix size: " + in.numRows + " x " +  in.numCols)
    if (this.numCols != in.numRows) throw new RuntimeException("* SparseMatrix.*(): cannot multiply matrices -- number of columns in first matrix must equal number of rows in second matrix")

    // Create transpose of second matrix
    logger.debug (" Generating transpose... ")
    val trans:SparseMatrix[B] = in.transpose

    for (row <- 0 until this.numRows) {
      if(row % 100 == 0){
        logger.debug("Multiplication: At Row " + row + " of " + this.numRows)
      }

      // Get row vector from THIS
      val rowVec = this.getRowVecSparse(row)

      for (col <- 0 until in.numCols) {
        // Get column vector from IN
        val colVec = trans.getRowVecSparse(col)

        // Perform dot product
        val result = rowVec.dot(colVec)

        // Store result
        if (result != defaultValue) {
          out.set(row, col, result)
        }
      }
    }

    // Ensure matrix is seen as proper size
    out.doExpansion(this.numRows - 1)
    out.maxColumnAccessed = in.numCols - 1

    out
  }


  // Scalar multiplication with every value in the array, including the defaultValue
  def scalarMultiply(factor:B) {
    defaultValue = num.times(defaultValue, factor)
    for (rowVec <- rowVecs) {
      rowVec.scalarMultiply(factor)
    }
  }

  def transpose:SparseMatrix[B] = {
    val out = new SparseMatrix(defaultValue)
    out.setMatrixSize(numCols, numRows)

    for (i <- 0 until numRows) {
      val rowVec = rowVecs(i)
      for (j <- 0 until rowVec.labels.size) {
        out.set(rowVec.labels(j), i, rowVec.values(j))
      }
    }

    out
  }

  /*
   * Load/Save methods (plain text)
   */
  def save(filenamePrefix:String) {
    val filename = filenamePrefix + ".matrix"
    // Saves sparse matrix as text file
    logger.debug ("* save: Started... (" + filename + ")")
    val pw = new PrintWriter(filename)
    // Save size and default value
    pw.println (numRows + " " + numCols + " " + defaultValue)

    // Save matrix
    for (row <- 0 until rowVecs.size) {
      val columnLabels = rowVecs(row).labels
      val columnValues = rowVecs(row).values
      for (colIdx <- 0 until columnLabels.size) {
        // Save as row <space> col <space> value
        pw.println (row + " " + columnLabels(colIdx) + " " + columnValues(colIdx))
      }
    }
    pw.close()

    logger.debug ("* save: Completed... (" + filename + ")")
  }

  def load(filenamePrefix:String, parseFn: String => B) {
    // Loads sparse matrix from text file, saved with 'save' method above
    val filename = filenamePrefix + ".matrix"
    val source = scala.io.Source.fromFile(filename)
    val lines = source.getLines()
    var numWeights:Int = 0

    logger.debug ("* load: Started... (" + filename + ")")

    // First line is <numRows> <numCols> <defaultValue>
    val header = lines.next().split(" ")
    setMatrixSize(header(0).toInt, header(1).toInt)
    defaultValue = parseFn(header(2))

    // Read each line of matrix data
    while (lines.hasNext) {
      val oneline = lines.next()

      // Lines are space delimited
      val split = oneline.split(" ")

      // Ensure that we have data from exactly 3 fields
      if (split.size == 3) {
        // Extract 3 fields
        val rowIdx = split(0).toInt       // Field 1: Row Index
        val colIdx = split(1).toInt       // Field 2: Column Index
        val value =  parseFn(split(2))    // Field 3: Value at (row, column)

        set(rowIdx, colIdx, value)
        numWeights += 1
      }
    }

    logger.info ("Number of values processed: " + numWeights)
    logger.info ("Sparsity of matrix: " + sparsity.formatted("%3.10f") + "%" )

    logger.debug ("* load: Completed... (" + filename + ")")
    source.close()
  }

  // Save as binary
  def saveTo(filenamePrefix:String): Unit = {
    // Step 1: Save schemas
    val os = new ObjectOutputStream(new FileOutputStream(filenamePrefix + ".matrixbin"))
    os.writeObject(this)
    os.close()
  }


  // return dense matrix and boolean mask specifying which rows are empty
  // converts all values to double precision
  def toDense(rows: Option[Int] = None, cols: Option[Int] = None): (DenseMatrix, Array[Boolean]) = {
    val mat = new DenseMatrix(defaultValue.toDouble, rows.getOrElse(rowVecs.size), cols.getOrElse(maxColumnAccessed + 1))
    val rowEmpty = new Array[Boolean](rowVecs.size)
    for (rowIdx <- 0 until rowVecs.size) {
      val row = rowVecs(rowIdx)
      rowEmpty(rowIdx) = row.labels.isEmpty
      for ((colIdx, value) <- row) {
        mat.set(rowIdx, colIdx, value.toDouble)
      }
    }
    (mat, rowEmpty)
  }

  def scaleRow(rowIdx: Int, factor: B): Unit = {
    rowVecs(rowIdx).scalarMultiply(factor)
  }

  def getSparseRowVec(row:Int): SparseVector[B] = {
    rowVecs(row)
  }

  def convertToDouble: SparseMatrix[Double] = {
    val rows = rowVecs.map(_.mapValues(_.toDouble))
    new SparseMatrix[Double](defaultValue.toDouble, rows)
  }

  def convertToFloat: SparseMatrix[Float] = {
    val rows = rowVecs.map(_.mapValues(_.toFloat))
    new SparseMatrix[Float](defaultValue.toFloat, rows)
  }
}

object SparseMatrix {
  val logger = LoggerFactory.getLogger(classOf[SparseMatrix[Double]])
  def loadDouble(filenamePrefix: String) = {
    val sm = new SparseMatrix[Double](0.0)
    sm.load(filenamePrefix, _.toDouble)
    sm
  }
  def loadFloat(filenamePrefix: String) = {
    val sm = new SparseMatrix[Float](0.0f)
    sm.load(filenamePrefix, _.toFloat)
    sm
  }

  // Load binary
  def loadDoubleFrom(filenamePrefix:String):SparseMatrix[Double] = {
    // Step 1: Load schemas
    val is = new ObjectInputStream(new FileInputStream(filenamePrefix + ".matrixbin"))
    val c = is.readObject().asInstanceOf[SparseMatrix[Double]]
    is.close()
    c
  }

}
