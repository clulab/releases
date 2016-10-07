package edu.arizona.sista.jama

import java.io.PrintWriter
import java.io.Serializable
import java.text.DecimalFormat
import java.text.DecimalFormatSymbols
import java.text.NumberFormat
import java.util.Arrays
import java.util.Locale

/**
 * Matrix which stores values only with double precision. This saves ~50% memory
 * space in comparison to double precision.
 *
 * @author Nepomuk Seiler
 * @since 1.1.0
 *
 */
object FloatMatrix {
  /**
   * Construct a matrix from a copy of a 2-D array.
   *
   * @param A Two-dimensional array of floats.
   * @exception IllegalArgumentException All rows must have the same length
   */
  def constructWithCopy(A: Array[Array[Float]]): FloatMatrix = {
    val m: Int = A.length
    val n: Int = A(0).length
    val X: FloatMatrix = new FloatMatrix(m, n)
    val C: Array[Array[Float]] = X.getArray
      for (i <- 0 until m) {
        if (A(i).length != n) {
          throw new IllegalArgumentException("All rows must have the same length.")
        }
        for (j <- 0 until n) {
          C(i)(j) = A(i)(j)
        }
      }
    return X
  }

  /**
   * Generate matrix with random elements
   *
   * @param m Number of rows.
   * @param n Number of colums.
   * @return An m-by-n matrix with uniformly distributed random elements.
   */
  def random(m: Int, n: Int): FloatMatrix = {
    val A: FloatMatrix = new FloatMatrix(m, n)
    val X: Array[Array[Float]] = A.getArray
    for (i <- 0 until m) {
      for (j <- 0 until n) {
          X(i)(j) = Math.random.asInstanceOf[Float]
      }
    }
    return A
  }

  /**
   * Generate identity matrix
   *
   * @param m Number of rows.
   * @param n Number of colums.
   * @return An m-by-n matrix with ones on the diagonal and zeros elsewhere.
   */
  def identity(m: Int, n: Int): FloatMatrix = {
    val A: FloatMatrix = new FloatMatrix(m, n)
    val X: Array[Array[Float]] = A.getArray
    for (i <- 0 until m) {
      for (j <- 0 until n) {
          X(i)(j) = (if (i == j) 1.0f else 0.0f)
      }
    }
    return A
  }

  private final val serialVersionUID: Long = 5318241614427787535L
}

class FloatMatrix extends Cloneable with Serializable {
  /**
   * Construct an m-by-n matrix of zeros.
   *
   * @param m Number of rows.
   * @param n Number of colums.
   */
  def this(m: Int, n: Int) {
    this()
    this.m = m
    this.n = n
    A = Array.ofDim[Float](m, n)
  }

  /**
   * Construct an m-by-n constant matrix.
   *
   * @param m Number of rows.
   * @param n Number of colums.
   * @param s Fill the matrix with this scalar value.
   */
  def this(m: Int, n: Int, s: Float) {
    this()
    this.m = m
    this.n = n
    A = Array.ofDim[Float](m, n)
      for (i <- 0 until m) {
        for (j <- 0 until n) {
          A(i)(j) = s
        }
      }
  }

  /**
   * Construct a matrix from a 2-D array.
   *
   * @param A Two-dimensional array of floats.
   * @exception IllegalArgumentException All rows must have the same length
   * @see #constructWithCopy
   */
  def this(A: Array[Array[Float]]) {
    this()
    m = A.length
    n = A(0).length
      for (i <- 0 until m) {
        if (A(i).length != n) {
          throw new IllegalArgumentException("All rows must have the same length.")
        }
      }
    this.A = A
  }

  /**
   * Construct a matrix quickly without checking arguments.
   *
   * @param A Two-dimensional array of floats.
   * @param m Number of rows.
   * @param n Number of colums.
   */
  def this(A: Array[Array[Float]], m: Int, n: Int) {
    this()
    this.A = A
    this.m = m
    this.n = n
  }

  /**
   * Construct a matrix from a one-dimensional packed array
   *
   * @param vals One-dimensional array of floats, packed by columns (ala
   *             Fortran).
   * @param m Number of rows.
   * @exception IllegalArgumentException Array length must be a multiple of m.
   */
  def this(vals: Array[Float], m: Int) {
    this()
    this.m = m
    this.n = (if (m != 0) vals.length / m else 0)
    if (m * n != vals.length) {
      throw new IllegalArgumentException("Array length must be a multiple of m.")
    }
    A = Array.ofDim[Float](m, n)
    for (i <- 0 until m) {
      for (j <- 0 until n) {
        A(i)(j) = vals(i + j * m)
      }
    }
  }

  /**
   * Make a deep copy of a matrix
   */
  def copy: FloatMatrix = {
    val X: FloatMatrix = new FloatMatrix(m, n)
    val C: Array[Array[Float]] = X.getArray
    for (i <- 0 until m) {
      for (j <- 0 until n) {
        C(i)(j) = A(i)(j)
      }
    }
    return X
  }

  /**
   * Clone the FloatMatrix object.
   */
  override def clone: AnyRef = {
    return this.copy
  }

  /**
   * Access the internal two-dimensional array.
   *
   * @return Pointer to the two-dimensional array of matrix elements.
   */
  def getArray: Array[Array[Float]] = {
    return A
  }

  /**
   * Copy the internal two-dimensional array.
   *
   * @return Two-dimensional array copy of matrix elements.
   */
  def getArrayCopy: Array[Array[Float]] = {
    val C: Array[Array[Float]] = Array.ofDim[Float](m, n)
      for (i <- 0 until m) {
        for (j <- 0 until n) {
          C(i)(j) = A(i)(j)
        }
      }
    return C
  }

  /**
   * Make a one-dimensional column packed copy of the internal array.
   *
   * @return FloatMatrix elements packed in a one-dimensional array by
   *         columns.
   */
  def getColumnPackedCopy: Array[Float] = {
    val vals: Array[Float] = new Array[Float](m * n)
    for (i <- 0 until m) {
      for (j <- 0 until n) {
        vals(i + j * m) = A(i)(j)
      }
    }
    return vals
  }

  /**
   * Make a one-dimensional row packed copy of the internal array.
   *
   * @return FloatMatrix elements packed in a one-dimensional array by rows.
   */
  def getRowPackedCopy: Array[Float] = {
    val vals: Array[Float] = new Array[Float](m * n)
    for (i <- 0 until m) {
      for (j <- 0 until n) {
        vals(i * n + j) = A(i)(j)
      }
    }
    return vals
  }

  /**
   * Get row dimension.
   *
   * @return m, the number of rows.
   */
  def getRowDimension: Int = {
    return m
  }

  /**
   * Get column dimension.
   *
   * @return n, the number of columns.
   */
  def getColumnDimension: Int = {
    return n
  }

  /**
   * Get a single element.
   *
   * @param i Row index.
   * @param j Column index.
   * @return A(i,j)
   * @exception ArrayIndexOutOfBoundsException
   */
  def get(i: Int, j: Int): Float = {
    return A(i)(j)
  }

  /**
   * Get a submatrix.
   *
   * @param i0 Initial row index
   * @param i1 Final row index
   * @param j0 Initial column index
   * @param j1 Final column index
   * @return A(i0:i1,j0:j1)
   * @exception ArrayIndexOutOfBoundsException Submatrix indices
   */
  def getFloatMatrix(i0: Int, i1: Int, j0: Int, j1: Int): FloatMatrix = {
    val X: FloatMatrix = new FloatMatrix(i1 - i0 + 1, j1 - j0 + 1)
    val B: Array[Array[Float]] = X.getArray
    try { {
      var i: Int = i0
      while (i <= i1) {
        {
          {
            var j: Int = j0
            while (j <= j1) {
              {
                B(i - i0)(j - j0) = A(i)(j)
              }
              ({
                j += 1; j - 1
              })
            }
          }
        }
        ({
          i += 1; i - 1
        })
      }
    }
    }
    catch {
      case e: ArrayIndexOutOfBoundsException => {
        throw new ArrayIndexOutOfBoundsException("Submatrix indices. Index " + e.getMessage + ". Dimension " + X.getRowDimension + "|" + X.getColumnDimension + "[row|col]")
      }
    }
    return X
  }

  /**
   * Get a submatrix.
   *
   * @param r Array of row indices.
   * @param c Array of column indices.
   * @return A(r(:),c(:))
   * @exception ArrayIndexOutOfBoundsException Submatrix indices
   */
  def getFloatMatrix(r: Array[Int], c: Array[Int]): FloatMatrix = {
    val X: FloatMatrix = new FloatMatrix(r.length, c.length)
    val B: Array[Array[Float]] = X.getArray
    try { {
      var i: Int = 0
      while (i < r.length) {
        {
          {
            var j: Int = 0
            while (j < c.length) {
              {
                B(i)(j) = A(r(i))(c(j))
              }
              ({
                j += 1; j - 1
              })
            }
          }
        }
        ({
          i += 1; i - 1
        })
      }
    }
    }
    catch {
      case e: ArrayIndexOutOfBoundsException => {
        throw new ArrayIndexOutOfBoundsException("Submatrix indices. Index " + e.getMessage + ". Dimension " + X.getRowDimension + "|" + X.getColumnDimension + "[row|col]")
      }
    }
    return X
  }

  /**
   * Get a submatrix.
   *
   * @param i0 Initial row index
   * @param i1 Final row index
   * @param c Array of column indices.
   * @return A(i0:i1,c(:))
   * @exception ArrayIndexOutOfBoundsException Submatrix indices
   */
  def getFloatMatrix(i0: Int, i1: Int, c: Array[Int]): FloatMatrix = {
    val X: FloatMatrix = new FloatMatrix(i1 - i0 + 1, c.length)
    val B: Array[Array[Float]] = X.getArray
    try { {
      var i: Int = i0
      while (i <= i1) {
        {
          {
            var j: Int = 0
            while (j < c.length) {
              {
                B(i - i0)(j) = A(i)(c(j))
              }
              ({
                j += 1; j - 1
              })
            }
          }
        }
        ({
          i += 1; i - 1
        })
      }
    }
    }
    catch {
      case e: ArrayIndexOutOfBoundsException => {
        throw new ArrayIndexOutOfBoundsException("Submatrix indices. Index " + e.getMessage + ". Dimension " + X.getRowDimension + "|" + X.getColumnDimension + "[row|col]")
      }
    }
    return X
  }

  /**
   * Get a submatrix.
   *
   * @param r Array of row indices.
   * @param i0 Initial column index
   * @param i1 Final column index
   * @return A(r(:),j0:j1)
   * @exception ArrayIndexOutOfBoundsException Submatrix indices
   */
  def getFloatMatrix(r: Array[Int], j0: Int, j1: Int): FloatMatrix = {
    val X: FloatMatrix = new FloatMatrix(r.length, j1 - j0 + 1)
    val B: Array[Array[Float]] = X.getArray
    try { {
      var i: Int = 0
      while (i < r.length) {
        {
          {
            var j: Int = j0
            while (j <= j1) {
              {
                B(i)(j - j0) = A(r(i))(j)
              }
              ({
                j += 1; j - 1
              })
            }
          }
        }
        ({
          i += 1; i - 1
        })
      }
    }
    }
    catch {
      case e: ArrayIndexOutOfBoundsException => {
        throw new ArrayIndexOutOfBoundsException("Submatrix indices. Index " + e.getMessage + ". Dimension " + X.getRowDimension + "|" + X.getColumnDimension + "[row|col]")
      }
    }
    return X
  }

  /**
   * Set a single element.
   *
   * @param i Row index.
   * @param j Column index.
   * @param s A(i,j).
   * @exception ArrayIndexOutOfBoundsException
   */
  def set(i: Int, j: Int, s: Float) {
    A(i)(j) = s
  }

  /**
   * Set a submatrix.
   *
   * @param i0 Initial row index
   * @param i1 Final row index
   * @param j0 Initial column index
   * @param j1 Final column index
   * @param X A(i0:i1,j0:j1)
   * @exception ArrayIndexOutOfBoundsException Submatrix indices
   */
  def setFloatMatrix(i0: Int, i1: Int, j0: Int, j1: Int, X: FloatMatrix) {
    try { {
      var i: Int = i0
      while (i <= i1) {
        {
          {
            var j: Int = j0
            while (j <= j1) {
              {
                A(i)(j) = X.get(i - i0, j - j0)
              }
              ({
                j += 1; j - 1
              })
            }
          }
        }
        ({
          i += 1; i - 1
        })
      }
    }
    }
    catch {
      case e: ArrayIndexOutOfBoundsException => {
        throw new ArrayIndexOutOfBoundsException("Submatrix indices. Index " + e.getMessage + ". Dimension " + X.getRowDimension + "|" + X.getColumnDimension + "[row|col]")
      }
    }
  }

  /**
   * Set a submatrix.
   *
   * @param r Array of row indices.
   * @param c Array of column indices.
   * @param X A(r(:),c(:))
   * @exception ArrayIndexOutOfBoundsException Submatrix indices
   */
  def setFloatMatrix(r: Array[Int], c: Array[Int], X: FloatMatrix) {
    try { {
      var i: Int = 0
      while (i < r.length) {
        {
          {
            var j: Int = 0
            while (j < c.length) {
              {
                A(r(i))(c(j)) = X.get(i, j)
              }
              ({
                j += 1; j - 1
              })
            }
          }
        }
        ({
          i += 1; i - 1
        })
      }
    }
    }
    catch {
      case e: ArrayIndexOutOfBoundsException => {
        throw new ArrayIndexOutOfBoundsException("Submatrix indices. Index " + e.getMessage + ". Dimension " + X.getRowDimension + "|" + X.getColumnDimension + "[row|col]")
      }
    }
  }

  /**
   * Set a submatrix.
   *
   * @param r Array of row indices.
   * @param j0 Initial column index
   * @param j1 Final column index
   * @param X A(r(:),j0:j1)
   * @exception ArrayIndexOutOfBoundsException Submatrix indices
   */
  def setFloatMatrix(r: Array[Int], j0: Int, j1: Int, X: FloatMatrix) {
    try { {
      var i: Int = 0
      while (i < r.length) {
        {
          {
            var j: Int = j0
            while (j <= j1) {
              {
                A(r(i))(j) = X.get(i, j - j0)
              }
              ({
                j += 1; j - 1
              })
            }
          }
        }
        ({
          i += 1; i - 1
        })
      }
    }
    }
    catch {
      case e: ArrayIndexOutOfBoundsException => {
        throw new ArrayIndexOutOfBoundsException("Submatrix indices. Index " + e.getMessage + ". Dimension " + X.getRowDimension + "|" + X.getColumnDimension + "[row|col]")
      }
    }
  }

  /**
   * Set a submatrix.
   *
   * @param i0 Initial row index
   * @param i1 Final row index
   * @param c Array of column indices.
   * @param X A(i0:i1,c(:))
   * @exception ArrayIndexOutOfBoundsException Submatrix indices
   */
  def setFloatMatrix(i0: Int, i1: Int, c: Array[Int], X: FloatMatrix) {
    try { {
      var i: Int = i0
      while (i <= i1) {
        {
          {
            var j: Int = 0
            while (j < c.length) {
              {
                A(i)(c(j)) = X.get(i - i0, j)
              }
              ({
                j += 1; j - 1
              })
            }
          }
        }
        ({
          i += 1; i - 1
        })
      }
    }
    }
    catch {
      case e: ArrayIndexOutOfBoundsException => {
        throw new ArrayIndexOutOfBoundsException("Submatrix indices. Index " + e.getMessage + ". Dimension " + X.getRowDimension + "|" + X.getColumnDimension + "[row|col]")
      }
    }
  }

  /**
   * FloatMatrix transpose.
   *
   * @return A'
   */
  def transpose: FloatMatrix = {
    val X: FloatMatrix = new FloatMatrix(n, m)
    val C: Array[Array[Float]] = X.getArray
    for (i <- 0 until m) {
      for (j <- 0 until n) {
        C(j)(i) = A(i)(j)
      }
    }
    return X
  }

  /**
   * One norm
   *
   * @return maximum column sum.
   */
  /*
  def norm1: Float = {
    var f: Float = 0
    {
      var j: Int = 0
      while (j < n) {
        {
          var s: Float = 0
          {
            var i: Int = 0
            while (i < m) {
              {
                s += Math.abs(A(i)(j))
              }
              ({
                i += 1; i - 1
              })
            }
          }
          f = Math.max(f, s)
        }
        ({
          j += 1; j - 1
        })
      }
    }
    return f
  }
  */

  /**
   * Infinity norm
   *
   * @return maximum row sum.
   */
  /*
  def normInf: Float = {
    var f: Float = 0
    {
      var i: Int = 0
      while (i < m) {
        {
          var s: Float = 0
          {
            var j: Int = 0
            while (j < n) {
              {
                s += Math.abs(A(i)(j))
              }
              ({
                j += 1; j - 1
              })
            }
          }
          f = Math.max(f, s)
        }
        ({
          i += 1; i - 1
        })
      }
    }
    return f
  }
  */

  /**
   * Unary minus
   *
   * @return -A
   */
  /*
  def uminus: FloatMatrix = {
    val X: FloatMatrix = new FloatMatrix(m, n)
    val C: Array[Array[Float]] = X.getArray
    {
      var i: Int = 0
      while (i < m) {
        {
          {
            var j: Int = 0
            while (j < n) {
              {
                C(i)(j) = -A(i)(j)
              }
              ({
                j += 1; j - 1
              })
            }
          }
        }
        ({
          i += 1; i - 1
        })
      }
    }
    return X
  }
  */

  /**
   * C = A + B
   *
   * @param B another matrix
   * @return A + B
   */
  def plus(B: FloatMatrix): FloatMatrix = {
    checkFloatMatrixDimensions(B)
    val X: FloatMatrix = new FloatMatrix(m, n)
    val C: Array[Array[Float]] = X.getArray
    for (i <- 0 until m) {
      for (j <- 0 until n) {
        C(i)(j) = A(i)(j) + B.A(i)(j)
      }
    }
    return X
  }

  /**
   * A = A + B
   *
   * @param B another matrix
   * @return A + B
   */
  def plusEquals(B: FloatMatrix): FloatMatrix = {
    checkFloatMatrixDimensions(B)
    for (i <- 0 until m) {
      for (j <- 0 until n) {
        A(i)(j) = A(i)(j) + B.A(i)(j)
      }
    }
    return this
  }

  /**
   * C = A - B
   *
   * @param B another matrix
   * @return A - B
   */
  def minus(B: FloatMatrix): FloatMatrix = {
    checkFloatMatrixDimensions(B)
    val X: FloatMatrix = new FloatMatrix(m, n)
    val C: Array[Array[Float]] = X.getArray
    for (i <- 0 until m) {
      for (j <- 0 until n) {
        C(i)(j) = A(i)(j) - B.A(i)(j)
      }
    }
    return X
  }

  /**
   * A = A - B
   *
   * @param B another matrix
   * @return A - B
   */
  def minusEquals(B: FloatMatrix): FloatMatrix = {
    checkFloatMatrixDimensions(B)
    for (i <- 0 until m) {
      for (j <- 0 until n) {
        A(i)(j) = A(i)(j) - B.A(i)(j)
      }
    }
    return this
  }

  /**
   * Element-by-element multiplication, C = A.*B
   *
   * @param B another matrix
   * @return A.*B
   */
  def arrayTimes(B: FloatMatrix): FloatMatrix = {
    checkFloatMatrixDimensions(B)
    val X: FloatMatrix = new FloatMatrix(m, n)
    val C: Array[Array[Float]] = X.getArray
    for (i <- 0 until m) {
      for (j <- 0 until n) {
        C(i)(j) = A(i)(j) * B.A(i)(j)
      }
    }
    return X
  }

  /**
   * Element-by-element multiplication in place, A = A.*B
   *
   * @param B another matrix
   * @return A.*B
   */
  def arrayTimesEquals(B: FloatMatrix): FloatMatrix = {
    checkFloatMatrixDimensions(B)
    for (i <- 0 until m) {
      for (j <- 0 until n) {
        A(i)(j) = A(i)(j) * B.A(i)(j)
      }
    }
    return this
  }

  /**
   * Element-by-element right division, C = A./B
   *
   * @param B another matrix
   * @return A./B
   */
  def arrayRightDivide(B: FloatMatrix): FloatMatrix = {
    checkFloatMatrixDimensions(B)
    val X: FloatMatrix = new FloatMatrix(m, n)
    val C: Array[Array[Float]] = X.getArray
    for (i <- 0 until m) {
      for (j <- 0 until n) {
        C(i)(j) = A(i)(j) / B.A(i)(j)
      }
    }
    return X
  }

  /**
   * Element-by-element right division in place, A = A./B
   *
   * @param B another matrix
   * @return A./B
   */
  def arrayRightDivideEquals(B: FloatMatrix): FloatMatrix = {
    checkFloatMatrixDimensions(B)
    for (i <- 0 until m) {
      for (j <- 0 until n) {
        A(i)(j) = A(i)(j) / B.A(i)(j)
      }
    }
    return this
  }

  /**
   * Element-by-element left division, C = A.\B
   *
   * @param B another matrix
   * @return A.\B
   */
  def arrayLeftDivide(B: FloatMatrix): FloatMatrix = {
    checkFloatMatrixDimensions(B)
    val X: FloatMatrix = new FloatMatrix(m, n)
    val C: Array[Array[Float]] = X.getArray
    for (i <- 0 until m) {
      for (j <- 0 until n) {
        C(i)(j) = B.A(i)(j) / A(i)(j)
      }
    }
    return X
  }

  /**
   * Element-by-element left division in place, A = A.\B
   *
   * @param B another matrix
   * @return A.\B
   */
  def arrayLeftDivideEquals(B: FloatMatrix): FloatMatrix = {
    checkFloatMatrixDimensions(B)
    for (i <- 0 until m) {
      for (j <- 0 until n) {
        A(i)(j) = B.A(i)(j) / A(i)(j)
      }
    }
    return this
  }

  /**
   * Multiply a matrix by a scalar, C = s*A
   *
   * @param s scalar
   * @return s*A
   */
  def times(s: Float): FloatMatrix = {
    val X: FloatMatrix = new FloatMatrix(m, n)
    val C: Array[Array[Float]] = X.getArray
    for (i <- 0 until m) {
      for (j <- 0 until n) {
        C(i)(j) = s * A(i)(j)
      }
    }
    return X
  }

  /**
   * Multiply a matrix by a scalar in place, A = s*A
   *
   * @param s scalar
   * @return replace A by s*A
   */
  def timesEquals(s: Float): FloatMatrix = {
    for (i <- 0 until m) {
      for (j <- 0 until n) {
        A(i)(j) = s * A(i)(j)
      }
    }
    return this
  }

  /**
   * Linear algebraic matrix multiplication, A * B
   *
   * @param B another matrix
   * @return FloatMatrix product, A * B
   * @exception IllegalArgumentException FloatMatrix inner dimensions must
   *            agree.
   */
  def times(B: FloatMatrix): FloatMatrix = {
    if (B.m != n) {
      throw new IllegalArgumentException("FloatMatrix inner dimensions must agree.")
    }
    val X: FloatMatrix = new FloatMatrix(m, B.n)
    val C: Array[Array[Float]] = X.getArray
    val Bcolj: Array[Float] = new Array[Float](n)
    for (j <- 0 until B.n) {
      for (k <- 0 until n) {
        Bcolj(k) = B.A(k)(j)
      }
      for (i <- 0 until m) {
        val Arowi: Array[Float] = A(i)
        var s: Float = 0
        for (k <- 0 until n) {
          s += Arowi(k) * Bcolj(k)
        }
        C(i)(j) = s
      }
    }
    return X
  }

  def parallelTimes(B: FloatMatrix): FloatMatrix = {
    if (B.m != n) {
      throw new IllegalArgumentException("FloatMatrix inner dimensions must agree.")
    }
    val X: FloatMatrix = new FloatMatrix(m, B.n)
    val C: Array[Array[Float]] = X.getArray
    var rowsDone: Int = 0
    var printEvery: Int = B.n / 1000
    for (j <- (0 until B.n).par) {
      val Bcolj: Array[Float] = new Array[Float](n)
      for (k <- 0 until n) {
        Bcolj(k) = B.A(k)(j)
      }
      for (i <- 0 until m) {
        val Arowi: Array[Float] = A(i)
        var s: Float = 0
        for (k <- 0 until n) {
          s += Arowi(k) * Bcolj(k)
        }
        C(i)(j) = s
      }
      rowsDone += 1
      if (rowsDone % printEvery == 0) println(s"finished $rowsDone rows of ${B.n}")
    }
    return X

  }

  /**
   * FloatMatrix trace.
   *
   * @return sum of the diagonal elements.
   */
  def trace: Float = {
    var t: Float = 0
    for (i <- 0 until Math.min(m, n)) {
      t += A(i)(i)
    }
    return t
  }

  /**
   * Note: This method creates a Matrix object which needs almost doubled
   * memory size.
   *
   * @return
   */
  /*
  def toMatrix: Nothing = {
    val matrix: Nothing = new Nothing(m, n)
    for (i <- 0 until m) {
      for (j <- 0 until n) {
        matrix.set(i, j, A(i)(j))
      }
    }
    return matrix
  }
  */

  /**
   * Print the matrix to stdout. Line the elements up in columns with a
   * Fortran-like 'Fw.d' style format.
   *
   * @param w Column width.
   * @param d Number of digits after the decimal.
   */
  /*
  def print(w: Int, d: Int) {
    print(new PrintWriter(System.out, true), w, d)
  }
  */

  /**
   * Print the matrix to the output stream. Line the elements up in columns
   * with a Fortran-like 'Fw.d' style format.
   *
   * @param output Output stream.
   * @param w Column width.
   * @param d Number of digits after the decimal.
   */
  /*
  def print(output: PrintWriter, w: Int, d: Int) {
    val format: DecimalFormat = new DecimalFormat
    format.setDecimalFormatSymbols(new DecimalFormatSymbols(Locale.US))
    format.setMinimumIntegerDigits(1)
    format.setMaximumFractionDigits(d)
    format.setMinimumFractionDigits(d)
    format.setGroupingUsed(false)
    print(output, format, w + 2)
  }
  */

  /**
   * Print the matrix to stdout. Line the elements up in columns. Use the
   * format object, and right justify within columns of width characters. Note
   * that is the matrix is to be read back in, you probably will want to use a
   * NumberFormat that is set to US Locale.
   *
   * @param format A Formatting object for individual elements.
   * @param width Field width for each column.
   * @see java.text.DecimalFormat#setDecimalFormatSymbols
   */
  /*
  def print(format: NumberFormat, width: Int) {
    print(new PrintWriter(System.out, true), format, width)
  }
  */

  /**
   * Print the matrix to the output stream. Line the elements up in columns.
   * Use the format object, and right justify within columns of width
   * characters. Note that is the matrix is to be read back in, you probably
   * will want to use a NumberFormat that is set to US Locale.
   *
   * @param output the output stream.
   * @param format A formatting object to format the matrix elements
   * @param width Column width.
   * @see java.text.DecimalFormat#setDecimalFormatSymbols
   */
  /*
  def print(output: PrintWriter, format: NumberFormat, width: Int) {
    output.println
    {
      var i: Int = 0
      while (i < m) {
        {
          {
            var j: Int = 0
            while (j < n) {
              {
                val s: String = format.format(A(i)(j))
                val padding: Int = Math.max(1, width - s.length)
                {
                  var k: Int = 0
                  while (k < padding) {
                    {
                      output.print(' ')
                    }
                    ({
                      k += 1; k - 1
                    })
                  }
                }
                output.print(s)
              }
              ({
                j += 1; j - 1
              })
            }
          }
          output.println
        }
        ({
          i += 1; i - 1
        })
      }
    }
    output.println
  }
  */

  /*
  override def hashCode: Int = {
    val prime: Int = 31
    var result: Int = 1
    result = prime * result + Arrays.hashCode(A)
    return result
  }
  */

  /*
  override def equals(obj: AnyRef): Boolean = {
    if (this eq obj) {
      return true
    }
    if (obj == null) {
      return false
    }
    if (getClass ne obj.getClass) {
      return false
    }
    val other: FloatMatrix = obj.asInstanceOf[FloatMatrix]
    if (!Arrays.deepEquals(A, other.A)) {
      return false
    }
    return true
  }
  */

  /** Check if size(A) == size(B) **/
  private def checkFloatMatrixDimensions(B: FloatMatrix) {
    if (B.m != m || B.n != n) {
      throw new IllegalArgumentException("FloatMatrix dimensions must agree.")
    }
  }

  /**
   * Array for internal storage of elements.
   *
   * @serial internal array storage.
   */
  private var A: Array[Array[Float]] = null
  /**
   * Row and column dimensions.
   *
   * @serial row dimension.
   * @serial column dimension.
   */
  private var m: Int = 0
  private var n: Int = 0
}
