package edu.arizona.sista.qa.retrieval

import org.apache.lucene.store.SimpleFSDirectory
import java.io.File
import org.apache.lucene.index.{Term, IndexReader}
import edu.arizona.sista.struct.Counter

/**
 * Scores two documents using tfidf + cosine similarity
 * User: mihais
 * Date: 4/4/13
 */
class IRModel(val docFreqExtractor:DocFreqExtractor) {

  /** See lecture 6 in Manning's IR course for the formula for lnn */
  def lnn(orig:Counter[String]):Counter[String] = {
    val lnnVector = new Counter[String]
    for (key <- orig.keySet) {
      lnnVector.setCount(key, 1 + math.log(orig.getCount(key)))
    }
    lnnVector
  }

  /** See lecture 6 in Manning's IR course for the formula for ltn */
  def ltn(orig:Counter[String]):Counter[String] = {
    val N = docFreqExtractor.numDocs
    val ltnVector = new Counter[String]
    var df = 0
    for (key <- orig.keySet) {
      df = docFreqExtractor.df(key)
      if(df == 0) df = 1 // just to avoid division by 0; it doesn't really matter, this term will not be used in the dot product
      val v = (1 + math.log(orig.getCount(key))) * math.log(N / df)
      ltnVector.setCount(key, v)
    }
    ltnVector
  }
}

trait DocFreqExtractor {
  def df(term:String):Int
  def numDocs:Int
}

class DocFreqExtractorFromIndex(
  val indexDir:String,
  val field:String) extends DocFreqExtractor {

  val indexReader = IndexReader.open(new SimpleFSDirectory(new File(indexDir)))

  def df(term:String):Int = indexReader.docFreq(new Term(field, term))
  def numDocs:Int = indexReader.numDocs()

  def close() { indexReader.close() }
}
