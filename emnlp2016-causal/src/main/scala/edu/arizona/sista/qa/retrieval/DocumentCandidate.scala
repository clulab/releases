package edu.arizona.sista.qa.retrieval

import edu.arizona.sista.processors.Document

/**
 * Stores one document candidate, as retrieved from the index
 * User: mihais
 * Date: 3/15/13
 */
class DocumentCandidate (
  val docid:String,

  /** NLP annotation of the entire document; retrieved from index */
  val annotation:Document,

  /** Paragraph ids for this document; retrieved from index */
  val paragraphs:Array[Int],

  /** Document score from the IR system */
  val docScore:Double) extends Serializable {

  override def equals(other:Any):Boolean = {
    other match {
      case that:DocumentCandidate => (docid == that.docid)
      case _ => false
    }
  }

  override def hashCode = docid.hashCode
}
