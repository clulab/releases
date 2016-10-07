package edu.arizona.sista.qa.segmenter

import edu.arizona.sista.qa.retrieval.{IRModel, DocFreqExtractorFromIndex}
import edu.arizona.sista.struct.{Counters, Counter}
import edu.arizona.sista.qa.index.{TermFilter, Indexer}
import collection.mutable.ListBuffer
import edu.arizona.sista.processors.Document

/**
 * Matches segments based on a bag-of-words model
 * User: peter
 * Date: 4/26/13
 */
class SegmentMatcherBOW(
  val termFilter:TermFilter,
  val indexDir:String) extends SegmentMatcher {


  def score(segA:Segment, segB:Segment):Double = {
    // NOTE: Here we use LTN to incorporate the docFrequency of the segment terms, and LNN on the
    // other segment so as not to weigh frequent terms down too heavily.
    //val segAVector = buildVectorFromSegmentLTN(indexDir, segA)
    val segAVector = buildVectorFromSegmentLNN(segA)        // Efficiency test
    val segBVector = buildVectorFromSegmentLNN(segB)

    // the cosine does L2 normalization inside; that's why we use lnn and ltn above
    Counters.cosine(segAVector, segBVector)
  }

  def scoreWithPrecomputedLTNVector(preLTN:Counter[String], segB:Segment):Double = {
    // Dramatically increases scoring speed if the same LTNs are used repeatedly, especially on large corpora.
    val segBVector = buildVectorFromSegmentLNN(segB)

    // the cosine does L2 normalization inside; that's why we use lnn and ltn above
    Counters.cosine(preLTN, segBVector)
  }

  def scoreWithPrecomputedLTNVector(preLTN:Counter[String], docB:Document):Double = {
    // Dramatically increases scoring speed if the same LTNs are used repeatedly, especially on large corpora.
    val docBVector = buildVectorFromDocumentLNN(docB)

    // the cosine does L2 normalization inside; that's why we use lnn and ltn above
    Counters.cosine(preLTN, docBVector)
  }

  def buildVectorFromSegmentLTN(indexDir:String, seg:Segment):Counter[String] = {
    val dfx = new DocFreqExtractorFromIndex(indexDir, Indexer.TEXT)
    val model = new IRModel(dfx)
    val ltn = model.ltn(new Counter[String](termFilter.extractValidLemmasFromSegment(seg)))
    dfx.close()
    ltn
  }

  def buildVectorFromSegmentLNN(seg:Segment):Counter[String] = {
    val model = new IRModel(null)
    model.lnn(new Counter[String](termFilter.extractValidLemmasFromSegment(seg)))
  }

  def buildVectorFromDocumentLNN(doc:Document):Counter[String] = {
    val model = new IRModel(null)
    model.lnn(new Counter[String](termFilter.extractValidLemmas(doc)))
  }
}

