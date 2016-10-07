package edu.arizona.sista.qa.discourse

import edu.arizona.sista.processors.Document
import edu.arizona.sista.qa.ranking.{ProcessedQuestionSegments, ProcessedQuestion, RankingModel}
import edu.arizona.sista.qa.index.TermFilter
import edu.arizona.sista.qa.segmenter._
import edu.arizona.sista.qa.retrieval.AnswerCandidate
import edu.arizona.sista.qa.segmenter.Segment
import edu.arizona.sista.qa.segmenter.SegmenterDiscourse
import edu.arizona.sista.qa.segmenter.SegmentMatcherBOW
import edu.arizona.sista.struct.Counter
import java.util.Properties
import collection.mutable.ArrayBuffer
import org.slf4j.LoggerFactory
import DiscourseModelCusp.logger

import edu.arizona.sista.utils.{Profiler, StringUtils}
import java.io.PrintWriter

/**
 * Discourse ranking model using an n-gram representation scheme for discourse features
 * User: peter
 * Date: 5/15/13
 */
class DiscourseModelNGram(props:Properties) extends RankingModel {
  val termFilter = new TermFilter()
  val indexDir = props.getProperty("index")
  val qProcessor = new QuestionProcessor(props)
  //var ngramSize:Int = 5

  def mkFeatures( answer:AnswerCandidate,
                  question:ProcessedQuestion,
                  externalFeatures:Option[Counter[String]],
                  errorPw:PrintWriter = null): (Counter[String], String) = {

    question match {
      case q:ProcessedQuestionSegments => (mkFeaturesNGram (answer, q), null)
      case _ => throw new RuntimeException ("DiscourseModelCusp.mkFeatures(): question passed is not of type ProcessedQuestionSegments")
    }

  }

  def mkProcessedQuestion(question:Document):Array[ProcessedQuestion] = {
    // Changed to return an array of ProcessedQuestions.  In many cases this will just contain one processedQuestion, but when using the hybrid method it may contain two.
    if (props.getProperty("discourse.question_processor").toUpperCase == "ONESEG") return Array[ProcessedQuestion](qProcessor.mkProcessedQuestionOneArgument(question))
    if (props.getProperty("discourse.question_processor").toUpperCase == "SVO") return Array[ProcessedQuestion]( qProcessor.mkProcessedQuestionSVO(question) )
    if (props.getProperty("discourse.question_processor").toUpperCase == "HYBRIDSVO") {
      val processedQuestions = new ArrayBuffer[ProcessedQuestion]()
      processedQuestions.append (qProcessor.mkProcessedQuestionOneArgument(question))         // Add ONESEG by default
      processedQuestions.append (qProcessor.mkProcessedQuestionSVO(question))                 // Also use Subj/Verb/Obj/IndObj
      return processedQuestions.toArray
    }

    // default to QSEG if method is unknown
    return Array[ProcessedQuestion](qProcessor.mkProcessedQuestionOneArgument(question))
  }



  def mkFeaturesNGram( answer:AnswerCandidate,
                       question:ProcessedQuestionSegments): Counter[String] = {
    val discourseSegmenter = new SegmenterDiscourse(props.getProperty("discourse.connectives_list_size", "FULL"))
    val segmentMatcher = new SegmentMatcherBOW(termFilter, indexDir)
    val features = new Counter[String]()
    val doc = answer.doc.annotation
    val minScoreThresh:Double = StringUtils.getDouble(props, "discourse.match_threshold", 0.0)
    val binaryScore:Boolean = StringUtils.getBool(props, "discourse.binary_score", false)
    val ngramSizeStart = StringUtils.getInt(props, "discourse.ngram_start_size", 2)
    val ngramSizeEnd = StringUtils.getInt(props, "discourse.ngram_end_size", 2)

    // Features take the form of strings, such as: "OTHER2 BECAUSE OTHER3"  (X because Y -- only X and Y are now numbered "OTHER" segments)

    // Step 1: IR Feature must be included
    // mihai: this is no longer needed if we use the hierarchical model
    if (props.getProperty("ranker.train_method", "incremental").toLowerCase != "hierarchical") {
      features.setCount("ir", answer.score)
    }

    // Prepare segments from question (Arguments (X, Y), as well as verb segments)
    val qMatchSegs = question.segments

    // Step 2: Locate discourse connectives in answer text
    val discourseSegs = discourseSegmenter.segment(doc, answer.sentenceStart, answer.sentenceEnd)
    // Option: Stop "OTHER" segments on EOS boundaries, rather than bridging these boundaries (like the CUSP model)
    //val discourseSegs = discourseSegmenter.segmentWithEOS(doc, answer.sentenceStart, answer.sentenceEnd)


    // TODO: Combinatorial expansion? (It won't impact QSEG, but should impact multisegment question segmentation models like SVO)

    // Step 3: Relabel "OTHER" segments with arguments, if they question segments (e.g. QSEG, SVO, XVY, etc)
    val labelledSegs = new ArrayBuffer[Segment]()        // New list of labelled segments
    val segScores = new ArrayBuffer[Double]()            // Array of BOW match scores for each segment (to weight the feature score)
    for (disSeg <- discourseSegs) {
      if (disSeg.label == "OTHER") {
        var labelNew:String = ""
        var labelScore:Double = 0
        for (i <- 0 until qMatchSegs.size) {
          val otherSeg = qMatchSegs(i)
          val score = segmentMatcher.score(disSeg, otherSeg)
          // Step 3B: Store non-zero scores as features. (Append index of other segments to their label. e.g. OTHER -> OTHER3)
          if (score > minScoreThresh) {
            labelNew += otherSeg.label
            labelScore += score
          }
        }
        if (labelNew.length > 0) {
          // Relabel from "OTHER" to new label
          labelledSegs.append( new Segment(labelNew, disSeg.doc, disSeg.startOffset, disSeg.endOffset) )
          segScores.append ( labelScore )
        } else {
          // Maintain original segment label
          labelledSegs.append(disSeg)
          segScores.append ( labelScore )   // "OTHER" segments receive a score of 0, since there is nothing specific to match
        }
      }
      labelledSegs.append(disSeg)
      segScores.append ( 1 )    // Assign a score of 1 per discourse connective
    }

    for (ngramSize <- ngramSizeStart to ngramSizeEnd) {
      // Step 4: Generate n-grams over this discourse represention
      val ngrams = new ArrayBuffer[String]()
      val ngramScores = new ArrayBuffer[Double]()
      for (i <- 0 until labelledSegs.size-ngramSize) {
        var feature:String = ""
        var score:Double = 0
        for (j <- 0 until ngramSize) {
          feature += labelledSegs(i+j).label + " "
          score += segScores(i+j)
        }
        ngrams.append(feature)
        ngramScores.append(score)
      }

      // Step 5: Return a counter of these n-gram features
      for (i <- 0 until ngrams.size) {
        val feature = ngrams(i) + " N" + ngramSize.toString
        var featureScore = ngramScores(i)                                                           // Real-valued score
        logger.debug(" Feature: " + feature)

        if (binaryScore) featureScore = 1                                                           // binary score
        features.setCount(feature, featureScore)
      }
    }

    // return list of features
    features
  }

}


object DiscourseModelNGram {
  val logger = LoggerFactory.getLogger(classOf[DiscourseModelCusp])
}