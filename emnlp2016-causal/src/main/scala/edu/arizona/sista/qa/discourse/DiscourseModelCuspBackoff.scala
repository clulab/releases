package edu.arizona.sista.qa.discourse

import java.util.Properties
import edu.arizona.sista.qa.ranking.{ProcessedQuestionSegments, ProcessedQuestion, RankingModel}
import edu.arizona.sista.qa.index.TermFilter
import edu.arizona.sista.qa.segmenter.{Segment, SegmentMatcherBOW, SegmenterDiscourse, QuestionProcessor}
import edu.arizona.sista.utils.StringUtils
import edu.arizona.sista.qa.word2vec.Word2VecModel
import edu.arizona.sista.qa.retrieval.AnswerCandidate
import edu.arizona.sista.struct.Counter
import edu.arizona.sista.processors.Document
import collection.mutable.{ListBuffer, ArrayBuffer}
import org.slf4j.LoggerFactory
import java.io.PrintWriter

/**
 * Created with IntelliJ IDEA.
 * User: peter
 * Date: 12/12/13
 */
class DiscourseModelCuspBackoff(props:Properties) extends RankingModel {
  val termFilter = new TermFilter()
  val indexDir = props.getProperty("index")
  val qProcessor = new QuestionProcessor(props)
  val modelWord2Vec = new Word2VecModel(props)
  val modelCusp = new DiscourseModelCusp(props)

  def mkFeatures( answer:AnswerCandidate,
                  question:ProcessedQuestion,
                  externalFeatures:Option[Counter[String]],
                  errorPw:PrintWriter = null): (Counter[String], String) = {

    question match {
      case q:ProcessedQuestionSegments => (mkFeaturesCuspDispatch (answer, q), null)
      case _ => throw new RuntimeException ("DiscourseModelCusp.mkFeatures(): question passed is not of type ProcessedQuestionSegments")
    }

  }


  def mkFeaturesCuspDispatch (answer:AnswerCandidate,
                              question:ProcessedQuestionSegments): Counter[String] = {

    val featuresMarkersW2V = mkFeaturesCusp(answer, question)
    //val featuresMarkers = modelCusp.mkFeaturesCusp(answer, question)
    val featuresWord2Vec = modelWord2Vec.mkFeatures(answer, question, None)._1
    //var combined = featuresMarkers + featuresMarkersW2V + featuresWord2Vec
    var combined = featuresMarkersW2V + featuresWord2Vec
    return combined
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

  /**
   * Features take the form of strings, such as: "OTHER BECAUSE OTHER".  Here, we relabel OTHER segments based on matches with
   * question segments (e.g. QSEG, XVY, SVOI, etc), and generate both those string features (e.g. "QSEG BECAUSE QSEG") and
   * corresponding scores (e.g. 0.90) for those features.
   */
  def mkFeaturesCusp( answer:AnswerCandidate,
                      question:ProcessedQuestionSegments): Counter[String] = {

    val discourseSegmenter = new SegmenterDiscourse(props.getProperty("discourse.connectives_list_size", "FULL"))
    val segmentMatcher = new SegmentMatcherBOW(termFilter, indexDir)
    val features = new Counter[String]()
    val doc = answer.doc.annotation
    val sentRangeStart = StringUtils.getInt(props, "discourse.sentrange_start", 0)
    val sentRangeEnd = StringUtils.getInt(props, "discourse.sentrange_end", 3)
    val minScoreThresh:Double = StringUtils.getDouble(props, "discourse.match_threshold_cusp", 0.20)
    val minScoreThreshW2V:Double = StringUtils.getDouble(props, "discourse.match_threshold_w2v_cusp", 0.75)
    val appendSR:Boolean = StringUtils.getBool(props, "discourse.add_sentence_range_marker", true)
    val binaryScore:Boolean = StringUtils.getBool(props, "discourse.binary_score", false)


    // Step 1: IR Feature must be included
    // mihai: this is no longer needed if we use the hierarchical model
    if (props.getProperty("ranker.train_method", "incremental").toLowerCase != "hierarchical") {
      features.setCount("ir", answer.score)
    }

    // Step 1A: If we failed to properly segment the question, then return an empty feature list.  Default back to other methods (e.g. IR)
    if (question.questionType == "failed") return features

    // Step 2: Locate discourse connectives in answer text
    val discourseSegs = discourseSegmenter.segment(doc, answer.sentenceStart, answer.sentenceEnd)
    val qMatchSegs = question.segments
    val qMatchSegsLTN = question.segmentsLTN

    val w2vWordsQuestion = modelWord2Vec.mkWordsFromAnnotation( question.segments(0).doc )

    // Step 3: Cycle through all possible "cusp" representations of the discourse connectives and OTHER segments, checking for matches.
    for (disSeg <- discourseSegs) {
      if (disSeg.label != "OTHER") {
        for (sentRange <- sentRangeStart to sentRangeEnd) {
          val cuspStartOffset = findEOSBoundaryBefore(doc, disSeg.startOffset, sentRange)
          val cuspEndOffset = findEOSBoundaryAfter(doc, disSeg.endOffset, sentRange)
          val cuspSegBefore = new Segment("BEFORE", doc, cuspStartOffset, disSeg.startOffset)
          val cuspSegAfter = new Segment("AFTER", doc, disSeg.endOffset, cuspEndOffset)

          val w2vWordsBefore = extractWordsFromSegment(cuspSegBefore)
          val w2vWordsAfter = extractWordsFromSegment(cuspSegAfter)

          // Step 3A: Try to match the BEFORE segment of the cusp with segments from the question
          val beforeMatches = new ArrayBuffer[(String, Double)]
          for (i <- 0 until qMatchSegs.size) {
            // Normal QSEG
            val otherSeg = qMatchSegs(i)

            val score = segmentMatcher.scoreWithPrecomputedLTNVector(qMatchSegsLTN(i), cuspSegBefore)       // Use precomputed LTNs for dramatic speed increase
            // If the score between the BEFORE segment and a given question segment is greater than minScoreThresh, then store the new label from the question segment (the score)
            if (score > minScoreThresh) {
              beforeMatches.append ( (otherSeg.label, score) )
            } else {
              // QSEG_W2V as back-off
              val scorew2v = modelWord2Vec.model.textSimilarity(w2vWordsQuestion, w2vWordsBefore)
              //println ("scorew2v: " + scorew2v + " (w2vBefore: " + w2vWordsBefore.toList + "  w2vWordsQuestion: " + w2vWordsQuestion.toList + ")")
              if (scorew2v > minScoreThreshW2V) {
                beforeMatches.append ( (otherSeg.label + "W2V", scorew2v) )
              }
            }
          }

          // Step 3B: Try to match the AFTER segment of the cusp with segments from the question
          val afterMatches = new ArrayBuffer[(String, Double)]
          for (i <- 0 until qMatchSegs.size) {
            // Normal QSEG
            val otherSeg = qMatchSegs(i)

            val score = segmentMatcher.scoreWithPrecomputedLTNVector(qMatchSegsLTN(i), cuspSegAfter)      // Use precomputed LTNs for dramatic speed increase
            // If the score between the BEFORE segment and a given question segment is greater than minScoreThresh, then store the new label from the question segment (the score)
            if (score > minScoreThresh) {
              afterMatches.append ( (otherSeg.label, score) )
            } else {
              // QSEG_W2V as back-off
              val scorew2v = modelWord2Vec.model.textSimilarity(w2vWordsQuestion, w2vWordsAfter)
              if (scorew2v > minScoreThreshW2V) {
                afterMatches.append ( (otherSeg.label + "W2V", scorew2v) )
                //println ("scorew2v: " + scorew2v)
              }
            }

          }

          // Step 4: Generate all combinations of features
          val beforeCombinations = (1 to beforeMatches.size).flatMap(beforeMatches.toList.combinations).toList
          val afterCombinations = (1 to afterMatches.size).flatMap(afterMatches.toList.combinations).toList

          // Step 4A: Check if we've found at least one match in the BEFORE or AFTER cusp segments
          if ((beforeCombinations.size > 0) || (afterCombinations.size > 0)) {

            // Step 4B: If one cusp segment did not match, then use a wildcard for its feature description
            // Note: Append wildcard match ( "*", 0.0 ) to the end of the before and after combinations lists
            val wildcard = new ListBuffer[(String, Double)]
            wildcard.append  ( ("*", 0.0) )

            val beforeCombinations1 = new ListBuffer[ List[(String, Double)] ]
            for (bc <- beforeCombinations) beforeCombinations1.append( bc )
            beforeCombinations1.append (wildcard.toList)

            val afterCombinations1 = new ListBuffer[ List[(String, Double)] ]
            for (ac <- afterCombinations) afterCombinations1.append( ac )
            afterCombinations1.append (wildcard.toList)

            // Step 4C: Iterate over all feature combinations -- Store each feature combination, and its respective score
            for (before <- beforeCombinations1) {
              var beforeLabel:String = ""
              var beforeScore:Double = 0.0f
              for (a <- 0 until before.size) {
                beforeLabel += before(a)._1
                beforeScore += (before(a)._2 / before.size.toDouble)
              }

              for (after <- afterCombinations1) {
                var afterLabel:String = ""
                var afterScore:Double = 0.0f
                for (b <- 0 until after.size) {
                  afterLabel += after(b)._1
                  afterScore += (after(b)._2 / after.size.toDouble)
                }

                // Step 4D: Generate one feature description string and store each feature in feature counter
                if ((beforeLabel != "*") || (afterLabel != "*")) {
                  // Feature description string
                  var featureDesc:String = beforeLabel + " " + disSeg.label + " " + afterLabel
                  //if (question.verbSegments.size > 0) featureDesc += " " + question.verbSegments(0).label + "MODEL"     // Append whether we're using the Verb or Proposition segmentation model (unused?)
                  if (appendSR) featureDesc += " SR" + sentRange             // Append Sentence Range

                  // Feature Score
                  var featureScore = (beforeScore + afterScore) / 2          // Real-valued feature
                  if (binaryScore) featureScore = 1                          // binary score

                  // Store one feature

                  if (features.getCount(featureDesc) < featureScore) {       // If the feature already exists, use the highest score
                    features.setCount(featureDesc, featureScore)
                  }
                  //features.incrementCount(featureDesc, featureScore)

                  // logger.debug ("Feature Found!  (feature=" + featureDesc + ")  (score:" + featureScore + ")")
                }
              }
            }
          }  // Step 4 (feature generation)

        }
      }
    } // Step 3 (segment matching/scoring)

    // return list of features
    features
  }


  /**
   * Returns a Tuple2(sendOffset, tokenOffset) representing an end-of-sentence boundary that is 'numBefore' sentences before 'position'
   * @param doc
   * @param position
   * @param numBefore
   * @return
   */
  def findEOSBoundaryBefore(doc:Document, position:Tuple2[Int, Int], numBefore:Int):Tuple2[Int, Int] = {
    var newSentIdx = position._1 - numBefore
    if (newSentIdx < 0) newSentIdx = 0
    return (newSentIdx, 0)
  }

  /**
   * Returns a Tuple2(sendOffset, tokenOffset) representing an end-of-sentence boundary that is 'numAfter' sentences after 'position'
   * @param doc
   * @param position
   * @param numAfter
   * @return
   */
  def findEOSBoundaryAfter(doc:Document, position:Tuple2[Int, Int], numAfter:Int):Tuple2[Int, Int] = {
    var newSentIdx = position._1 + numAfter
    if (newSentIdx >= doc.sentences.size) newSentIdx = doc.sentences.size-1
    return (newSentIdx, doc.sentences(newSentIdx).size)
  }


  def extractWordsFromSegment(seg:Segment):Array[String] = {
    val outWords = new ArrayBuffer[String]
    val start = seg.startOffset
    val end = seg.endOffset

    for (s <- start._1 to end._1) {
      val sent = seg.doc.sentences(s)

      // Determine start/stop offsets for the current sentence in a given segment
      var sStart = 0
      var sEnd = 0
      if (s == start._1 && s == end._1) {
        // Segment spans single sentence
        sStart = start._2
        sEnd = end._2
      } else {
        // Segment spans multiple sentences
        if (s == start._1) {
          // we're on the first sentence
          sStart = start._2
          sEnd = sent.size
        } else if (s == end._1) {
          // we're on the last sentence
          sStart = 0
          sEnd = end._2
        } else {
          // we're on a middle sentence
          sStart = 0
          sEnd = sent.size
        }
      }

      for (i <- sStart until sEnd) {
        val word = sent.words(i)
        outWords += word
      }
    }

    outWords.toArray
  }



}


object DiscourseModelCuspBackoff {
  val logger = LoggerFactory.getLogger(classOf[DiscourseModelCuspBackoff])
}
