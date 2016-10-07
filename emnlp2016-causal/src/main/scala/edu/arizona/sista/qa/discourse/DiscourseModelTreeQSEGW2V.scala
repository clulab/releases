package edu.arizona.sista.qa.discourse

import org.slf4j.LoggerFactory
import edu.arizona.sista.qa.ranking.{ProcessedQuestionSegments, ProcessedQuestion, RankingModel}
import edu.arizona.sista.qa.retrieval.AnswerCandidate
import edu.arizona.sista.struct.Counter
import edu.arizona.sista.processors.Document
import edu.arizona.sista.qa.segmenter.{Segment, SegmentMatcherBOW, QuestionProcessor}
import java.util.Properties
import edu.arizona.sista.qa.discparser.{Node, DiscReader}
import edu.arizona.sista.qa.index.TermFilter
import DiscourseModelTree.logger
import edu.arizona.sista.utils.StringUtils
import collection.mutable.ArrayBuffer
import edu.arizona.sista.qa.word2vec.Word2VecModel
import java.io.PrintWriter

/**
 * Discourse model based on the discourse parser of Feng and Hurst
 * This corrently works only for collections where the answer is the entire document!
 * Note: this version has been modified to only generate QSEG's with Word2Vec matches
 * User: mihais
 * Date: 12/11/13
 */
class DiscourseModelTreeQSEGW2V (val props:Properties) extends RankingModel {
  lazy val termFilter = new TermFilter()
  lazy val qProcessor = new QuestionProcessor(props)
  lazy val discReader = new DiscReader(props.getProperty("discourse.dir"))
  lazy val indexDir = props.getProperty("index")
  lazy val segmentMatcher = new SegmentMatcherBOW(termFilter, indexDir)
  lazy val minScoreThreshW2V = StringUtils.getDouble(props, "discourse.match_threshold_w2v_tree", 0.75)
  lazy val featureGenerationMethod = props.getProperty("discourse.feature_generation_method", "standard")
  lazy val useSentenceDocID = StringUtils.getBool(props, "discourse.useSentenceDocID", false)
  lazy val onlyIntraSentence = StringUtils.getBool(props, "discourse.intrasentence", false)

  var modelWord2Vec = new Word2VecModel(props)

  def mkFeatures(answer: AnswerCandidate,
                 question: ProcessedQuestion,
                 externalFeatures: Option[Counter[String]],
                 errorPw:PrintWriter = null): (Counter[String], String) = {
    question match {
      case q:ProcessedQuestionSegments => (mkFeaturesCuspDispatch(answer, q), null)
      case _ => throw new RuntimeException ("DiscourseModelTree.mkFeatures(): question passed is not of type ProcessedQuestionSegments")
    }
  }

  def mkFeaturesCuspDispatch (answer:AnswerCandidate,
                              question:ProcessedQuestionSegments): Counter[String] = {

    val featuresParser = mkFeaturesTree(answer, question)
    val featuresWord2Vec = modelWord2Vec.mkFeatures(answer, question, None)._1
    var combined = featuresParser + featuresWord2Vec
    combined
  }


  def mkFeaturesTree(answer: AnswerCandidate,
                     question:ProcessedQuestionSegments): Counter[String] = {
    val features = new Counter[String]()

    // Step 1: IR Feature must be included
    // mihai: this is no longer needed if we use the hierarchical model
    if (props.getProperty("ranker.train_method", "incremental").toLowerCase != "hierarchical") {
      features.setCount("ir", answer.score)
    }

    // Step 1A: If we failed to properly segment the question, then return an empty feature list. Default back to other methods (e.g. IR)
    if (question.questionType == "failed") return features

    try {
      var tree:Node = null

      // Retrieve stored discourse tree from disk
      if (useSentenceDocID == true) {
        // sentence level granularity -- filename contains indicies for start and end of sentence
        val id = answer.doc.docid + "s" + answer.sentenceStart + "e" + answer.sentenceEnd
        tree = discReader.parse(id)
      } else {
        // document level granularity -- DocID is entire answer ID (e.g. stackoverflow, Y!A corpus)
        tree = discReader.parse(answer.doc.docid)
      }

      // Step 2: create features from each node in the discourse tree
      // logger.debug("Creating features based in discourse tree from question: " +
      // termFilter.extractValidLemmasFromSegment(question.segments.head))
      //      logger.debug("Answer docid: " + answer.doc.docid)
      //      logger.debug("Answer discourse tree:\n" + tree)

      val qSegLabel = question.segments.head.label
      val qSegLTN = question.segmentsLTN.head
      generateFeaturesFromTree(tree, question, features)
    } catch {
      case e:Exception => {
        logger.error("CAUGHT EXCEPTION: " + e)
      }
    }

    // System.exit(1)
    features
  }

  def generateFeaturesFromTree(tree:Node,
                               question: ProcessedQuestionSegments,
                               features:Counter[String]) {
    if(! tree.isTerminal) {

      if(! onlyIntraSentence || tree.isIntraSentence.get) {
        val left = tree.children(0)
        val right = tree.children(1)
        val label = tree.label

        //        logger.debug("Creating feature for discourse label: " + label)
        //        logger.debug("Left segment: " + left.toString(false))
        //        logger.debug("Right segment: " + right.toString(false))
        val qMatchSegs = question.segments
        val qMatchSegsLTN = question.segmentsLTN

        val w2vWordsQuestion = modelWord2Vec.mkWordsFromAnnotation( question.segments(0).doc )
        val w2vWordsLeft = modelWord2Vec.mkWordsFromAnnotation( left.text )
        val w2vWordsRight = modelWord2Vec.mkWordsFromAnnotation( right.text )

        // Find matches between question segments and left side of tree
        var leftMatches = new ArrayBuffer[(String, Double)]
        for (i <- 0 until qMatchSegs.size) {
          val currentSeg = qMatchSegs(i)
          // Original
          /*
          val score = segmentMatcher.scoreWithPrecomputedLTNVector(qMatchSegsLTN(i), left.text)
          if (score > minScoreThresh) {
            leftMatches.append ( (currentSeg.label, score) )
          }
          */
          // QSEG_W2V
          val scorew2v = modelWord2Vec.model.textSimilarity(w2vWordsQuestion, w2vWordsLeft)
          //println ("scorew2v: " + scorew2v + " (w2vLeft: " + w2vWordsLeft.toList + "  w2vWordsQuestion: " + w2vWordsQuestion.toList + ")")
          if (scorew2v > minScoreThreshW2V) {
            leftMatches.append ( (currentSeg.label + "W2V", scorew2v) )
          }
        }

        // Find matches between question segments and right side of tree
        var rightMatches = new ArrayBuffer[(String, Double)]
        for (i <- 0 until qMatchSegs.size) {
          val currentSeg = qMatchSegs(i)
          // Original
          /*
          val score = segmentMatcher.scoreWithPrecomputedLTNVector(qMatchSegsLTN(i), right.text)
          if (score > minScoreThresh) {
            rightMatches.append ( (currentSeg.label, score) )
          }
          */
          // QSEG_W2V
          val scorew2v = modelWord2Vec.model.textSimilarity(w2vWordsQuestion, w2vWordsRight)
          //println ("scorew2v: " + scorew2v + " (w2vRight: " + w2vWordsRight.toList + "  w2vWordsQuestion: " + w2vWordsQuestion.toList + ")")
          if (scorew2v > minScoreThreshW2V) {
            rightMatches.append ( (currentSeg.label + "W2V", scorew2v) )
          }
        }

        // -- Generate features based on matches --

        // Add "other" match to any side that did not have a positive match
        if (leftMatches.size == 0) leftMatches.append( ("OTHER", 0.0) )
        if (rightMatches.size == 0) rightMatches.append( ("OTHER", 0.0) )

        // Several different methods of feature combination are supported.
        var leftScore:Double = 0
        var rightScore:Double = 0
        var leftLabel:String = ""
        var rightLabel:String = ""
        if (featureGenerationMethod.toLowerCase == "standard") {
          // Standard feature generation: Generate a single feature with the left and right side containing all matches.
          // left
          for (i <- 0 until leftMatches.size) {
            leftScore += leftMatches(i)._2
            leftLabel += leftMatches(i)._1
          }
          leftScore /= leftMatches.size.toDouble

          // right
          for (i <- 0 until rightMatches.size) {
            rightScore += rightMatches(i)._2
            rightLabel += rightMatches(i)._1
          }
          rightScore /= rightMatches.size.toDouble

          // store the feature
          val feature = leftLabel + "-" + label + "-" + rightLabel
          val value = (leftScore + rightScore) / 2.0
          //          logger.debug("FEATURE: " + feature + " -> " + value)
          if(value > features.getCount(feature)) features.setCount(feature, value) // keep the maximum value found for a given feature

        } else if (featureGenerationMethod.toLowerCase == "expansion") {
          // TODO: Add combinatorial expansion-based generation of standard feature


        } else if (featureGenerationMethod.toLowerCase == "manytoone") {
          // Many to one: If a given side matches more than one question segment, then relabel that as a single multisegment-class feature (e.g. "MSEG")
          // left
          for (i <- 0 until leftMatches.size) {
            leftScore += leftMatches(i)._2
            leftLabel += leftMatches(i)._1
          }
          leftScore /= leftMatches.size.toDouble
          if (leftMatches.size > 1) leftLabel = "MULTI"

          // right
          for (i <- 0 until rightMatches.size) {
            rightScore += rightMatches(i)._2
            rightLabel += rightMatches(i)._1
          }
          rightScore /= rightMatches.size.toDouble
          if (rightMatches.size > 1) rightLabel = "MULTI"

          // store the feature
          val feature = leftLabel + "-" + label + "-" + rightLabel
          val value = (leftScore + rightScore) / 2.0
          logger.debug("FEATURE: " + feature + " -> " + value)
          if(value > features.getCount(feature)) features.setCount(feature, value) // keep the maximum value found for a given feature

        }
      }

      // Recurse down subtrees
      for(c <- tree.children) {
        generateFeaturesFromTree(c, question, features)
      }
    }
  }

  /*
  def generateFeaturesFromTreeOLD(tree:Node,
  qSegLabel:String,
  qSegLTN:Counter[String],
  features:Counter[String]) {
  if(! tree.isTerminal) {
  val left = tree.children(0)
  val right = tree.children(1)
  val label = tree.label

  logger.debug("Creating feature for discourse label: " + label)
  logger.debug("Left segment: " + left.toString(false))
  logger.debug("Right segment: " + right.toString(false))

  val leftScore = segmentMatcher.scoreWithPrecomputedLTNVector(qSegLTN, left.text)
  val leftLabel = if(leftScore > minScoreThresh) qSegLabel else "OTHER"
  val rightScore = segmentMatcher.scoreWithPrecomputedLTNVector(qSegLTN, right.text)
  val rightLabel = if(rightScore > minScoreThresh) qSegLabel else "OTHER"

  val feature = leftLabel + "-" + label + "-" + rightLabel
  val value = (leftScore + rightScore) / 2.0
  logger.debug("FEATURE: " + feature + " -> " + value)

  //features.incrementCount(feature, value)
  // keep just the max
  if(value > features.getCount(feature)) {
  features.setCount(feature, value)
  }

  for(c <- tree.children) {
  generateFeaturesFromTree(c, qSegLabel, qSegLTN, features)
  }
  }
  }
  */

  def mkProcessedQuestion(question:Document):Array[ProcessedQuestion] = {
    if (props.getProperty("discourse.question_processor").toUpperCase == "ONESEG") return Array[ProcessedQuestion](qProcessor.mkProcessedQuestionOneArgument(question))
    if (props.getProperty("discourse.question_processor").toUpperCase == "SVO") return Array[ProcessedQuestion]( qProcessor.mkProcessedQuestionSVO(question) )
    if (props.getProperty("discourse.question_processor").toUpperCase == "HYBRIDSVO") {
      val processedQuestions = new ArrayBuffer[ProcessedQuestion]()
      processedQuestions.append (qProcessor.mkProcessedQuestionOneArgument(question)) // Add ONESEG by default
      processedQuestions.append (qProcessor.mkProcessedQuestionSVO(question)) // Also use Subj/Verb/Obj/IndObj
      return processedQuestions.toArray
    }

    // default to QSEG if method is unknown
    return Array[ProcessedQuestion](qProcessor.mkProcessedQuestionOneArgument(question))

    /*
// default to QSEG
return Array[ProcessedQuestion](qProcessor.mkProcessedQuestionOneArgument(question))
*/

  }

}

object DiscourseModelTreeQSEGW2V {
  val logger = LoggerFactory.getLogger(classOf[DiscourseModelTreeQSEGW2V])
}

