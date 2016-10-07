package edu.arizona.sista.qa.discourse

import org.slf4j.LoggerFactory
import edu.arizona.sista.qa.index.TermFilter
import edu.arizona.sista.qa.segmenter.{SegmentMatcherBOW, QuestionProcessor}
import edu.arizona.sista.qa.discparser.{Node, DiscReader}
import edu.arizona.sista.utils.StringUtils
import edu.arizona.sista.qa.retrieval.AnswerCandidate
import edu.arizona.sista.qa.ranking.{RankingModel, ProcessedQuestionSegments, ProcessedQuestion}
import edu.arizona.sista.struct.Counter
import java.util.Properties
import edu.arizona.sista.processors.Document
import DiscourseModelTreeKernel.logger
import java.io.PrintWriter

/**
 *
 * User: mihais
 * Date: 8/9/13
 */
class DiscourseModelTreeKernel(val props:Properties) extends RankingModel {
  lazy val termFilter = new TermFilter()
  lazy val qProcessor = new QuestionProcessor(props)
  lazy val discReader = new DiscReader(props.getProperty("discourse.dir"))
  lazy val indexDir = props.getProperty("index")
  lazy val segmentMatcher = new SegmentMatcherBOW(termFilter, indexDir)
  lazy val minScoreThresh = StringUtils.getDouble(props, "discourse.match_threshold", 0.0)

  override def usesKernels:Boolean = true

  def mkFeatures(answer: AnswerCandidate,
                 question: ProcessedQuestion,
                 externalFeatures: Option[Counter[String]],
                 errorPw:PrintWriter = null): (Counter[String], String) = {
    question match {
      case q:ProcessedQuestionSegments => mkFeaturesTree (answer, q)
      case _ => throw new RuntimeException ("DiscourseModelTree.mkFeatures(): question passed is not of type ProcessedQuestionSegments")
    }
  }

  def mkFeaturesTree(answer: AnswerCandidate,
                     question:ProcessedQuestionSegments): (Counter[String], String) = {
    val features = new Counter[String]()
    var kernel = "(ROOT )"

    // Step 1: IR Feature must be included
    // mihai: this is no longer needed if we use the hierarchical model
    if (props.getProperty("ranker.train_method", "incremental").toLowerCase != "hierarchical") {
      features.setCount("ir", answer.score)
    }

    // Step 1A: If we failed to properly segment the question, then return an empty feature list.  Default back to other methods (e.g. IR)
    // this should not happen!
    if (question.questionType == "failed") return (features, kernel)

    try {
      val tree = discReader.parse(answer.doc.docid)

      // Step 2: create features from each node in the discourse tree
      logger.debug("Creating features based in discourse tree from question: " +
        termFilter.extractValidLemmasFromSegment(question.segments.head))
      logger.debug("Answer docid: " + answer.doc.docid)
      logger.debug("Answer discourse tree:\n" + tree)

      val qSegLabel = question.segments.head.label
      val qSegLTN = question.segmentsLTN.head
      val os = new StringBuilder
      generateTreeKernel(os, tree, qSegLabel, qSegLTN)
      kernel = os.toString()
      logger.debug("Kernelized datum: " + kernel)

    } catch {
      case e:Exception => {
        logger.error("CAUGHT EXCEPTION: " + e)
      }
    }

    // System.exit(1)
    (features, kernel)
  }

  def generateTreeKernel(os:StringBuilder,
                         tree:Node,
                         qSegLabel:String,
                         qSegLTN:Counter[String]) {
    if(tree.isTerminal) {
      val score = segmentMatcher.scoreWithPrecomputedLTNVector(qSegLTN, tree.text)
      val label = if(score > minScoreThresh) qSegLabel else "OTHER"
      os.append(s"($label *)")
    } else {
      os.append("(")
      os.append(tree.label)
      for(c <- tree.children) {
        os.append(" ")
        generateTreeKernel(os, c, qSegLabel, qSegLTN)
      }
      os.append(")")
    }
  }

  def mkProcessedQuestion(question:Document):Array[ProcessedQuestion] = {
    // default to QSEG
    return Array[ProcessedQuestion](qProcessor.mkProcessedQuestionOneArgument(question))
  }
}

object DiscourseModelTreeKernel {
  val logger = LoggerFactory.getLogger(classOf[DiscourseModelTree])
}
