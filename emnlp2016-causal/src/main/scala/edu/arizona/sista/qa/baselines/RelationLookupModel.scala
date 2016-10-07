package edu.arizona.sista.qa.baselines

import java.io.PrintWriter
import java.util.Properties

import edu.arizona.sista.processors.Document
import edu.arizona.sista.processors.fastnlp.FastNLPProcessor
import edu.arizona.sista.qa.index.TermFilter
import edu.arizona.sista.qa.ranking.{ProcessedQuestionSegments, ProcessedQuestion, RankingModel}
import edu.arizona.sista.qa.retrieval.AnswerCandidate
import edu.arizona.sista.qa.segmenter.QuestionProcessor
import edu.arizona.sista.qa.translation.TransView
import edu.arizona.sista.qa.word2vec.{Word2VecRelationModel, Word2VecRelation}
import edu.arizona.sista.struct.Counter
import org.slf4j.LoggerFactory
import edu.arizona.sista.utils.StringUtils


import scala.collection.mutable.ArrayBuffer

/**
  * Created by bsharp on 4/27/16.
  */
class RelationLookupModel(props:Properties) extends RankingModel {
  val qProcessor = new QuestionProcessor(props)
  val directory = props.getProperty("relation_lookup.input_directory", "")
  val ext = props.getProperty("relation_lookup.input_extension", "")
  val lenThreshold = StringUtils.getInt(props, "relation_lookup.len_threshold", 0)

  val model = new RelationLookup(directory, ext, lenThreshold)
  lazy val processor = new FastNLPProcessor()
  val termFilter = new TermFilter
  val viewName = props.getProperty("view.view", "words_content")
  //TODO: right now, only words are supported... (in the end to end sense)

  def mkFeatures( answer:AnswerCandidate,
                  question:ProcessedQuestion,
                  externalFeatures:Option[Counter[String]],
                  errorPw:PrintWriter = null): (Counter[String], String) = {

    question match {
      case q:ProcessedQuestionSegments => (mkFeaturesCuspDispatch (answer, q), null)
      case _ => throw new RuntimeException ("Word2VecModel.mkFeatures(): question passed is not of type ProcessedQuestionSegments")
    }

  }

  def mkFeaturesCuspDispatch (answer:AnswerCandidate,
                              question:ProcessedQuestionSegments): Counter[String] = {
    return mkFeaturesRelationLookup(answer, question)
  }

  def mkProcessedQuestion(question:Document):Array[ProcessedQuestion] = {
    // default to QSEG if method is unknown
    return Array[ProcessedQuestion](qProcessor.mkProcessedQuestionOneArgument(question))
  }

  def mkFeaturesRelationLookup( answer:AnswerCandidate,
                                  question:ProcessedQuestionSegments): Counter[String] = {

    val features = new Counter[String]()

    // Add basic IR feature
    if (props.getProperty("ranker.train_method", "incremental").toLowerCase != "hierarchical") {
      features.setCount("ir", answer.score)
    }

    // Step 1A: If we failed to properly segment the question, then return an empty feature list.  Default back to other methods (e.g. IR)
    if (question.questionType == "failed") return features

    // Step 2: Locate words in question and answer
    val q = if (viewName.contains("dep") || viewName.contains("trans")) {
      docAnnotations(question.annotation)
    } else {
      question.annotation
    }

    val a = if (viewName.contains("dep") || viewName.contains("trans")) {
      docAnnotations(answer.annotation)
    } else {
      answer.annotation
    }

    val QView = new TransView(q)
    val AView = new TransView(a)

    QView.makeView(viewName)
    AView.makeView(viewName)

    val questionTokens = QView.features.toArray
    val answerTokens = AView.features.toArray

    def checkRange(dbl: Double) = {
      if (dbl.isNaN || dbl == Double.MinValue || dbl == Double.MaxValue) 0.0 else dbl
    }

    // Determine whether the question contains a cause or an effect (i.e. is the question the target or
    // context language?)
    val qText = q.sentences.map(_.getSentenceText())
    val map = RelationLookupModel.selectMap(qText, model)

    // Step 3: Determine word2vec scores
    var scoreTextMatches = checkRange(model.textMatches(questionTokens, answerTokens, map))
    if (scoreTextMatches > 100) {
      scoreTextMatches = 1.0
    }
    else scoreTextMatches = 0.0
    //scoreTextMatches = Math.log(scoreTextMatches + 1)
    features.setCount("REL_LOOKUP_TEXTSIM", scoreTextMatches)

//    val scoreAvgMatches = checkRange(model.averageMatches(questionTokens, answerTokens, map))
//    features.setCount("REL_LOOKUP_AVGSIM", scoreAvgMatches)
//
//    val scoreMaxMatches = checkRange(model.maxMatches(questionTokens, answerTokens, map))
//    features.setCount("REL_LOOKUP_MAXSIM", scoreMaxMatches)

    // I am excluding MIN because everything will have a min of 0

    // return list of features
    features
  }

  /*
    def mkWordsFromQuestion(question:ProcessedQuestionSegments):Array[String] = {
      val out = new ArrayBuffer[String]
      mkWordsFromAnnotation( question.segments(0).doc )
    }
  */

  def mkWordsFromAnnotation(annotation:Document):Array[String] = {
    val out = new ArrayBuffer[String]
    for (sent <- annotation.sentences) {
      for (word <- sent.words) {
        out += word
      }
    }
    out.toArray
  }
  // Helper method for annotation
  def docToString(doc:Document):String = {
    // Transform a Document
    val textBuffer = new ArrayBuffer[String]
    for (sentence <- doc.sentences) {
      var text = sentence.words.mkString(" ")
      //TODO: Remove final spaces
      textBuffer += text
    }
    textBuffer.mkString(" ")
  }
  // Helper method for annotation
  def docAnnotations(doc:Document):Document = {
    val text = docToString(doc)
    val newDoc = processor.annotate(text)
    newDoc
  }




}

object RelationLookupModel {
  val logger = LoggerFactory.getLogger(classOf[RelationLookupModel])

  // Based on the text of the question, select whether the question text is from the target or source vocabulary
  // Returns the (questionMatrix, answerMatrix, questionMatDims, answerMatDims)
  def selectMap(qText:Array[String],
                model:RelationLookup): Map[String, Counter[String]] = {
    val qHead = qText.head

    // Word-based heuristics

    for (cRegex <- Word2VecRelationModel.causeRegexes) {
      if (qHead.toLowerCase.matches(cRegex)) {
        return model.contextMap
      }
    }
    for (rRegex <- Word2VecRelationModel.resultRegexes) {
      if (qHead.toLowerCase.matches(rRegex)) {
        return model.targetMap
      }
    }

    println (s"**WARNING: Question didn't match causal patterns: ${qText.mkString(" ")}")
    println (s"** Question head: $qHead")
    println ("**Defaulting to: Question=Cause, Answer=Effect")

    model.targetMap
  }
}
