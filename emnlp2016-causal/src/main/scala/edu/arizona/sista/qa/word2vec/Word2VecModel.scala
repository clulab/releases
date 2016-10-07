package edu.arizona.sista.qa.word2vec

import java.util.Properties
import edu.arizona.sista.qa.ranking.{ProcessedQuestionSegments, ProcessedQuestion, RankingModel}
import edu.arizona.sista.qa.segmenter.QuestionProcessor
import edu.arizona.sista.qa.retrieval.AnswerCandidate
import edu.arizona.sista.struct.Counter
import edu.arizona.sista.processors.Document
import org.slf4j.LoggerFactory
import collection.mutable.ArrayBuffer
import java.io.PrintWriter
import edu.arizona.sista.qa.index.TermFilter
import edu.arizona.sista.processors.corenlp.CoreNLPProcessor
import edu.arizona.sista.processors.fastnlp.FastNLPProcessor
import edu.arizona.sista.qa.translation.TransView
import edu.arizona.sista.utils.{StringUtils, FrequencyFile}


/**
 * Created with IntelliJ IDEA.
 * User: peter
 * Date: 12/5/13
 */
class Word2VecModel(props:Properties) extends RankingModel {
  val existingFrequencyFile = StringUtils.getStringOption(props, "w2v.filtering.existing_frequency_file")
  val numWordsToUse = StringUtils.getIntOption(props, "w2v.filtering.num_words")
  val minCount = StringUtils.getIntOption(props, "w2v.filtering.min_count")
  val existingWordSet: Option[Set[String]] = existingFrequencyFile.map(filename => FrequencyFile.parseFrequencyFile(filename, numWordsToUse, minCount))

  val indexDir = props.getProperty("index")
  val qProcessor = new QuestionProcessor(props)
  lazy val vectorFilename = props.getProperty("word2vec.vectors", "")
  lazy val model = new Word2vec(vectorFilename, existingWordSet)
  lazy val processor = new FastNLPProcessor()
  val termFilter = new TermFilter


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
    return mkFeaturesWord2Vec(answer, question)
  }

  def mkProcessedQuestion(question:Document):Array[ProcessedQuestion] = {
    // default to QSEG if method is unknown
    return Array[ProcessedQuestion](qProcessor.mkProcessedQuestionOneArgument(question))
  }


  def mkFeaturesWord2Vec( answer:AnswerCandidate,
                             question:ProcessedQuestionSegments): Counter[String] = {

    val features = new Counter[String]()
    val viewName = props.getProperty("view.view", "words_content")

    // Add basic IR feature
    if (props.getProperty("ranker.train_method", "incremental").toLowerCase != "hierarchical") {
      features.setCount("ir", answer.score)
    }

    // Step 1A: If we failed to properly segment the question, then return an empty feature list.  Default back to other methods (e.g. IR)
    if (question.questionType == "failed") return features

    // Step 2: Locate words in question and answer
    //val wordsQuestion = mkWordsFromAnnotation( question.segments(0).doc )   // TODO:  I think this is safe, but double check?  Or create a new type of ProcessedQuestion for word2vec cases
    //val wordsAnswer = mkWordsFromAnnotation( answer.doc.annotation )
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

    // map view name to (Question tokens, Answer tokens), so that we can lookup tokens by model's view

    val QView = new TransView(q)
    val AView = new TransView(a)

    QView.makeView(viewName)
    AView.makeView(viewName)
    
    val questionTokens = QView.features.toArray
    val answerTokens = AView.features.toArray

    def checkRange(dbl: Double) = {
      if (dbl.isNaN || dbl == Double.MinValue || dbl == Double.MaxValue) 0.0 else dbl
    }

    // Step 3: Determine word2vec scores
    val scoreTextSimilarity = checkRange(model.textSimilarity(questionTokens, answerTokens))
    features.setCount("WORD2VEC_TEXTSIM", scoreTextSimilarity)

    val scoreAvgSimilarity = checkRange(model.avgSimilarity(questionTokens, answerTokens))
    features.setCount("WORD2VEC_AVGSIM", scoreAvgSimilarity)

    val scoreMaxSimilarity = checkRange(model.maxSimilarity(questionTokens, answerTokens))
    features.setCount("WORD2VEC_MAXSIM", scoreMaxSimilarity)

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


object Word2VecModel {
  val logger = LoggerFactory.getLogger(classOf[Word2VecModel])
}
