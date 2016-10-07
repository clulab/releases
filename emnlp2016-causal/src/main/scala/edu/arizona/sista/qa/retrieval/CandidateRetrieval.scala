package edu.arizona.sista.qa.retrieval

import java.util.Properties
import edu.arizona.sista.qa.index.TermFilter
import edu.arizona.sista.utils.StringUtils
import edu.arizona.sista.processors.Document

/**
 * Retrieves and scores candidate answers
 * User: mihais
 * Date: 3/14/13
 */
class CandidateRetrieval(props:Properties) {
  val sentencesPerAnswer = getSentencesPerAnswer(props, "retrieval.candidate_sizes")

  val paragraphsPerAnswer = getParagraphsPerAnswer(props, "retrieval.candidate_sizes", "1p")

  val indexDir = props.getProperty("index")

  val maxDocs = StringUtils.getInt(props, "retrieval.max_docs", 100)

  val docWeight = StringUtils.getDouble(props, "retrieval.doc_weight", 0.6)
  val docSyntaxWeight = StringUtils.getDouble(props, "retrieval.doc_syntax_weight", 0.0)
  val answerSyntaxWeight = StringUtils.getDouble(props, "retrieval.answer_syntax_weight", 0.0)

  val termFilter = new TermFilter

  var useWordsInsteadOfLemmas = StringUtils.getBool(props, "retrieval.wordsInsteadOfLemmas", false)

  def getSentencesPerAnswer(props:Properties, name:String):Int = {
    val v = props.getProperty(name)
    if (v == null) return -1
    if (! v.endsWith("s")) return -1
    v.substring(0, v.length - 1).toInt
  }

  def getParagraphsPerAnswer(props:Properties, name:String, default:String):Int = {
    val v = props.getProperty(name, default)
    if (! v.endsWith("p")) return -1
    v.substring(0, v.length - 1).toInt
  }

  def usesParagraphGranularity:Boolean = (sentencesPerAnswer <= 0 && paragraphsPerAnswer > 0)

  def mkAnswerBuilder:AnswerBuilder = {
    if (sentencesPerAnswer > 0)
      return new SentenceAnswerBuilder(sentencesPerAnswer)
    if (paragraphsPerAnswer > 0)
      return new ParagraphAnswerBuilder(paragraphsPerAnswer)
    throw new RuntimeException("ERROR: do not know how to construct an AnswerBuilder!")
  }

  def retrieve(queryAnnotation:Document): List[AnswerCandidate] = {
    var documentRetriever:DocumentRetrieval = null
    if(docSyntaxWeight == 0.0) {
      // no syntax; just use the BOW model
      documentRetriever =
        new BagOfWordsDocumentRetrieval(
          indexDir,
          maxDocs,
          termFilter,
          queryAnnotation,
          useWordsInsteadOfLemmas)
    } else {
      // meta model combines BOW with syntax
      documentRetriever =
        new MetaDocumentRetrieval(
          indexDir,
          maxDocs,
          termFilter,
          queryAnnotation,
          docSyntaxWeight)
    }

    val documents = documentRetriever.retrieve

    val answerBuilder = mkAnswerBuilder
    val answers = answerBuilder.mkAnswerCandidates(documents)

    var paragraphScorer:PassageScorer = null
    if(answerSyntaxWeight == 0.0) {
      // no syntax; just use the BOW model
      paragraphScorer =
        new BagOfWordsPassageScorer(
          termFilter,
          indexDir,
          queryAnnotation)
    } else {
      // meta model combines BOW with syntax
      paragraphScorer =
        new MetaPassageScorer(
          termFilter,
          indexDir,
          queryAnnotation,
          answerSyntaxWeight)
    }

    for (answer <- answers) {
      val answerScore = paragraphScorer.score(answer)
      answer.setScore(answerScore, docWeight)
    }

    answers.sortBy(- _.score)
  }
}
