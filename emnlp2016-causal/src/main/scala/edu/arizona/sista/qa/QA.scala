package edu.arizona.sista.qa

//import discourse.DiscourseModelCusp
import java.util.Properties
import ranking.RankerVoting
import retrieval.{AnswerCandidate, CandidateRetrieval}
import collection.mutable
import edu.arizona.sista.utils.StringUtils
import scorer._
import collection.mutable.{ArrayBuffer, ListBuffer}
import nu.xom.{Attribute, Element}
import org.slf4j.LoggerFactory
import QA.logger
import edu.arizona.sista.processors.{Document, Processor}
import edu.arizona.sista.processors.corenlp.CoreNLPProcessor
import java.io.PrintWriter

/**
 * Main entry point for the QA system
 * User: mihais
 * Date: 3/14/13
 */
class QA (props:Properties) {
  lazy val queryProcessor:Processor = new CoreNLPProcessor()

  var candidateRetrieval = new CandidateRetrieval(props)

  def answer(question:String, maxAnswers:Int = -1): List[AnswerCandidate] = {
    logger.debug("Answering question: {}", question)
    answer(queryProcessor.annotate(question), maxAnswers)
  }


  def answerXML(question:String, maxAnswers:Int):nu.xom.Document = {
    val answers = answer(question, maxAnswers)
    toXML(answers)
  }

  def answer(queryAnnotation:Document, maxAnswers:Int): List[AnswerCandidate] = {
    //
    // the unsupervised IR model
    //
    val answers = candidateRetrieval.retrieve(queryAnnotation)

    //
    // post processing of answers
    //
    val ppAnswers = postProcessing(answers)

    // keep only the top maxAnswers
    if(maxAnswers > 0) return ppAnswers.slice(0, math.min(maxAnswers, ppAnswers.size))

    ppAnswers
  }

  def postProcessing(rawAnswers:List[AnswerCandidate]):List[AnswerCandidate] = {
    //
    // expand to paragraph boundaries (up and down) and remove duplicates
    //
    val expandToParagraphs = StringUtils.getBool(props, "retrieval.use_paragraph_expansion", true)
    var expandedAnswers = rawAnswers
    if (expandToParagraphs && ! candidateRetrieval.usesParagraphGranularity) {
      expandedAnswers = expandToParagraphBoundaries(rawAnswers)
      expandedAnswers = removeDuplicatedAnswers(expandedAnswers)
    }
    expandedAnswers
  }

  /** Converts the extracted answers to XML; useful for the QA servlet */
  def toXML(answers:List[AnswerCandidate], maxResults:Int = -1):nu.xom.Document = {
    val root = new Element("root", QA.NAMESPACE_URI)
    val xmlDoc = new nu.xom.Document(root)

    val answersElem = new Element("answers", QA.NAMESPACE_URI)
    root.appendChild(answersElem)
    var answerCount = 0
    for (answer <- answers) {
      if (maxResults < 0 || answerCount < maxResults) {
        answersElem.appendChild(answerToXML(answer))
      }
      answerCount += 1
    }

    xmlDoc
  }

  private def answerToXML(answer:AnswerCandidate):Element = {
    val answerElem = new Element("answer", QA.NAMESPACE_URI)
    answerElem.addAttribute(new Attribute("docid", answer.doc.docid))
    var i = answer.sentenceStart
    while(i < answer.sentenceEnd) {
      val words = answer.doc.annotation.sentences(i).words
      val sentElem = new Element("sentence", QA.NAMESPACE_URI)
      sentElem.addAttribute(new Attribute("offset", Integer.toString(i)))
      sentElem.appendChild(words.mkString(" "))
      answerElem.appendChild(sentElem)
      i += 1
    }
    answerElem
  }

  def expandToParagraphBoundaries(rawAnswers:List[AnswerCandidate]):List[AnswerCandidate] = {
    val ab = new ListBuffer[AnswerCandidate]
    for (ra <- rawAnswers) ab += expandToParagraphBoundaries(ra)
    ab.toList
  }

  /** Expands one answer to the left and right to the boundaries of paragraphs */
  def expandToParagraphBoundaries(rawAnswer:AnswerCandidate):AnswerCandidate = {
    val pars = rawAnswer.doc.paragraphs
    var start = rawAnswer.sentenceStart
    val startPara = pars(start)
    while(start >= 0 && pars(start) == startPara) start -= 1
    start += 1
    var end = rawAnswer.sentenceEnd - 1
    val endPara = pars(end)
    while(end < pars.length && pars(end) == endPara) end += 1
    new AnswerCandidate(rawAnswer.doc, start, end, rawAnswer.answerScore, rawAnswer.score)
  }

  def removeDuplicatedAnswers(answers:List[AnswerCandidate]):List[AnswerCandidate] = {
    val uniques = new mutable.HashSet[AnswerCandidate]
    // must traverse them in order to make sure we can keep the ones with the highest scores
    for (answer <- answers) {
      if (! uniques.contains(answer)) {
        uniques += answer
      }
    }
    // sort in descending order of scores
    uniques.toList.sortBy(- _.score)
  }
}

object QA {
  val logger = LoggerFactory.getLogger(classOf[QA])
  val NAMESPACE_URI = null

  def main(args:Array[String]) {
    val props = StringUtils.argsToProperties(args)

    if(props.containsKey("tune-no-syntax")) {
      lineSearchNoSyntax(props)
    } else if(props.containsKey("tune-syntax")) {
      lineSearchWithSyntax(props)
    } else if (! props.containsKey("gold")) {
      val qa = new QA(props)
      shell(qa, StringUtils.getInt(props, "shell.max_answers", 5))
    } else {
      val questionReader = new QuestionParser
      val questions = questionReader.parse(props.getProperty("gold"))
      val qa = new QA(props)
      val (p1s, mrrs, p1p, mrrp) = eval(qa, questions)
      // val (p1s, mrrs, p1p, mrrp) = avgQA(props, questions)
      logger.info("Overall scores: (sentence: P@1 " + p1s +
        ", MRR " + mrrs + ") (paragraph: P@1 " + p1p +
        ", MRR " + mrrp + ")")
      logger.info (" ***** Summary (retrieval.candidate_sizes=" +
        props.getProperty("retrieval.candidate_sizes") +
        "   .doc_freq_enabled=" +
        StringUtils.getBool(props, "retrieval.use_doc_freq", true) +
        "   .paragraph_expansion=" +
        StringUtils.getBool(props, "retrieval.use_paragraph_expansion", true) + " )")
    }
  }

  def lineSearchWithSyntax(props:Properties) {
    val questionReader = new QuestionParser
    val questions = questionReader.parse(props.getProperty("gold"))

    props.put("retrieval.use_paragraph_expansion", "false")
    props.put("retrieval.max_docs", "20")
    val state = new LineSearchWithSyntaxState()

    var downhill = false
    var prev:Double = -1.0
    for(docWeight <- 0.1 to 1.0 by 0.1) { // if(! downhill)) {
      props.put("retrieval.doc_weight", docWeight.toString)

      val score = lineSearchDocSyntaxWeight(props, questions, state)

      if(score < prev) {
        downhill = true
        logger.info("LINE SEARCH: found downhill for retrieval.doc_weight: " +
          score + " < " + prev)
      } else {
        prev = score
      }
    }

    logger.info("LINE SEARCH: final best score " + state.max +
      " for retrieval.doc_weight == " + state.docWeight +
      " retrieval.doc_syntax_weight == " + state.docSyntaxWeight +
      " retrieval.answer_syntax_weight == " + state.answerSyntaxWeight)
  }

  def lineSearchDocSyntaxWeight(
    props:Properties,
    questions:List[Question],
    state:LineSearchWithSyntaxState):Double = {
    var downhill = false
    var prev:Double = -1.0
    var max:Double = -1.0
    for(docSyntaxWeight <- 0.0 to 1.0 by 0.1) { // if(! downhill)) {
      props.put("retrieval.doc_syntax_weight", docSyntaxWeight.toString)

      val score = lineSearchAnswerSyntaxWeight(props, questions, state)

      if(score < prev) {
        downhill = true
        logger.info("LINE SEARCH: found downhill for retrieval.doc_syntax_weight: " +
          score + " < " + prev)
      } else {
        prev = score
        if(score > max) {
          max = score
        }
      }
    }
    max
  }

  def lineSearchAnswerSyntaxWeight(
    props:Properties,
    questions:List[Question],
    state:LineSearchWithSyntaxState):Double = {

    val docWeight = props.getProperty("retrieval.doc_weight").toDouble
    val docSyntaxWeight = props.getProperty("retrieval.doc_syntax_weight").toDouble

    var downhill = false
    var prev:Double = -1.0
    var max:Double = -1.0
    for(ansSyntaxWeight <- 0.0 to 1.0 by 0.1 if(! downhill)) {
      props.put("retrieval.answer_syntax_weight", ansSyntaxWeight.toString)

      val (sp1, smmr, pp1, pmrr) = avgQA(props, questions)
      logger.info("LINE SEARCH SCORE FOR retrieval.doc_weight == " + docWeight +
        " retrieval.doc_syntax_weight == " + docSyntaxWeight +
        " retrieval.answer_syntax_weight == " + ansSyntaxWeight +
        " is: " + (sp1, smmr, pp1, pmrr))

      if(sp1 < prev) {
        downhill = true
        logger.info("LINE SEARCH: found downhill for retrieval.answer_syntax_weight: " +
          sp1 + " < " + prev)
      } else {
        prev = sp1
        if(sp1 > max) {
          max = sp1
        }
        if(sp1 > state.max) {
          state.max = sp1
          state.docWeight = docWeight
          state.docSyntaxWeight = docSyntaxWeight
          state.answerSyntaxWeight = ansSyntaxWeight

          logger.info("LINE SEARCH: found best score " + state.max +
            " for retrieval.doc_weight == " + state.docWeight +
            " retrieval.doc_syntax_weight == " + state.docSyntaxWeight +
            " retrieval.answer_syntax_weight == " + state.answerSyntaxWeight)
        }
      }
    }
    max
  }

  def lineSearchNoSyntax(props:Properties) {
    val questionReader = new QuestionParser
    val questions = questionReader.parse(props.getProperty("gold"))

    props.put("retrieval.doc_syntax_weight", "0")
    props.put("retrieval.answer_syntax_weight", "0")
    props.put("retrieval.use_paragraph_expansion", "false")
    var maxSentencePrecision = 0.0
    var bestDocWeight = -1.0
    for(docWeight <- 0.0 to 1.0 by 0.1) {
      props.put("retrieval.doc_weight", docWeight.toString)
      val (sp1, smmr, pp1, pmrr) = avgQA(props, questions)
      logger.info("LINE SEARCH SCORE FOR retrieval.doc_weight == " + docWeight + " is: " + (sp1, smmr, pp1, pmrr))
      if(sp1 > maxSentencePrecision) {
        maxSentencePrecision = sp1
        bestDocWeight = docWeight
      }
    }
    logger.info("LINE SEARCH: found best retrieval.doc_weight == " + bestDocWeight + " with the best P@1 == " + maxSentencePrecision)
  }

  def avgQA(props:Properties, questions:List[Question]):(Double,Double,Double,Double) = {
    var sumSp1 = 0.0
    var sumSmrr = 0.0
    var sumPp1 = 0.0
    var sumPmrr = 0.0
    val sizes = List("3s", "4s", "5s", "6s")
    for(candSize <- sizes) {
      props.put("retrieval.candidate_sizes", candSize)
      val qa = new QA(props)
      val (sp1, smrr, pp1, pmrr) = eval(qa, questions)
      sumSp1 += sp1
      sumSmrr += smrr
      sumPp1 += pp1
      sumPmrr += pmrr
    }
    (sumSp1/sizes.size, sumSmrr/sizes.size, sumPp1/sizes.size, sumPmrr/sizes.size)
  }

  def shell(qa:QA, maxAnswers:Int) {
    var ok = true
    logger.info("Starting shell...")
    while(ok) {
      print("QA> ")
      val q = readLine()
      if (q != null) {
        val answers = qa.answer(q)
        var i = 1
        for(answer <- answers) {
          if (i <= maxAnswers) {
            println("Answer #" + i + ":")
            println(answer)
          }
          i += 1
        }
      } else {
        ok = false
      }
    }
  }

  def eval(qa:QA, questions:List[Question]): (Double,Double,Double,Double) = {
    var avgScoreSetSyntacticDep = new Scores()
    var avgScoreSetOracle20 = new Scores()
    var avgScoreSetOracle100 = new Scores()
    var avgScoreSetOracle1000 = new Scores()
    var avgSentF120:Double = 0
    var avgParaF120:Double = 0

    logger.info("Found " + questions.size + " questions.")

    for (i <- 0 until questions.size) {
      val question = questions(i)
      logger.info ("questions[" + i + "] : " + question.text)

      // TODO: This filename should come from the properties file
      val analysisFile:String = "analysis_out.txt"
      logger.info ("Opening report file... (filename = " + analysisFile + " )")
      val pw = new PrintWriter(analysisFile)
      pw.println ("------")
      pw.flush()

      pw.println("================================================================================")
      pw.println(" Question[" + i + "] : + " + question.text )

      pw.println("================================================================================")
      pw.println("================================================================================")
      pw.println(" Question[" + i + "] : + " + question.text)

      // answer
      val candAnswers = qa.answer(question.text)

      // score
      val oneScorer = new Scorer()
      val scoreSetSyntacticDep = oneScorer.computeScores(question, candAnswers)
      oneScorer.analysis (question, candAnswers, scoreSetSyntacticDep, pw, i)
      val scoreSetOracle20 = oneScorer.computeScoresOracle(question, candAnswers, 20)
      val scoreSetOracle100 = oneScorer.computeScoresOracle(question, candAnswers, 100)
      val scoreSetOracle1000 = oneScorer.computeScoresOracle(question, candAnswers, 1000)

      avgScoreSetSyntacticDep.addToAverage(scoreSetSyntacticDep)
      avgScoreSetOracle20.addToAverage(scoreSetOracle20)
      avgScoreSetOracle100.addToAverage(scoreSetOracle100)
      avgScoreSetOracle1000.addToAverage(scoreSetOracle1000)

      val (sentF120, paraF120) = oneScorer.computeF1Subset(question, candAnswers, 20)
      avgSentF120 += sentF120
      avgParaF120 += paraF120

    }

    avgSentF120 = avgSentF120 / questions.size
    avgParaF120 = avgParaF120 / questions.size

    // Display summary statistics
    logger.info (" ***** Summary Average Baseline (Percent) (sentence: P@1:" + avgScoreSetSyntacticDep.sent.overlapAt1*100 +
      "  MRR:" + avgScoreSetSyntacticDep.sent.overlapMRR*100 +
      ")  (paragraph: P@1:" + avgScoreSetSyntacticDep.para.overlapAt1*100 +
      "  MRR:" + avgScoreSetSyntacticDep.para.overlapMRR*100)

    // Display summary statistics (oracle N = 20)
    logger.info (" ***** Summary Average Oracle (N=20) (Percent) (sentence: P@1:" + avgScoreSetOracle20.sent.overlapAt1*100 +
      "  MRR:" + avgScoreSetOracle20.sent.overlapMRR*100 +
      ")  (paragraph: P@1:" + avgScoreSetOracle20.para.overlapAt1*100 +
      "  MRR:" + avgScoreSetOracle20.para.overlapMRR*100)

    // Display summary statistics (F1 scores only over first 20 candidate answers)
    logger.info (" ***** Summary Average F1 (N=20)  Sentence:" + avgSentF120 + "   Paragraph:" + avgParaF120 )

    // Display summary statistics (oracle N = 100)
    logger.info (" ***** Summary Average Oracle (N=100) (Percent) (sentence: P@1:" + avgScoreSetOracle100.sent.overlapAt1*100 +
      "  MRR:" + avgScoreSetOracle100.sent.overlapMRR*100 +
      ")  (paragraph: P@1:" + avgScoreSetOracle100.para.overlapAt1*100 +
      "  MRR:" + avgScoreSetOracle100.para.overlapMRR*100)

    // Display summary statistics (oracle N = 1000)
    logger.info (" ***** Summary Average Oracle (N=1000) (Percent) (sentence: P@1:" + avgScoreSetOracle1000.sent.overlapAt1*100 +
      "  MRR:" + avgScoreSetOracle1000.sent.overlapMRR*100 +
      ")  (paragraph: P@1:" + avgScoreSetOracle1000.para.overlapAt1*100 +
      "  MRR:" + avgScoreSetOracle1000.para.overlapMRR*100)

    // Return scores
    (avgScoreSetSyntacticDep.sent.overlapAt1, avgScoreSetSyntacticDep.sent.overlapMRR,
      avgScoreSetSyntacticDep.para.overlapAt1, avgScoreSetSyntacticDep.para.overlapMRR)

  }


}




class LineSearchWithSyntaxState (
  var max:Double = -1.0,
  var docWeight:Double = -1.0,
  var docSyntaxWeight:Double = -1.0,
  var answerSyntaxWeight:Double = -1.0)


