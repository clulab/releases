package edu.arizona.sista.qa.preprocessing.yahoo

import java.io.PrintWriter

import edu.arizona.sista.qa.preprocessing.yahoo.deep.{CQAQuestion, CQAQuestionParser}
import edu.arizona.sista.qa.translation.FreeTextAlignmentUtils

import scala.collection.mutable.ArrayBuffer

/**
  * Created by bsharp on 4/25/16.
  */
object CausalCheck {

  val CQAParser = new CQAQuestionParser

  def main(args:Array[String]): Unit = {

    // Load the Questions:
    val qFiles = FreeTextAlignmentUtils.findFiles("/data/nlp/corpora/yahoo", "xml")
    val questions = new ArrayBuffer[CQAQuestion]
    for {
      file <- qFiles
      filePath = file.getAbsolutePath
    } questions.appendAll(CQAParser.load(filePath, annotate = true))

    // Display:
    val out = new PrintWriter("/lhome/bsharp/causal/yahoo/extractedYahooCausalQuestions.txt")
    for (qId <- questions.indices) {
      val q = questions(qId)
      out.println ("\n========================================================================================")
      out.println (s"\nQuestion $qId: ${q.questionText}")
      val golds = q.goldAnswers
      for (ga <- golds) {
        out.println ("(*) Gold: " + ga.text)
      }
      val otherAnswers = q.answers.filter(_.gold == false)
      for (aId <- otherAnswers.indices) {
        val a = otherAnswers(aId)
        out.println ("(x) Incorr Answer: " + a.text)
      }
      out.flush()
    }
    out.close()

  }

}
