package edu.arizona.sista.qa.preprocessing.yahoo

import java.io.PrintWriter

import edu.arizona.sista.processors.Document
import edu.arizona.sista.qa.retrieval.AnswerCandidate
import edu.arizona.sista.qa.scorer.Question

/**
  * Created by bsharp on 5/9/16.
  */
object YahooKerasUtils {

  def dumpYahooDataforKeras(dir: String,
                            fold: String,
                            candidates: Array[Array[AnswerCandidate]],
                            questions: Array[Question],
                            queryAnnotations: Array[Document]): Unit = {
    // Here - dump out the input to keras!
    // four files:
    //    A:  with q+ans info (details to align and debug)
    //    B:  with questWords, tab separated, with word_POS
    //    C:  with candWords, tab separated, and with word_POS
    //    D:  with 0/1 for correctness!, comma separated

    val pwA = new PrintWriter(dir + s"A_questionInfo.$fold.tsv")
    val pwB = new PrintWriter(dir + s"B_questionTokens.$fold.tsv")
    val pwC = new PrintWriter(dir + s"C_candidateTokens.$fold.tsv")
    val pwD = new PrintWriter(dir + s"D_candidateCorrectness.$fold.tsv")

    for (qID <- candidates.indices) {
      val question = questions(qID)
      val questionDoc = queryAnnotations(qID)
      val goldDocId = question.goldAnswers.head.docid
      val questionCandidates = candidates(qID)
      for (candidateId <- questionCandidates.indices) {
        val candidate: AnswerCandidate = questionCandidates(candidateId)
        val candDocId = candidate.doc.docid
        val gold: Int = if (candDocId == goldDocId) 1 else 0
        // Print the docID of the candidate, the question, and the candidate text to file A
        pwA.println(Array(candDocId, questionDoc.sentences.map(_.getSentenceText()).mkString(","), candidate.getText.trim()).mkString("\t"))
        pwA.flush()
        // Print the Question tokens with POS, tab separated, to file B
        val qLemmas = questionDoc.sentences.flatMap(_.lemmas.get)
        val qTags = questionDoc.sentences.flatMap(_.tags.get)
        val qTokens = qLemmas.zip(qTags).map(tup => tup._1 + "_" + tup._2)
        pwB.println(qTokens.mkString("\t"))
        pwB.flush()
        // Print the Candidate tokens with POS, tab separated, to file C
        val cLemmas = candidate.doc.annotation.sentences.flatMap(_.lemmas.get)
        val cTags = candidate.doc.annotation.sentences.flatMap(_.tags.get)
        val cTokens = cLemmas.zip(cTags).map(tup => tup._1 + "_" + tup._2)
        pwC.println(cTokens.mkString("\t"))
        pwC.flush()
        // Print the Correctness status to file D
        pwD.println(gold)
        pwD.flush()
      }
    }
    pwA.close()
    pwB.close()
    pwC.close()
    pwD.close()
    println("Saved the Yahoo QA keras input files (A-D) to " + dir)

    // Exit if all data has been saved (test data is the last to go)
    if (fold == "test") {
      println("Now exiting.")
      sys.exit()
    }

  }

}
