package edu.arizona.sista.qa.word2vec

import java.io.PrintWriter
import edu.arizona.sista.learning.ScaleRange
import edu.arizona.sista.qa.ranking.{ProcessedQuestionSegments, RankerVoting}
import edu.arizona.sista.qa.retrieval.AnswerCandidate
import edu.arizona.sista.qa.scorer.QuestionParser
import edu.arizona.sista.struct.Counter
import edu.arizona.sista.utils.StringUtils

/**
  * Created by bsharp on 9/21/16.
  */
object ErrorAnalysisCausal {

  val weights = new Counter[String]
  weights.setCount("WORD2VEC_vanilla_MAXSIM", 0.02118)
  weights.setCount("WORD2VEC_vanilla_MINSIM", -1.70326)
  weights.setCount("WORD2VEC_vanilla_AVGSIM", -1.84217)
  weights.setCount("WORD2VEC_vanilla_TEXTSIM", 2.86727)
  weights.setCount("WORD2VEC_REL_BIDIR_causal_mar30_1hop_MAXSIM", 0.82839)
  weights.setCount("WORD2VEC_REL_BIDIR_causal_mar30_1hop_MINSIM", -0.44499)
  weights.setCount("WORD2VEC_REL_BIDIR_causal_mar30_1hop_AVGSIM", -2.17742)
  weights.setCount("WORD2VEC_REL_BIDIR_causal_mar30_1hop_TEXTSIM", 1.72517)
  weights.setCount("ir", -0.19607)

  def includeWeightInfo(features: Counter[String]): String = {
    val out = new StringBuilder
    val withWeights = features.toSeq.map(tup => (tup._1, tup._2, weights.getCount(tup._1), weights.getCount(tup._1) * tup._2, Math.abs(weights.getCount(tup._1) * tup._2)))
    val sorted = withWeights.sortBy(- _._5)
    var runningTotal: Double = 0.0
    for ((f, rawScore, w, finalScore, absVal) <- sorted) {
      runningTotal += finalScore
      out.append(s"  ${"%1$-50s".format(f)}\t")
      out.append(s"score: ${rawScore.formatted("%1.3f")}  *  ")
      out.append(s"weight: ${w.formatted("%1.3f")} = ${finalScore.formatted("%1.3f")}")
      out.append(s"\t [RunningTotal: $runningTotal]")
      out.append("\n")
    }

    out.toString()
  }

  def rescale(ranges:ScaleRange[String], features: Counter[String]): Counter[String] = {
    val scaled = new Counter[String]
    for (featureName <- features.keySet) {
      val featureValue = features.getCount(featureName)
      var min:Double = 0.0
      var max:Double = 0.0
      if(ranges.contains(featureName)) {
        min = ranges.min(featureName)
        max = ranges.max(featureName)
      }
      val scaledScore = scale(featureValue, min, max, 0.0, 1.0)
      scaled.setCount(featureName, scaledScore)
    }

    scaled
  }

  def scale(value:Double, min:Double, max:Double, lower:Double, upper:Double):Double = {
    if(min == max) return upper

    // the result will be a value in [lower, upper]
    lower + (upper - lower) * (value - min) / (max - min)
  }

  def main(args:Array[String]): Unit = {
    val props = StringUtils.argsToProperties(args)
    val questionParser = new QuestionParser
    val questionsFilename = props.getProperty("test")
    val questions = questionParser.parse(questionsFilename).toArray
    val rv = new RankerVoting(props)
    val model = new Word2VecRelationModel(props)
    val pw = new PrintWriter("/lhome/bsharp/causal/yahoo/EA/detailedPerQuestion1.txt")

    // Step 2: Generate model features from test questions
    val queryAnnotations = rv.mkAnnotations(questions)

    var candidates = Array.empty[Array[AnswerCandidate]]
    candidates = rv.mkCandidatesCQA(questions, queryAnnotations)

    val processedQuestions = queryAnnotations.map(model.mkProcessedQuestion)

    val questionsToAnalyze = Array(42, 51, 100, 125, 138, 194, 234, 241, 263, 268, 286, 345, 400, 402, 436, 458, 475, 484, 487, 557, 4, 106, 144, 158, 178, 228, 267, 280, 307, 308, 314, 326, 352, 408, 429, 449, 454, 473, 548, 583, 597)

    for (i <- questionsToAnalyze) {
      pw.println(s"Question [$i]: ${questions(i).toString()}")
      val question = processedQuestions(i).head
      pw.println("")
      pw.println(s"There are ${candidates(i).length} candidate answers:")
      for (j <- candidates(i).indices) {
        val cand = candidates(i)(j)
        pw.println (s"CANDIDATE $j")
        question match {
          case q:ProcessedQuestionSegments => model.generateModelErrorAnalysisForQuestion(cand, question.asInstanceOf[ProcessedQuestionSegments], pw)
          case _ => throw new RuntimeException ("Word2VecModel.mkFeatures(): question passed is not of type ProcessedQuestionSegments")
        }
        pw.println("")
      }
      pw.println("----------------------------------------------------------------")
    }

  }



}
