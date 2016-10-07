package edu.arizona.sista.qa.scorer

import edu.arizona.sista.qa.retrieval
import retrieval.AnswerCandidate
import collection.mutable.ArrayBuffer
import org.slf4j.LoggerFactory
import java.io._
import Scorer.logger

/**
 * User: peter
 * Date: 3/20/13
 */
class Scorer () {

  // Constructor


  // Methods
  def matchDocID (candAnswer:AnswerCandidate, goldAnswer:GoldAnswer) : Boolean = {
    if (candAnswer.doc.docid == goldAnswer.docid) {
      return true
    }
    false
  }

  def sentOverlapF1 (candAnswer:AnswerCandidate, goldAnswer:GoldAnswer) : Double = {
    val total = goldAnswer.sentenceOffsets.size
    var predicted : Double = 0
    var correct : Double = 0

    if (total == 0) {             // check for empty list of sentence offsets in gold list(this should never happen)
      return 0
    }

    if (!matchDocID(candAnswer, goldAnswer)) {          // Check that documentIDs match to make sure the sentenceOffsets are from the same document
      return 0
    }

    for (i <- 0 until candAnswer.sentences.size) {   // For each sentence offset in the list of answerCandidates
    var j = 0
      var found = false
      while(j < total && ! found) {                     // For each sentence offset in the list of goldAnswers
        if (candAnswer.sentences(i) == goldAnswer.sentenceOffsets(j)) {
          found = true
        }
        j += 1
      }
      if (found) correct += 1
      predicted += 1
    }

    if ((predicted == 0) || (correct == 0)) {         // no overlap between sentence offsets in list. F1 = 0
      return 0
    }

    val precision : Double = correct/predicted
    val recall : Double = correct/total.toDouble
    val f1 = (2 * precision * recall) / (precision + recall)

    f1
  }


  def paraOverlapF1 (candAnswer:AnswerCandidate, goldAnswer:GoldAnswer) : Double = {
    var predicted : Double = 0
    var correct : Double = 0

    // If the answerCandidate and goldAnswer don't come from the same document, then there will trivially be zero overlap (and an F1 of 0)
    if (!matchDocID(candAnswer, goldAnswer)) {
      return 0
    }

    val onedoc = candAnswer.doc
    // Create list of paragraph IDs used in AnswerCandidate
    val paraCandAnswer = new collection.mutable.HashSet[Int]
    for (i <- 0 until candAnswer.sentences.size) {
      if (candAnswer.sentences(i) >= 0) {         // Bug in indexer -- occasionally sentences have negative indicies.  I think it is caused by quotes. Temproary fix.
        paraCandAnswer += onedoc.paragraphs(candAnswer.sentences(i)) // Append the paragraph ID of a given sentence in the candidate answer
      }
    }
    var offset = 0
    for (p <- paraCandAnswer) {
      offset += 1
    }

    // Create list of paragraph IDs used in goldAnswer
    val paraGoldAnswer = new collection.mutable.HashSet[Int]()
    for (i <- 0 until goldAnswer.sentenceOffsets.size) {
      if ((goldAnswer.sentenceOffsets(i) >= 0) && (goldAnswer.sentenceOffsets(i) < onedoc.paragraphs.size)) {         // Bug in indexer for Y!A -- occasionally sentences have negative indicies.  I think it is caused by quotes. Temproary fix.
        paraGoldAnswer += onedoc.paragraphs(goldAnswer.sentenceOffsets(i))      // Append the paragraph ID of a given sentence in the candidate answer
      }
    }
    offset = 0
    for (p <- paraGoldAnswer) {
      offset += 1
    }

    // check for empty list of sentence offsets in gold list(this should never happen)
    val total = paraGoldAnswer.size
    if (total == 0) {
      return 0
    }

    // Compare two paragraph ID lists to calculate precision, recall, and F1
    for (cp <- paraCandAnswer) {   // For each sentence offset in the list of answerCandidates
    var found = false
      for (gp <- paraGoldAnswer) {                     // For each sentence offset in the list of goldAnswers
        if (cp == gp) {
          found = true
        }
      }
      if (found) correct += 1
      predicted += 1
    }

    if ((predicted == 0) || (correct == 0)) {         // no overlap between sentence offsets in list. F1 = 0
      return 0
    }

    val precision : Double = correct/predicted
    val recall : Double = correct/total.toDouble

    val f1 : Double = (2 * precision * recall) / (precision + recall)

    f1
  }


  def rerankAnswersOracle(question:Question, rankedAnswers:List[AnswerCandidate], topN:Int):List[AnswerCandidate] = {
    // Step 1: Initial checks
    // Check that candidate answers list contains elements
    if (rankedAnswers.size <= 1) return rankedAnswers

    // Ensure that N is not larger than the number of candidate answers
    var N = topN
    if (topN > rankedAnswers.size) N = rankedAnswers.size

    var topF1:Double = 0
    var topAnswer = rankedAnswers(0)

    // Step 2: Find the best answer in the first N
    for (a <- 0 until question.goldAnswers.size) {
      for (i <- 0 until N) {
        val curF1 = sentOverlapF1(rankedAnswers(i), question.goldAnswers(a))
        if (curF1 > topF1) {
          topAnswer = rankedAnswers(i)
          topF1 = curF1
        }
      }
    }

    // Step 3: Append best answer to the top of the list
    val oracleRanked = List(topAnswer) ++ rankedAnswers
    return oracleRanked
  }


  def computeScoresOracle(question:Question, candAnswers:List[AnswerCandidate], topN:Int) : Scores = {
    logger.debug(" * Scorer:computeScoresOracle...")
    val oracleRanked = rerankAnswersOracle(question, candAnswers, topN)
    computeScores(question, oracleRanked)
  }


  def computeF1Subset (question:Question, candAnswersList:List[AnswerCandidate], topN:Int = -1) : (Double, Double) = {
    var sentF1Sum:Double = 0
    var paraF1Sum:Double = 0

    var N : Int = topN
    if (N == -1) N = candAnswersList.size
    if (N > candAnswersList.size) N = candAnswersList.size

    for (i <- 0 until N) {
      var topSentF1:Double = 0
      var topParaF1:Double = 0
      for (a <- 0 until question.goldAnswers.size) {
        val sentF1 = sentOverlapF1(candAnswersList(i), question.goldAnswers(a))
        val paraF1 = paraOverlapF1(candAnswersList(i), question.goldAnswers(a))
        if (sentF1 > topSentF1) topSentF1 = sentF1
        if (paraF1 > topParaF1) topParaF1 = paraF1
      }
      sentF1Sum += topSentF1
      paraF1Sum += topParaF1
    }

    (sentF1Sum/N, paraF1Sum/N)
  }

  /**
   * Computes the highest sentence F1 score for ONE candidate answer
   * This is what we use to train the re-ranker
   * @param cand The candidate
   * @param goldAnswers Gold answers for this question
   * @return The best sentence F1 score for this candidate
   */
  def maxSentOverlapF1ForCandidate(cand:AnswerCandidate, goldAnswers:Array[GoldAnswer]):Double = {
    var max:Double = 0
    for(a <- 0 until goldAnswers.size) {
      val crt = sentOverlapF1(cand, goldAnswers(a))
      if(crt > max) max = crt
    }
    max
  }


  def computeScores (question:Question, candAnswersList:List[AnswerCandidate]) : Scores = {
    // This function populates an entire Scores() data structure with sentence and paragraph precision and recall measures
    var sentOverlapAt1 : Double = 0
    var sentOverlapAt5 : Double = 0
    var sentOverlapAt10 : Double = 0
    var sentOverlapAt20 : Double = 0
    var sentOverlapMRR : Double = 0
    var sentRecallAtN : Double = 0
    var sentRRPAt1 : Double = 0

    var paraOverlapAt1 : Double = 0
    var paraOverlapAt5 : Double = 0
    var paraOverlapAt10 : Double = 0
    var paraOverlapAt20 : Double = 0
    var paraOverlapMRR : Double = 0
    var paraRecallAtN : Double = 0
    var paraRRPAt1 : Double = 0

    //    logger.debug(" * Scorer:computeScores: started...")

    for (a <- 0 until question.goldAnswers.size) {

      val sentOverlapF1s = new ArrayBuffer[Double]()
      val paraOverlapF1s = new ArrayBuffer[Double]()

      // TODO: Temporary fix
      // This should almost never happen, but may if the question contains only stop terms (e.g. "Why or why not?")
      if (candAnswersList.size == 0) {
        sentOverlapF1s.append(0)
        paraOverlapF1s.append(0)
      }

      // First compute F1 scores for each answerCandidate
      for (i <- 0 until candAnswersList.size) {
        sentOverlapF1s.append(sentOverlapF1(candAnswersList(i), question.goldAnswers(a)))
        paraOverlapF1s.append(paraOverlapF1(candAnswersList(i), question.goldAnswers(a)))
      }

      // Compute P@1 (sent)
      if (sentOverlapF1s(0) > sentOverlapAt1) {
        sentOverlapAt1 = sentOverlapF1s(0) // P@1 F1
      }
      // Compute P@1 (para)
      if (paraOverlapF1s(0) > paraOverlapAt1) {
        paraOverlapAt1 = paraOverlapF1s(0)
      }

      // Compute P@5 (sent)
      if (sentOverlapF1s.size >= 5) {
        for (i <- 0 until 5) {
          sentOverlapAt5 += sentOverlapF1s(i)
          paraOverlapAt5 += paraOverlapF1s(i)
        }
        sentOverlapAt5 = sentOverlapAt5 / 5
        paraOverlapAt5 = paraOverlapAt5 / 5
      }
      // Compute P@10 (sent)
      if (sentOverlapF1s.size >= 10) {
        for (i <- 0 until 10) {
          sentOverlapAt10 += sentOverlapF1s(i)
          paraOverlapAt10 += paraOverlapF1s(i)
        }
        sentOverlapAt10 = sentOverlapAt10 / 10
        paraOverlapAt10 = paraOverlapAt10 / 10
      }
      // Compute P@20 (sent)
      if (sentOverlapF1s.size >= 20) {
        for (i <- 0 until 20) {
          sentOverlapAt20 += sentOverlapF1s(i)
          paraOverlapAt20 += paraOverlapF1s(i)
          // Compute Oracle@20 while we're at it
          //if (sentOverlapF1s(i) > sentRecallAtN) sentRecallAtN = sentOverlapF1s(i)
          //if (paraOverlapF1s(i) > paraRecallAtN) paraRecallAtN = paraOverlapF1s(i)
        }
        sentOverlapAt20 = sentOverlapAt20 / 20
        paraOverlapAt20 = paraOverlapAt20 / 20
      }

      // Oracle performance at N (candidate retrival size)
      for (i <- 0 until candAnswersList.size) {
        if (sentOverlapF1s(i) > sentRecallAtN) sentRecallAtN = sentOverlapF1s(i)
        if (paraOverlapF1s(i) > paraRecallAtN) paraRecallAtN = paraOverlapF1s(i)
      }


      // Re-Ranking Precision @ 1
      if (sentRecallAtN > 0) {
        sentRRPAt1 = sentOverlapAt1
      } else {
        sentRRPAt1 = -1
      }

      if (paraRecallAtN > 0) {
        paraRRPAt1 = paraOverlapAt1
      } else {
        paraRRPAt1 = -1
      }


      // Compute MRR (sent)
      var maxMRR : Double = 0
      var maxRank = 0
      for (i <- 0 until candAnswersList.size) {
        val oneMRR = sentOverlapF1s(i) / (i+1)
        if (oneMRR > maxMRR) {
          maxMRR = oneMRR
          maxRank = i
        }
      }
      if (maxMRR > sentOverlapMRR) {
        sentOverlapMRR = maxMRR
      }

      // Compute MRR (para)
      maxMRR = 0
      for (i <- 0 until candAnswersList.size) {
        val oneMRR = paraOverlapF1s(i) / (i+1)
        if (oneMRR > maxMRR) {
          maxMRR = oneMRR
          maxRank = i
        }
      }
      if (maxMRR > paraOverlapMRR) {
        paraOverlapMRR = maxMRR
      }

    }

    // Package scores in storage class
    val sentScores = new MetricSet(sentOverlapAt1, sentOverlapAt5, sentOverlapAt10, sentOverlapAt20, sentOverlapMRR, sentRecallAtN, sentRRPAt1)
    val paraScores = new MetricSet(paraOverlapAt1, paraOverlapAt5, paraOverlapAt10, paraOverlapAt20, paraOverlapMRR, paraRecallAtN, paraRRPAt1)
    val oneScoreSet = new Scores(sentScores, paraScores)

    logger.debug(" * Scorer:computeScores: Summary of scores: {}", oneScoreSet.toString)

    oneScoreSet
  }


  def computeBootstrapResampling(scoresBefore:Array[Scores], scoresAfter:Array[Scores], numSamples:Int):Scores = {
    val rand = new java.util.Random()

    // Step 1: Check sizes are the same
    if (scoresBefore.size != scoresAfter.size) throw new RuntimeException ("Scorer.computeBootstrapResampling(): ERROR: scoresBefore and scoresAfter have different lengths")
    val numDataPoints = scoresBefore.size

    // Step 2: compute deltas
    val deltas = new ArrayBuffer[Scores]
    for (i <- 0 until scoresBefore.size) {
      val delta = scoresAfter(i) - scoresBefore(i)
      deltas.append(delta)
    }

    // Step 3: Resample 'numSample' times, computing the mean each time.  Store the results.
    val pw = new PrintWriter("bootstrap.txt")
    val means = new ArrayBuffer[Scores]
    for (i <- 0 until numSamples) {
      var mean = new Scores()
      for (j <- 0 until numDataPoints) {
        val randIdx = rand.nextInt(numDataPoints)
        mean = mean + deltas(randIdx)
        pw.print (randIdx + " ")
      }
      pw.println ("")
      mean = mean / numDataPoints
      means.append(mean)
    }
    pw.close()

    // Step 4: Compute proportion of means at or below 0 (the null hypothesis)
    var proportionBelowZero = new Scores()
    for (i <- 0 until numSamples) {
      val oneMean = means(i)
      //println ("bootstrap: mean: " + oneMean)
      val tempCounter = new Scores()

      // Check individual scores
      if (oneMean.sent.overlapAt1 <= 0) tempCounter.sent.overlapAt1 += 1
      if (oneMean.sent.overlapAt5 <= 0) tempCounter.sent.overlapAt5 += 1
      if (oneMean.sent.overlapAt10 <= 0) tempCounter.sent.overlapAt10 += 1
      if (oneMean.sent.overlapAt20 <= 0) tempCounter.sent.overlapAt20 += 1
      if (oneMean.sent.overlapMRR <= 0) tempCounter.sent.overlapMRR += 1
      if (oneMean.para.overlapAt1 <= 0) tempCounter.para.overlapAt1 += 1
      if (oneMean.para.overlapAt5 <= 0) tempCounter.para.overlapAt5 += 1
      if (oneMean.para.overlapAt10 <= 0) tempCounter.para.overlapAt10 += 1
      if (oneMean.para.overlapAt20 <= 0) tempCounter.para.overlapAt20 += 1
      if (oneMean.para.overlapMRR <= 0) tempCounter.para.overlapMRR += 1

      proportionBelowZero = proportionBelowZero + tempCounter
    }
    proportionBelowZero = proportionBelowZero / numSamples

    // debug
    logger.debug (proportionBelowZero.toString)
    logger.debug (" **## Sent P@1 p-value: " + proportionBelowZero.sent.overlapAt1)

    // Return a Scores storage class containing the p value for each field
    proportionBelowZero
  }


  def computeSummaryStatistics(pw:PrintWriter, scoresIn:Array[Scores]):(Scores, Scores, Scores) = {
    // Computes mean, variance, and standard deviation of a set of scores in 'scoresDelta'
    var scoresMean = new Scores()                                                              // Mean
    for (j <- 0 until scoresIn.size) {
      scoresMean = scoresMean + scoresIn(j)
      pw.println ("   j:" + j + "  " + scoresIn(j))
    }
    scoresMean =  scoresMean / scoresIn.size.toDouble
    pw.println ("  mean: " + scoresMean)

    var scoresVariance = new Scores()                                                               // Variance
    for (j <- 0 until scoresIn.size) {
      val scoresVar = (scoresIn(j) - scoresMean).square()
      scoresVariance = scoresVariance + (scoresVar / scoresIn.size.toDouble)
    }
    var scoresSD = scoresVariance.sqrt()                                                            // Standard Deviation
    pw.println ("  variance: " + scoresVariance)
    pw.println ("  sd: " + scoresSD)
    pw.println ( "" )
    pw.println ( "" )

    (scoresMean, scoresVariance, scoresSD)

  }

  def saveScoresArray(scores:Array[Scores], filename:String) {
    val pw = new PrintWriter(filename)
    // Iteratively save each set of scores
    for (i <- 0 until scores.size) {
      val sent:String = scores(i).sent.saveToString
      val para:String = scores(i).para.saveToString
      pw.println (sent)
      pw.println (para)
    }

    pw.close()
  }

  def loadScoresArray(filename:String): Array[Scores] = {
    val scores = new ArrayBuffer[Scores]()
    val source = scala.io.Source.fromFile(filename)
    val lines = source.getLines()

    while (lines.hasNext) {
      val sentMetric = new MetricSet()
      sentMetric.parseFromString(lines.next())
      val paraMetric = new MetricSet()
      paraMetric.parseFromString(lines.next())
      val score = new Scores(sentMetric, paraMetric)
      scores.append(score)
    }

    scores.toArray
  }

  def loadScoresArrayFilteredByGrade(filename:String, gradeLevels:Array[Int], includedGradeLevels:Array[Int]): Array[Scores] = {
    val scores = new ArrayBuffer[Scores]()
    val source = scala.io.Source.fromFile(filename)
    val lines = source.getLines()

    var qIndex:Int = 0
    while (lines.hasNext) {
      val sentMetric = new MetricSet()
      sentMetric.parseFromString(lines.next())
      val paraMetric = new MetricSet()
      paraMetric.parseFromString(lines.next())
      val score = new Scores(sentMetric, paraMetric)
      if (includedGradeLevels.contains(gradeLevels(qIndex))) {
        println ("Include question " + qIndex + "-- grade level " + gradeLevels(qIndex))
        scores.append(score)
      }
      qIndex += 1
    }

    scores.toArray
  }


  def analysis(question:Question, candAnswers:List[AnswerCandidate], scoresBaseline:Scores, pw:PrintWriter, questionNum:Int) {

    pw.print("======================================================================= \r\n \r\n")
    pw.print(" Q[" + questionNum + "]: " + question.text + "\r\n \r\n")

    for (i <- 0 until question.goldAnswers.size) {
      pw.print (" GA[" + i + "]    (doc_id:" + question.goldAnswers(i).docid +
        "   sentenceOffsets:" + question.goldAnswers(i).sentenceOffsets.toList + " ) \r\n")
      pw.print (question.goldAnswers(i).text + "\r\n \r\n")
    }

    pw.print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - \r\n")
    pw.print(" ANSWERS \r\n")

    // Method 1
    analysisTextMethod(candAnswers, pw, "Baseline", scoresBaseline)

    pw.print ("\r\n")
  }

  def analysisTextMethod(candAnswers:List[AnswerCandidate], pw:PrintWriter, methodText:String, scoreSet:Scores) {
    // Writes an analysis summary for one method (baseline, X BECAUSE Y, etc...) whose sorted answer set
    // is currently in candAnswers, such that the top answer is at offset 0, and the scores (P@1, MRR)
    // are reported in scoreSet
    pw.print(" METHOD: " + methodText + "\r\n")

    pw.print(" PERFORMANCE:  Sentence P@1:" + (100*scoreSet.sent.overlapAt1).formatted("%2.4f") +
      "%\t   P@5:" + (100*scoreSet.sent.overlapAt5).formatted("%2.4f") +
      "%\t   P@10:" + (100*scoreSet.sent.overlapAt10).formatted("%2.4f") +
      "%\t   P@20:" + (100*scoreSet.sent.overlapAt20).formatted("%2.4f") +
      "%\t   Recall@N:" + (100*scoreSet.sent.recallAtN).formatted("%2.4f") +
      "%\t   MRR:" + (100*scoreSet.sent.overlapMRR).formatted("%2.4f") + "%\r\n" )
    pw.print("              Paragraph P@1:" + (100*scoreSet.para.overlapAt1).formatted("%2.4f") +
      "%\t   P@5:" + (100*scoreSet.para.overlapAt5).formatted("%2.4f") +
      "%\t   P@10:" + (100*scoreSet.para.overlapAt10).formatted("%2.4f") +
      "%\t   P@20:" + (100*scoreSet.para.overlapAt20).formatted("%2.4f") +
      "%\t   Recall@N:" + (100*scoreSet.para.recallAtN).formatted("%2.4f") +
      "%\t   MRR:" + (100*scoreSet.para.overlapMRR).formatted("%2.4f") + "%\r\n" )

    // TODO: Temporary fix for searches that return zero candidate answers back.  This should almost never happen, but may for queries with only stop words (e.g. "Why or why not?")
    if (candAnswers.size > 0) {
      //pw.print(" SCORE: " + candAnswers(0).score + " \r\n")
      pw.println(" SCORE: " + candAnswers(0).score + " \t(DOC_SCORE: " + candAnswers(0).doc.docScore + "  ANS_SCORE:" + candAnswers(0).answerScore + ")")
      pw.print(" (doc_id:" + candAnswers(0).doc.docid + "   sentenceOffsets:" + candAnswers(0).sentenceStart + " to " + candAnswers(0).sentenceEnd + ") \r\n")
      pw.print(" TEXT: ")

      for (i <- candAnswers(0).sentenceStart until candAnswers(0).sentenceEnd) {
        val sent = candAnswers(0).doc.annotation.sentences(i)
        for (j <- 0 until sent.words.size) {
          pw.print( sent.words(j) + " ")
        }
      }
    }
    pw.print ("\r\n")
    pw.print ("\r\n")
  }

  def analysisTextMethodCompare(candAnswers:List[AnswerCandidate], pw:PrintWriter, methodText:String, scoreSet:Scores, scoreSetBaseline:Scores) {
    // Writes an analysis summary for one method (baseline, X BECAUSE Y, etc...) whose sorted answer set
    // is currently in candAnswers, such that the top answer is at offset 0, and the scores (P@1, MRR)
    // are reported in scoreSet
    pw.print(" METHOD: " + methodText + "\r\n")
    pw.print(" PERFORMANCE:  Sentence P@1:" + (100*scoreSet.sent.overlapAt1).formatted("%2.4f") +
      "%\t   P@5:" + (100*scoreSet.sent.overlapAt5).formatted("%2.4f") +
      "%\t   P@10:" + (100*scoreSet.sent.overlapAt10).formatted("%2.4f") +
      "%\t   P@20:" + (100*scoreSet.sent.overlapAt20).formatted("%2.4f") +
      "%\t   Recall@N:" + (100*scoreSet.sent.recallAtN).formatted("%2.4f") +
      "%\t   RRP@1:" + (100*scoreSet.sent.RRPAt1).formatted("%2.4f") +
      "%\t   MRR:" + (100*scoreSet.sent.overlapMRR).formatted("%2.4f") + "%\r\n" )
    pw.print("                 delta: P@1:" + (100*(scoreSet.sent.overlapAt1-scoreSetBaseline.sent.overlapAt1)).formatted("%2.4f") +
      "%\t   P@5:" + (100*(scoreSet.sent.overlapAt5-scoreSetBaseline.sent.overlapAt5)).formatted("%2.4f") +
      "%\t   P@10:" + (100*(scoreSet.sent.overlapAt10-scoreSetBaseline.sent.overlapAt10)).formatted("%2.4f") +
      "%\t   P@20:" + (100*(scoreSet.sent.overlapAt20-scoreSetBaseline.sent.overlapAt20)).formatted("%2.4f") +
      "%\t   Recall@N:" + (100*(scoreSet.sent.recallAtN-scoreSetBaseline.sent.recallAtN)).formatted("%2.4f") +
      "%\t   RRP@1:" + (100*(scoreSet.sent.RRPAt1-scoreSetBaseline.sent.RRPAt1)).formatted("%2.4f") +
      "%\t   MRR:" + (100*(scoreSet.sent.overlapMRR-scoreSetBaseline.sent.overlapMRR)).formatted("%2.4f") + "%\r\n")
    pw.print("              Paragraph P@1:" + (100*scoreSet.para.overlapAt1).formatted("%2.4f") +
      "%\t   P@5:" + (100*scoreSet.para.overlapAt5).formatted("%2.4f") +
      "%\t   P@10:" + (100*scoreSet.para.overlapAt10).formatted("%2.4f") +
      "%\t   P@20:" + (100*scoreSet.para.overlapAt20).formatted("%2.4f") +
      "%\t   Recall@N:" + (100*scoreSet.para.recallAtN).formatted("%2.4f") +
      "%\t   RRP@1:" + (100*scoreSet.para.RRPAt1).formatted("%2.4f") +
      "%\t   MRR:" + (100*scoreSet.para.overlapMRR).formatted("%2.4f") + "%\r\n" )
    pw.print("                 delta: P@1:" + (100*(scoreSet.para.overlapAt1-scoreSetBaseline.para.overlapAt1)).formatted("%2.4f") +
      "%\t   P@5:" + (100*(scoreSet.para.overlapAt5-scoreSetBaseline.para.overlapAt5)).formatted("%2.4f") +
      "%\t   P@10:" + (100*(scoreSet.para.overlapAt10-scoreSetBaseline.para.overlapAt10)).formatted("%2.4f") +
      "%\t   P@20:" + (100*(scoreSet.para.overlapAt20-scoreSetBaseline.para.overlapAt20)).formatted("%2.4f") +
      "%\t   Recall@N:" + (100*(scoreSet.para.recallAtN-scoreSetBaseline.para.recallAtN)).formatted("%2.4f") +
      "%\t   RRP@1:" + (100*(scoreSet.para.RRPAt1-scoreSetBaseline.para.RRPAt1)).formatted("%2.4f") +
      "%\t   MRR:" + (100*(scoreSet.para.overlapMRR-scoreSetBaseline.para.overlapMRR)).formatted("%2.4f") + "%\r\n")

    if (candAnswers.size > 0) {
      pw.println(" SCORE: " + candAnswers(0).score + " \t(DOC_SCORE: " + candAnswers(0).doc.docScore + "  ANS_SCORE:" + candAnswers(0).answerScore + ")")
      pw.print(" (doc_id:" + candAnswers(0).doc.docid + "   sentenceOffsets:" + candAnswers(0).sentenceStart + " to " + candAnswers(0).sentenceEnd + ") \r\n")
      pw.print(" TEXT: ")
      for (i <- candAnswers(0).sentenceStart until candAnswers(0).sentenceEnd) {
        val sent = candAnswers(0).doc.annotation.sentences(i);
        for (j <- 0 until sent.words.size) {
          pw.print( sent.words(j) + " ")
        }
      }
    } else {
      pw.print(" ERROR: No candidate answers found! \r\n")
    }
    pw.print ("\r\n")
    pw.print ("\r\n")
  }



  def analysisTextSummary(pw:PrintWriter, methodText:String, scoreSet:Scores) {
    // Writes an analysis summary for one method (baseline, X BECAUSE Y, etc...) whose sorted answer set
    // is currently in candAnswers, such that the top answer is at offset 0, and the scores (P@1, MRR)
    // are reported in scoreSet
    pw.println("======================================================================================")
    pw.println(" SUMMARY")
    pw.println("======================================================================================")
    pw.print(" METHOD: " + methodText + "\r\n")
    pw.print(" PERFORMANCE:  Sentence P@1:" + (100*scoreSet.sent.overlapAt1).formatted("%2.4f") +
      "%\t   P@5:" + (100*scoreSet.sent.overlapAt5).formatted("%2.4f") +
      "%\t   P@10:" + (100*scoreSet.sent.overlapAt10).formatted("%2.4f") +
      "%\t   P@20:" + (100*scoreSet.sent.overlapAt20).formatted("%2.4f") +
      "%\t   Recall@N:" + (100*scoreSet.sent.recallAtN).formatted("%2.4f") +
      "%\t   RRP@1:" + (100*scoreSet.sent.RRPAt1).formatted("%2.4f") +
      "%\t   MRR:" + (100*scoreSet.sent.overlapMRR).formatted("%2.4f") + "%\r\n" )
    pw.print("              Paragraph P@1:" + (100*scoreSet.para.overlapAt1).formatted("%2.4f") +
      "%\t   P@5:" + (100*scoreSet.para.overlapAt5).formatted("%2.4f") +
      "%\t   P@10:" + (100*scoreSet.para.overlapAt10).formatted("%2.4f") +
      "%\t   P@20:" + (100*scoreSet.para.overlapAt20).formatted("%2.4f") +
      "%\t   Recall@N:" + (100*scoreSet.para.recallAtN).formatted("%2.4f") +
      "%\t   RRP@1:" + (100*scoreSet.para.RRPAt1).formatted("%2.4f") +
      "%\t  MRR:" + (100*scoreSet.para.overlapMRR).formatted("%2.4f") + "%\r\n" )

    pw.print ("\r\n")
    pw.print ("\r\n")
  }

  def analysisTextSummaryCompare(pw:PrintWriter, methodText:String, scoreSet:Scores, scoreSetBaseline:Scores) {
    // Writes an analysis summary for one method (baseline, X BECAUSE Y, etc...) whose sorted answer set
    // is currently in candAnswers, such that the top answer is at offset 0, and the scores (P@1, MRR)
    // are reported in scoreSet
    pw.println("======================================================================================")
    pw.println(" SUMMARY")
    pw.println("======================================================================================")
    pw.print(" METHOD: " + methodText + "\r\n")
    pw.print(" PERFORMANCE:  Sentence P@1:" + (100*scoreSet.sent.overlapAt1).formatted("%2.4f") +
      "%\t   P@5:" + (100*scoreSet.sent.overlapAt5).formatted("%2.4f") +
      "%\t   P@10:" + (100*scoreSet.sent.overlapAt10).formatted("%2.4f") +
      "%\t   P@20:" + (100*scoreSet.sent.overlapAt20).formatted("%2.4f") +
      "%\t   Recall@N:" + (100*scoreSet.sent.recallAtN).formatted("%2.4f") +
      "%\t   RRP@1:" + (100*scoreSet.sent.RRPAt1).formatted("%2.4f") +
      "%\t   MRR:" + (100*scoreSet.sent.overlapMRR).formatted("%2.4f") + "%\r\n" )
    pw.print("                 delta: P@1:" + (100*(scoreSet.sent.overlapAt1-scoreSetBaseline.sent.overlapAt1)).formatted("%2.4f") +
      "%\t   P@5:" + (100*(scoreSet.sent.overlapAt5-scoreSetBaseline.sent.overlapAt5)).formatted("%2.4f") +
      "%\t   P@10:" + (100*(scoreSet.sent.overlapAt10-scoreSetBaseline.sent.overlapAt10)).formatted("%2.4f") +
      "%\t   P@20:" + (100*(scoreSet.sent.overlapAt20-scoreSetBaseline.sent.overlapAt20)).formatted("%2.4f") +
      "%\t   Recall@N:" + (100*(scoreSet.sent.recallAtN-scoreSetBaseline.sent.recallAtN)).formatted("%2.4f") +
      "%\t   RRP@1:" + (100*(scoreSet.sent.RRPAt1-scoreSetBaseline.sent.RRPAt1)).formatted("%2.4f") +
      "%\t   MRR:" + (100*(scoreSet.sent.overlapMRR-scoreSetBaseline.sent.overlapMRR)).formatted("%2.4f") + "%\r\n")
    pw.print("              Paragraph P@1:" + (100*scoreSet.para.overlapAt1).formatted("%2.4f") +
      "%\t   P@5:" + (100*scoreSet.para.overlapAt5).formatted("%2.4f") +
      "%\t   P@10:" + (100*scoreSet.para.overlapAt10).formatted("%2.4f") +
      "%\t   P@20:" + (100*scoreSet.para.overlapAt20).formatted("%2.4f") +
      "%\t   Recall@N:" + (100*scoreSet.para.recallAtN).formatted("%2.4f") +
      "%\t   RRP@1:" + (100*scoreSet.para.RRPAt1).formatted("%2.4f") +
      "%\t   MRR:" + (100*scoreSet.para.overlapMRR).formatted("%2.4f") + "%\r\n" )
    pw.print("                 delta: P@1:" + (100*(scoreSet.para.overlapAt1-scoreSetBaseline.para.overlapAt1)).formatted("%2.4f") +
      "%\t   P@5:" + (100*(scoreSet.para.overlapAt5-scoreSetBaseline.para.overlapAt5)).formatted("%2.4f") +
      "%\t   P@10:" + (100*(scoreSet.para.overlapAt10-scoreSetBaseline.para.overlapAt10)).formatted("%2.4f") +
      "%\t   P@20:" + (100*(scoreSet.para.overlapAt20-scoreSetBaseline.para.overlapAt20)).formatted("%2.4f") +
      "%\t   Recall@N:" + (100*(scoreSet.para.recallAtN-scoreSetBaseline.para.recallAtN)).formatted("%2.4f") +
      "%\t   RRP@1:" + (100*(scoreSet.para.RRPAt1-scoreSetBaseline.para.RRPAt1)).formatted("%2.4f") +
      "%\t   MRR:" + (100*(scoreSet.para.overlapMRR-scoreSetBaseline.para.overlapMRR)).formatted("%2.4f") + "%\r\n")

    pw.print ("\r\n")
    pw.print ("\r\n")
  }


  def analysisTextSummaryPValues(pw:PrintWriter, methodText:String, scoreSetPValues:Scores) {
    // Writes an analysis summary for one method (baseline, X BECAUSE Y, etc...) whose sorted answer set
    // is currently in candAnswers, such that the top answer is at offset 0, and the scores (P@1, MRR)
    // are reported in scoreSet
    pw.println("======================================================================================")
    pw.println(" SUMMARY  (P-values)")
    pw.println("======================================================================================")
    pw.print(" METHOD: " + methodText + "\r\n")
    pw.print(" PERFORMANCE:  Sentence P@1:" + (100*scoreSetPValues.sent.overlapAt1).formatted("%2.4f") +
      "%\t   P@5:" + (100*scoreSetPValues.sent.overlapAt5).formatted("%2.4f") +
      "%\t   P@10:" + (100*scoreSetPValues.sent.overlapAt10).formatted("%2.4f") +
      "%\t   P@20:" + (100*scoreSetPValues.sent.overlapAt20).formatted("%2.4f") +
      "%\t   RRP@1:" + (100*scoreSetPValues.sent.RRPAt1).formatted("%2.4f") +
      "%\t   MRR:" + (100*scoreSetPValues.sent.overlapMRR).formatted("%2.4f") + "%\r\n" )

    pw.print("              Paragraph P@1:" + (100*scoreSetPValues.para.overlapAt1).formatted("%2.4f") +
      "%\t   P@5:" + (100*scoreSetPValues.para.overlapAt5).formatted("%2.4f") +
      "%\t   P@10:" + (100*scoreSetPValues.para.overlapAt10).formatted("%2.4f") +
      "%\t   P@20:" + (100*scoreSetPValues.para.overlapAt20).formatted("%2.4f") +
      "%\t   RRP@1:" + (100*scoreSetPValues.para.RRPAt1).formatted("%2.4f") +
      "%\t   MRR:" + (100*scoreSetPValues.para.overlapMRR).formatted("%2.4f") + "%\r\n" )

    pw.print ("\r\n")
    pw.print ("\r\n")
  }

}


object Scorer {
  val logger = LoggerFactory.getLogger(classOf[Scorer])
}
