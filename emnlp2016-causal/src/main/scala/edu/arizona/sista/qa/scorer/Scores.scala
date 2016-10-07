package edu.arizona.sista.qa.scorer

import java.io.FileWriter
import scala.math.sqrt

/**
 * Scores storage class
 * User: peter
 * Date: 3/20/13
 */
class Scores ( var sent:MetricSet, var para:MetricSet) {
  var numSamples: Double = 0
  var numSamplesSentRRP: Double = 0
  var numSamplesParaRRP: Double = 0

  // Constructor
  def this() = this(new MetricSet(), new MetricSet())      // Initialize to zero score, when used as an accumulator

  // Methods

  def +(in:Scores): Scores = {
    new Scores (
      this.sent + in.sent,
      this.para + in.para)
  }

  def -(in:Scores): Scores = {
    new Scores (
      this.sent - in.sent,
      this.para - in.para)
  }

  def *(toMult:Double): Scores = {
    new Scores (
      this.sent * toMult,
      this.para * toMult)
  }

  def *(in:Scores): Scores = {
    new Scores (
      this.sent * in.sent,
      this.para * in.para)
  }

  def /(toDiv:Double): Scores = {
    new Scores (
      this.sent / toDiv,
      this.para / toDiv)
  }

  def square(): Scores = {
    new Scores (
      this.sent * this.sent,
      this.para + this.para)
  }

  def sqrt(): Scores = {
    new Scores (
      this.sent.sqrt(),
      this.para.sqrt() )
  }

  def addToAverage(toAdd:Scores) {
    // RRPAt1 is only calculated for the proportion of questions that are correctly retrieved by the IR baseline, which
    // requires calculating two averages (the average for P@1, MRR, and Recall@N metrics, and the average for the RRPAt1)

    // Step 1: Store current RRPAt1 values
    val prevSentRRPAt1 = this.sent.RRPAt1
    val prevParaRRPAt1 = this.para.RRPAt1

    // Step 2: Perform normal average across all metrics
    numSamples += 1
    this.sent = (this.sent * (1-(1/numSamples))) + (toAdd.sent * (1/numSamples))
    this.para = (this.para * (1-(1/numSamples))) + (toAdd.para * (1/numSamples))

    // Step 3: Calculate RRPAt1 scores, with RRP-specific accumulators
    // Note: RRP values of -1 signify they are invalid, and should not be added to the average
    if (toAdd.sent.RRPAt1 > -1) {
      numSamplesSentRRP += 1
      this.sent.RRPAt1 = (this.sent.RRPAt1 * (1-(1/numSamplesSentRRP))) + (toAdd.sent.RRPAt1 * (1/numSamplesSentRRP))
    } else {
      this.sent.RRPAt1 = prevSentRRPAt1
    }

    if (toAdd.para.RRPAt1 > -1) {
      numSamplesParaRRP += 1
      this.para.RRPAt1 = (this.para.RRPAt1 * (1-(1/numSamplesParaRRP))) + (toAdd.para.RRPAt1 * (1/numSamplesParaRRP))
    } else {
      this.para.RRPAt1 = prevParaRRPAt1
    }

  }


  override def toString : String = {
    var outstring = new StringBuilder

    outstring.append("sentence (")
    outstring.append(sent.toString)
    outstring.append(")   paragraph (")
    outstring.append(para.toString)
    outstring.append(") ")

    outstring.toString()
  }

  def analysisReportSummary(os:FileWriter, methodText:String) {
    // Performance summary
    os.write(" METHOD: " + methodText + "\r\n")
    os.write(" PERFORMANCE:  Sentence P@1:" + (100*sent.overlapAt1).formatted("%2.2f") + "%         MRR:" + (100*sent.overlapMRR).formatted("%2.2f") + "% \r\n" )
    os.write("              Paragraph P@1:" + (100*para.overlapAt1).formatted("%2.2f") + "%         MRR:" + (100*para.overlapMRR).formatted("%2.2f") + "% \r\n" )
    os.write("               Sentence P@5:" + (100*sent.overlapAt5).formatted("%2.2f") + "%    Para P@5:" + (100*para.overlapAt5).formatted("%2.2f") + "% \r\n" )
    os.write("              Sentence P@10:" + (100*sent.overlapAt10).formatted("%2.2f") + "%  Para P@10:" + (100*para.overlapAt10).formatted("%2.2f") + "% \r\n" )
    os.write("              Sentence P@20:" + (100*sent.overlapAt20).formatted("%2.2f") + "%  Para P@20:" + (100*para.overlapAt20).formatted("%2.2f") + "% \r\n" )
    os.write ("\r\n")
  }

  def analysisReportSummaryCompare(os:FileWriter, methodText:String, baseline:Scores) {
    // Performance summary with baseline comparison
    os.write(" METHOD: " + methodText + "\r\n")
    os.write(" PERFORMANCE:  Sentence P@1:" + (100*sent.overlapAt1).formatted("%2.2f") + "%   MRR:" + (100*sent.overlapMRR).formatted("%2.2f") + "%" )
    os.write("       delta: P@1:" + (100*(sent.overlapAt1-baseline.sent.overlapAt1)).formatted("%2.2f") + "   MRR:" + (100*(sent.overlapMRR-baseline.sent.overlapMRR)).formatted("%2.2f") + "%\r\n")
    os.write("              Paragraph P@1:" + (100*para.overlapAt1).formatted("%2.2f") + "%   MRR:" + (100*para.overlapMRR).formatted("%2.2f") + "%" )
    os.write("       delta: P@1:" + (100*(para.overlapAt1-baseline.para.overlapAt1)).formatted("%2.2f") + "   MRR:" + (100*(para.overlapMRR-baseline.para.overlapMRR)).formatted("%2.2f") + "%\r\n")

    os.write("              Sentence P@5:" + (100*sent.overlapAt5).formatted("%2.2f") + "%    Para P@5:" + (100*para.overlapAt5).formatted("%2.2f") + "%" )
    os.write("       delta: Sentence P@5:" + (100*(sent.overlapAt5-baseline.sent.overlapAt5)).formatted("%2.2f") + "%    Para P@5:" + (100*(para.overlapAt5-baseline.para.overlapAt5)).formatted("%2.2f") + "%\r\n")
    os.write("             Sentence P@10:" + (100*sent.overlapAt10).formatted("%2.2f") + "%   Para P@10:" + (100*para.overlapAt10).formatted("%2.2f") + "%" )
    os.write("      delta: Sentence P@10:" + (100*(sent.overlapAt10-baseline.sent.overlapAt10)).formatted("%2.2f") + "%   Para P@10:" + (100*(para.overlapAt10-baseline.para.overlapAt10)).formatted("%2.2f") + "%\r\n")
    os.write("             Sentence P@20:" + (100*sent.overlapAt20).formatted("%2.2f") + "%   Para P@20:" + (100*para.overlapAt20).formatted("%2.2f") + "%" )
    os.write("      delta: Sentence P@20:" + (100*(sent.overlapAt20-baseline.sent.overlapAt20)).formatted("%2.2f") + "%   Para P@20:" + (100*(para.overlapAt20-baseline.para.overlapAt20)).formatted("%2.2f") + "%\r\n")

    os.write ("\r\n")
  }


}



/*
 * Storage class for one set of scores (Precision @1, @5, @10, @20, Mean Reciprocol Rank (MRR), Recall@N, and Reranking Precision@1 (RRPAt1) )
 */
class MetricSet (
  var overlapAt1 : Double,
  var overlapAt5 : Double,
  var overlapAt10 : Double,
  var overlapAt20 : Double,
  var overlapMRR : Double,
  var recallAtN : Double,
  var RRPAt1 : Double) {

  // Constructor
  def this() = this(0, 0, 0, 0, 0, 0, 0)      // Initialize to zero score, when used as an accumulator

  // Methods / Operators
  def +(in:MetricSet): MetricSet = {
    new MetricSet (
      this.overlapAt1 + in.overlapAt1,
      this.overlapAt5 + in.overlapAt5,
      this.overlapAt10 + in.overlapAt10,
      this.overlapAt20 + in.overlapAt20,
      this.overlapMRR + in.overlapMRR,
      this.recallAtN + in.recallAtN,
      this.RRPAt1 + in.RRPAt1)
  }

  def -(in:MetricSet): MetricSet = {
    new MetricSet (
      this.overlapAt1 - in.overlapAt1,
      this.overlapAt5 - in.overlapAt5,
      this.overlapAt10 - in.overlapAt10,
      this.overlapAt20 - in.overlapAt20,
      this.overlapMRR - in.overlapMRR,
      this.recallAtN - in.recallAtN,
      this.RRPAt1 - in.RRPAt1)
  }

  def *(toMult:Double): MetricSet = {
    new MetricSet (
      this.overlapAt1 * toMult,
      this.overlapAt5 * toMult,
      this.overlapAt10 * toMult,
      this.overlapAt20 * toMult,
      this.overlapMRR * toMult,
      this.recallAtN * toMult,
      this.RRPAt1 * toMult)
  }

  def *(in:MetricSet): MetricSet = {
    new MetricSet (
      this.overlapAt1 * in.overlapAt1,
      this.overlapAt5 * in.overlapAt5,
      this.overlapAt10 * in.overlapAt10,
      this.overlapAt20 * in.overlapAt20,
      this.overlapMRR * in.overlapMRR,
      this.recallAtN * in.recallAtN,
      this.RRPAt1 * in.RRPAt1)
  }

  def /(toDiv:Double): MetricSet = {
    new MetricSet (
      this.overlapAt1 / toDiv,
      this.overlapAt5 / toDiv,
      this.overlapAt10 / toDiv,
      this.overlapAt20 / toDiv,
      this.overlapMRR / toDiv,
      this.recallAtN / toDiv,
      this.RRPAt1 / toDiv)
  }

  def square(): MetricSet = {
    new MetricSet (
      this.overlapAt1 * this.overlapAt1,
      this.overlapAt5 * this.overlapAt5,
      this.overlapAt10 * this.overlapAt10,
      this.overlapAt20 * this.overlapAt20,
      this.overlapMRR * this.overlapMRR,
      this.recallAtN * this.recallAtN,
      this.RRPAt1 * this.RRPAt1)
  }

  def sqrt(): MetricSet = {
    new MetricSet (
      scala.math.sqrt(this.overlapAt1),
      scala.math.sqrt(this.overlapAt5),
      scala.math.sqrt(this.overlapAt10),
      scala.math.sqrt(this.overlapAt20),
      scala.math.sqrt(this.overlapMRR),
      scala.math.sqrt(this.recallAtN),
      scala.math.sqrt(this.RRPAt1) )
  }

  override def toString : String = {
    var outstring = new StringBuilder

    outstring.append(" P@1:" + overlapAt1.formatted("%3.5f"))
    outstring.append(" P@5:" + overlapAt5.formatted("%3.5f"))
    outstring.append(" P@10:" + overlapAt10.formatted("%3.5f"))
    outstring.append("  MRR:" + overlapMRR.formatted("%3.5f"))
    outstring.append(" Recall@N:" + recallAtN.formatted("%3.5f"))
    outstring.append(" RRP@1:" + RRPAt1.formatted("%3.5f"))

    outstring.toString()
  }

  def saveToString:String = {
    var outstring = new StringBuilder

    outstring.append (overlapAt1.toString + ",")
    outstring.append (overlapAt5.toString + ",")
    outstring.append (overlapAt10.toString + ",")
    outstring.append (overlapMRR.toString + ",")
    outstring.append (recallAtN.toString + ",")
    outstring.append (RRPAt1.toString)

    outstring.toString()
  }

  def parseFromString(in:String) = {
    var fields = in.split(",")
    if (fields.size != 6) throw new RuntimeException("ERROR: Scores.parseFromString: Input string does not have exactly 6 fields. ")
    overlapAt1 = fields(0).toDouble     // P@1
    overlapAt5 = fields(1).toDouble     // P@5
    overlapAt10 = fields(2).toDouble    // P@10
    overlapMRR = fields(3).toDouble     // MRR
    recallAtN = fields(4).toDouble      // Recall@N
    RRPAt1 = fields(5).toDouble         // RRP@1
  }

}





/*


class AvgScores () {
  // NOTE: This could use some cleanup

  // Constructor
  var sentOverlapAt1: Double = 0
  var paraOverlapAt1: Double = 0
  var sentOverlapMRR: Double = 0
  var paraOverlapMRR: Double = 0

  var sentOverlapAt5 : Double = 0
  var sentOverlapAt10 : Double = 0
  var sentOverlapAt20 : Double = 0
  var paraOverlapAt5 : Double = 0
  var paraOverlapAt10 : Double = 0
  var paraOverlapAt20 : Double = 0

  var binaryOverlapAt1: Double = 0

  var sentRecallAtN : Double = 0
  var paraRecallAtN : Double = 0

  var sentRRPAt1 : Double = 0
  var paraRRPAt1 : Double = 0


  var numSamples: Double = 0
  var numSentRRPSamples : Double = 0
  var numParaRRPSamples : Double = 0

  // Methods
  def add (toadd:Scores) {
    numSamples += 1
    sentOverlapAt1 = (sentOverlapAt1 * (1-(1/numSamples)) + (toadd.sentOverlapAt1 * (1/numSamples)))
    paraOverlapAt1 = (paraOverlapAt1 * (1-(1/numSamples)) + (toadd.paraOverlapAt1 * (1/numSamples)))
    sentOverlapMRR = (sentOverlapMRR * (1-(1/numSamples)) + (toadd.sentOverlapMRR * (1/numSamples)))
    paraOverlapMRR = (paraOverlapMRR * (1-(1/numSamples)) + (toadd.paraOverlapMRR * (1/numSamples)))

    sentOverlapAt5 = (sentOverlapAt5 * (1-(1/numSamples)) + (toadd.sentOverlapAt5 * (1/numSamples)))
    paraOverlapAt5 = (paraOverlapAt5 * (1-(1/numSamples)) + (toadd.paraOverlapAt5 * (1/numSamples)))
    sentOverlapAt10 = (sentOverlapAt10 * (1-(1/numSamples)) + (toadd.sentOverlapAt10 * (1/numSamples)))
    paraOverlapAt10 = (paraOverlapAt10 * (1-(1/numSamples)) + (toadd.paraOverlapAt10 * (1/numSamples)))
    sentOverlapAt20 = (sentOverlapAt20 * (1-(1/numSamples)) + (toadd.sentOverlapAt20 * (1/numSamples)))
    paraOverlapAt20 = (paraOverlapAt20 * (1-(1/numSamples)) + (toadd.paraOverlapAt20 * (1/numSamples)))

    binaryOverlapAt1 = (binaryOverlapAt1 * (1-(1/numSamples)) + (toadd.binaryOverlapAt1 * (1/numSamples)))

    sentRecallAtN = (sentRecallAtN * (1-(1/numSamples)) + (toadd.sentRecallAtN * (1/numSamples)))
    paraRecallAtN = (paraRecallAtN * (1-(1/numSamples)) + (toadd.paraRecallAtN * (1/numSamples)))

    if (toadd.sentRRPAt1 > -1) {
      numSentRRPSamples += 1
      sentRRPAt1 = (sentRRPAt1 * (1-(1/numSentRRPSamples)) + (toadd.sentRRPAt1 * (1/numSentRRPSamples)))
    }

    if (toadd.paraRRPAt1 > -1) {
      numParaRRPSamples += 1
      paraRRPAt1 = (paraRRPAt1 * (1-(1/numParaRRPSamples)) + (toadd.paraRRPAt1 * (1/numParaRRPSamples)))
    }


  }

  def analysisReportSummary(os:FileWriter, methodText:String) {
    // Performance summary
    os.write(" METHOD: " + methodText + "\r\n")
    os.write(" PERFORMANCE:  Sentence P@1:" + (100*sentOverlapAt1).formatted("%2.2f") + "%         MRR:" + (100*sentOverlapMRR).formatted("%2.2f") + "% \r\n" )
    os.write("              Paragraph P@1:" + (100*paraOverlapAt1).formatted("%2.2f") + "%         MRR:" + (100*paraOverlapMRR).formatted("%2.2f") + "% \r\n" )
    os.write("               Sentence P@5:" + (100*sentOverlapAt5).formatted("%2.2f") + "%    Para P@5:" + (100*paraOverlapAt5).formatted("%2.2f") + "% \r\n" )
    os.write("              Sentence P@10:" + (100*sentOverlapAt10).formatted("%2.2f") + "%  Para P@10:" + (100*paraOverlapAt10).formatted("%2.2f") + "% \r\n" )
    os.write("              Sentence P@20:" + (100*sentOverlapAt20).formatted("%2.2f") + "%  Para P@20:" + (100*paraOverlapAt20).formatted("%2.2f") + "% \r\n" )
    os.write("               AtLeast1 P@1:" + (100*binaryOverlapAt1).formatted("%2.2f") + "%  \r\n" )
    os.write ("\r\n")
  }

  def analysisReportSummaryCompare(os:FileWriter, methodText:String, avgScoreSetBaseline:AvgScores) {
    // Performance summary with baseline comparison
    os.write(" METHOD: " + methodText + "\r\n")
    os.write(" PERFORMANCE:  Sentence P@1:" + (100*sentOverlapAt1).formatted("%2.2f") + "%   MRR:" + (100*sentOverlapMRR).formatted("%2.2f") + "%" )
    os.write("       delta: P@1:" + (100*(sentOverlapAt1-avgScoreSetBaseline.sentOverlapAt1)).formatted("%2.2f") + "   MRR:" + (100*(sentOverlapMRR-avgScoreSetBaseline.sentOverlapMRR)).formatted("%2.2f") + "%\r\n")
    os.write("              Paragraph P@1:" + (100*paraOverlapAt1).formatted("%2.2f") + "%   MRR:" + (100*paraOverlapMRR).formatted("%2.2f") + "%" )
    os.write("       delta: P@1:" + (100*(paraOverlapAt1-avgScoreSetBaseline.paraOverlapAt1)).formatted("%2.2f") + "   MRR:" + (100*(paraOverlapMRR-avgScoreSetBaseline.paraOverlapMRR)).formatted("%2.2f") + "%\r\n")

    os.write("              Sentence P@5:" + (100*sentOverlapAt5).formatted("%2.2f") + "%    Para P@5:" + (100*paraOverlapAt5).formatted("%2.2f") + "%" )
    os.write("       delta: Sentence P@5:" + (100*(sentOverlapAt5-avgScoreSetBaseline.sentOverlapAt5)).formatted("%2.2f") + "%    Para P@5:" + (100*(paraOverlapAt5-avgScoreSetBaseline.paraOverlapAt5)).formatted("%2.2f") + "%\r\n")
    os.write("             Sentence P@10:" + (100*sentOverlapAt10).formatted("%2.2f") + "%   Para P@10:" + (100*paraOverlapAt10).formatted("%2.2f") + "%" )
    os.write("      delta: Sentence P@10:" + (100*(sentOverlapAt10-avgScoreSetBaseline.sentOverlapAt10)).formatted("%2.2f") + "%   Para P@10:" + (100*(paraOverlapAt10-avgScoreSetBaseline.paraOverlapAt10)).formatted("%2.2f") + "%\r\n")
    os.write("             Sentence P@20:" + (100*sentOverlapAt20).formatted("%2.2f") + "%   Para P@20:" + (100*paraOverlapAt20).formatted("%2.2f") + "%" )
    os.write("      delta: Sentence P@20:" + (100*(sentOverlapAt20-avgScoreSetBaseline.sentOverlapAt20)).formatted("%2.2f") + "%   Para P@20:" + (100*(paraOverlapAt20-avgScoreSetBaseline.paraOverlapAt20)).formatted("%2.2f") + "%\r\n")

    os.write("               AtLeast1 P@1:" + (100*binaryOverlapAt1).formatted("%2.2f") + "%             " )
    os.write("       delta: P@1:" + (100*(binaryOverlapAt1-avgScoreSetBaseline.binaryOverlapAt1)).formatted("%2.2f") + "%\r\n")

    os.write ("\r\n")
  }


  override def toString : String = {
    var outstring = new StringBuilder

    outstring.append("average: sentence (")
    outstring.append(" P@1:" + sentOverlapAt1)
    outstring.append("  MRR:" + sentOverlapMRR)
    outstring.append(")   paragraph (")
    outstring.append(" P@1:" + paraOverlapAt1)
    outstring.append("  MRR:" + paraOverlapMRR)
    outstring.append(")")

    outstring.toString()
  }

}

*/