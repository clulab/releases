package preprocessing

import java.io.PrintWriter

import edu.arizona.sista.struct.{Lexicon, Counter}

import scala.collection.mutable.ArrayBuffer

/**
  * Created by bsharp on 5/5/16.
  */
object CreateQAInputForKeras extends App {

  val stopWords = CreateGoldbergInput.stopWords

  def getQuestionTextsFromFile(fn:String):Seq[String] = {
    val out = new ArrayBuffer[String]
    val source = scala.io.Source.fromFile(fn)
    val lines = source.getLines()
    for (line <- lines){
      val fields = line.split("\t")
      out.append(fields(1).toLowerCase)
    }

    source.close()
    out
  }

  def getTokensFromFile(fn:String):Seq[Seq[String]] = {
    val out = new ArrayBuffer[Seq[String]]
    val source = scala.io.Source.fromFile(fn)
    val lines = source.getLines()
    for (line <- lines){
      val tokens = line.split("\t")
      out.append(tokens)
    }

    source.close()
    out
  }

  def filterTokens(in:Seq[String], keepTags:Set[String]):Seq[String] = {
    val out = for {
      token <- in
      tag = getTag(token)
      if startsWithOneOf(tag, keepTags)
      lemma = getLemma(token)
      if !stopWords.contains(lemma)
    } yield token
    if (out.length == 0) {
      println ("Nothing left after filtering: " + in.mkString(" "))
      return Seq("<<PADDING>>_PAD")
    }
    out
  }

  def getTag(s:String):String = s.split("_")(1)

  def getLemma(s:String):String = s.split("_")(0)

  def startsWithOneOf(s:String, set:Set[String]): Boolean = {
    for (prefix <- set) {
      if (s.startsWith(prefix)) return true
    }
    // else
    false
  }

  // Takes the question texts, and question and candidate answer tokens (one cand per "line" -- i.e. outer array), and for each decides
  // which is the cause/effect.  Returns two parallel arrays -- causeTokens and effectTokens
  def selectDirections(questionTexts:Seq[String], qTokens:Seq[Seq[String]], cTokens:Seq[Seq[String]]): (Seq[Seq[String]], Seq[Seq[String]]) = {
    val causeTokens = new ArrayBuffer[Seq[String]]
    val effectTokens = new ArrayBuffer[Seq[String]]

    for (i <- qTokens.indices) {
      val questionText = questionTexts(i)
      val q = qTokens(i)
      val c = cTokens(i)
      // If the question text is the effect text
      if (questionIsEffect(questionText)) {
        causeTokens.append(c)
        effectTokens.append(q)
      }
      else {
        // If the question text is the cause text
        causeTokens.append(q)
        effectTokens.append(c)
      }
    }

    (causeTokens, effectTokens)
  }

  def questionIsEffect(qText:String):Boolean = {

    // Used to detect Causal pattern, in each, X is the EFFECT
    // Example: What can cause X?
    val causeRegex1 = "^[Ww]hat ([a-z]+ ){0,3}cause.+"
    // Example: What could affect the X
    val causeRegex2 = "^[Ww]hat ([a-z]+ ){0,1}[ea]ffects? the .+"
    // Example: What might result in X?
    val causeRegex3 = "^[Wh]hat ([a-z]+ ){0,3}results? in .+"
    // Combine
    val causeRegexes = Array(causeRegex1, causeRegex2, causeRegex3)

    // Used to detect Causal pattern, in each, Y is the CAUSE
    // Example: What is the result of Y?
    val resultRegex1 = "^[Wh]hat ([a-z]+ ){0,3}results? of .+"
    // Example: What effect does Y have on plants?
    val resultRegex2 = "^[Wh]hat ([a-z]+ ){0,3}[ea]ffects? .+"
    // Combine
    val resultRegexes = Array(resultRegex1, resultRegex2)

    // If the question matches the CAUSE regexes (note, none of the questions are of form "What does X cause?")
    for (cRegex <- causeRegexes) {
      // i.e. something like "What causes X?"
    if (qText.toLowerCase.matches(cRegex)) {
        return true
      }
    }
    for (rRegex <- resultRegexes) {
      // i.e. something like "What is the result of Y?"
      if (qText.toLowerCase.matches(rRegex)) {
        return false
      }
    }

    // default...
    false
  }

  def convertTokensToIndices(tokens:Seq[Seq[String]], lex:Lexicon[String]):Seq[Seq[Int]] = {
    tokens.map(seq => convertTokens(seq, lex))
  }

  // Converts the lemma_POSs to the index of the lemma in the provided lexicons
  def convertTokens(tokens:Seq[String], lex:Lexicon[String]): Seq[Int] = {
    val out = for {
      tok <- tokens
      lemma = getLemma(tok)   // Get the lemma portion of the token
      index = lex.get(lemma)
      if index.isDefined
    } yield index.get

    if (out.isEmpty) return Seq(0)

    out
  }

  def padAndCrop(in:Seq[Int], length:Int):Seq[Int] = {
    // If it's too long, crop it
    if (in.length > length) return in.slice(0, length)
    // else, if it's too short, pad it
    val padded = in ++ Seq.fill[Int](length - in.length)(0)
    assert(padded.length == length)
    padded
  }

  // Save the keras input to the desired file, one comma separated row per line
  def saveTo(rows:Seq[Seq[Int]], filename:String): Unit = {
    val pw = new PrintWriter(filename)
    rows.foreach(row => pw.println(row.mkString(",")))
    pw.close()
  }

  val contentTags = Set("NN", "VB")

  val cvFold = 0
  val dir = s"/lhome/bsharp/causal/yahoo/CVForCNN/fold$cvFold/"
  val fold = "train"

  // Question data file
  val fileA = dir + s"A_questionInfo.$fold.tsv"
  val questionTexts = getQuestionTextsFromFile(fileA)

  // Question tokens file -- lemma_POS tab separates, one candidate per line
  val fileB = dir + s"B_questionTokens.$fold.tsv"
  val qTokens = getTokensFromFile(fileB)
  val qTokensFiltered = qTokens.map(toks => filterTokens(toks, contentTags))
  val qLens = qTokensFiltered.map(cand => cand.length)
  val avgQLen = qLens.sum.toDouble / qLens.length.toDouble
  val maxQLen = qLens.max
  println (s"Average question length (after POS filtering): $avgQLen")
  println (s"Max question length: $maxQLen")

  // Canidate tokens file -- lemma_POS tab separates, one candidate per line
  val fileC = dir + s"C_candidateTokens.$fold.tsv"
  val cTokens = getTokensFromFile(fileC)
  val cTokensFiltered = cTokens.map(toks => filterTokens(toks, contentTags))
  val counter = new Counter[String]
  cTokensFiltered.foreach(tokens => tokens.foreach(counter.incrementCount(_)))
  println (counter.topKeys(50))
  val cLens = cTokensFiltered.map(cand => cand.length)
  val avgCLen = cLens.sum.toDouble / cLens.length.toDouble
  val cDevs = cLens.map(len => Math.pow((len - avgCLen), 2))
  println (cLens.sortBy(- _).mkString(", "))
  val cVar = cDevs.sum / cDevs.length.toDouble
  val moreThan31 = cLens.count(_ > 31)
  val maxCLen = cLens.max

  val histBins = Array.fill[Double](30)(0.0)
  for (len <- cLens) {
    for (i <- 0 until 30) {
      if (len < (i + 1) * 10 && len >= i * 10) histBins(i) += 1.0
    }
  }
  println (histBins.mkString("\t"))

  println (s"Average candidate answer length (after POS filtering): $avgCLen")
  println (s"Max candidate answer length: $maxCLen")
  println (s"There are $moreThan31 candidates with more than 31 tokens (out of ${cTokens.length} candidates)")
  println (s"Variance of the candidate answers: $cVar")
  println (s"Standard deviation of candidate answer lengths: ${Math.sqrt(cVar)}")

  // Correctness file -- 0/1, one candidate per line
  val fileD = dir + s"D_candidateCorrectness.$fold.tsv"
  val source = scala.io.Source.fromFile(fileD)
  val correctness = source.getLines().toSeq.map(_.toInt)
  val cLenWithCorr = cLens.zip(correctness)
  val goldMoreThan31 = cLenWithCorr.count(tup => tup._1 > 31 && tup._2 == 1)
  val goldLens = cLenWithCorr.filter(tup => tup._2 == 1).unzip._1
  val avgGoldLen = goldLens.sum.toDouble / goldLens.length.toDouble
  val goldDevs = goldLens.map(len => Math.pow((len - avgGoldLen), 2))
  //println (cLens.sortBy(- _).mkString(", "))
  val goldVar = goldDevs.sum / goldDevs.length.toDouble
  val numQuest = correctness.sum
  println (s"There are $goldMoreThan31 candidates with more than 31 tokens (out of $numQuest gold)")
  println (s"Avg Gold len: $avgGoldLen")
  println ("Gold standard dev: " + Math.sqrt(goldVar))

  // Divide the question and candidate tokens into causes and effects
  val (causeTokens, effectTokens) = selectDirections(questionTexts, qTokensFiltered, cTokensFiltered)

  // Load the lexicons and convert the lemmas to lexicon IDs
  val causeLexFile = "/lhome/bsharp/keras/causal/agiga_causal_mar30_lemmas.src.lex"
  val effectLexFile = "/lhome/bsharp/keras/causal/agiga_causal_mar30_lemmas.dst.lex"

  val causeLexicon = Lexicon.loadFrom[String](causeLexFile)
  val effectLexicon = Lexicon.loadFrom[String](effectLexFile)

  val causeIndices = convertTokensToIndices(causeTokens, causeLexicon)
  val effectIndices = convertTokensToIndices(effectTokens, effectLexicon)

  // Add padding and crop to length if necessary
  val desiredLength:Int = 31
  val paddedCauseIndices = causeIndices.map(padAndCrop(_, desiredLength))
  val paddedEffectIndices = effectIndices.map(padAndCrop(_, desiredLength))

  // Save to keras input file:
  // Save the cause indices
  val causeOut = s"/lhome/bsharp/keras/causal/yahoo/CV/fold$cvFold/yahoo_input.$fold.src.csv"
  saveTo(paddedCauseIndices, causeOut)
  // Save the effect indices
  val effectOut = s"/lhome/bsharp/keras/causal/yahoo/CV/fold$cvFold/yahoo_input.$fold.dst.csv"
  saveTo(paddedEffectIndices, effectOut)

}
