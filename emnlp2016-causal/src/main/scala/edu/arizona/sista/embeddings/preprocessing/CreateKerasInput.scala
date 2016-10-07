package preprocessing

import java.io.PrintWriter
import edu.arizona.sista.struct.Lexicon
import extractionUtils.Word2vec
import scala.collection.mutable.ArrayBuffer
import scala.util.Random

/**
  * Created by bsharp on 4/14/16.
  */
object CreateKerasInput {

  // Open a file and return the tokens on each line
  def getLines(fn:String, lenCutoff:Int = 0):Seq[Seq[String]] = {
    val source = scala.io.Source.fromFile(fn)
    val lines = source.getLines().toSeq
    val out = for {
      line <- lines
      split = line.split(" ").toSeq
      if lenCutoff == 0 || split.length <= lenCutoff
    } yield split
    out
  }


  // Convert the String sequence into an Int sequence, adding unknown words to the lexicon as necessary
  def toIndexes(seq:Seq[Seq[String]]):(Seq[Seq[Int]], Lexicon[String]) = {
    val out = new ArrayBuffer[Seq[Int]]
    val lex = new Lexicon[String]
    lex.add("<<PAD>>")
    for (innerSeq <- seq) {
      val newInnerSeq = new ArrayBuffer[Int]
      for (token <- innerSeq) {
        val index = lex.add(token)
        newInnerSeq.append(index)
      }
      out.append(newInnerSeq.toSeq)
    }
    (out.toSeq, lex)
  }

  // Takes in the vocabulary size and either the src or dst rows and
  // returns an Array[Array[Int]] where each row (outer array) is a vocabulary item
  // and each column (inner Array) is an index in the src/dst where that lexical item occurs
  def makeReverseIndex(vocabSize:Int, in:Seq[Seq[Int]]): Array[Array[Int]] = {
    // Initialize the output array
    val out = Array.fill[ArrayBuffer[Int]](vocabSize)(new ArrayBuffer[Int])

    for {
      row <- in.indices
      lexicalItem <- in(row)
    } out(lexicalItem).append(row)

    // return the reverse index
    out.map(_.toArray)
  }

  def grabRandomNew(rand:Random, corrDsts:Seq[Seq[Int]], dst:Seq[Seq[Int]], revIndex:Array[Array[Int]], i:Int = 0):Seq[Int] = {
    // Find all indices we CAN'T use
//    if (i % 1000 == 0) println ("Grabbing random...")
    val offLimits = getOffLimits(corrDsts, revIndex)
//    if (i % 1000 == 0) println ("Found offlimits...")
    val orig = dst.indices.toSet
//    if (i % 100 == 0) println ("Made set of orig...")
    val possible = orig.diff(offLimits).toArray.sorted
//    if (i % 100 == 0) println ("Did set diff and sorted...")
//    var choose = offLimits.head
//    while (offLimits.contains(choose)) {
//      choose = rand.nextInt(dst.length)
//    }
    val choose = rand.nextInt(possible.length)

    dst(possible(choose))
  }

  def getOffLimits(corrDsts:Seq[Seq[Int]], revIndex:Array[Array[Int]]): Set[Int] = {
//    println ("There are " + corrDsts.length + " corrDsts")
    val flattened = corrDsts.flatten.toSet

//    println ("Flattened length = " + flattened.size)
    val offLimits = new ArrayBuffer[Int]
    for (item <- flattened) offLimits.appendAll(revIndex(item))
//    println ("all appended")
    offLimits.toSet

  }

  // Randomly generate the negative examples
  def negSamples(src:Seq[Seq[Int]], dst:Seq[Seq[Int]], dstVocabSize:Int, nNegPerSrc:Int):Seq[Seq[Int]] = {

    val reverseIndexDst = makeReverseIndex(dstVocabSize, dst)

    val negs = new ArrayBuffer[Seq[Int]]
    val rand = new Random()
    rand.setSeed(426)
    val N = src.length
    for (i <- 0 until N) {
      if (i % 1000 == 0) println (s"Generating neg for src item $i of $N")
      val currSrc = src(i)
      // Get all the dsts that correspond with the *exact* src item
      val allCorrDsts = getAllDsts(currSrc, src, dst)
      //if (i % 1000 == 0) println (s"Got ${allCorrDsts.length} dsts for src item $i of $N")
      //val corrDst = dst(i)
      // Grab a random distribution that has no lexical overlap with any of these items
//      val negDst = grabRandom(rand, N, dst, allCorrDsts, None)

      for (j <- 0 until nNegPerSrc) {
        val negDst = grabRandom(rand, N, dst, allCorrDsts, None)
//        val negDst = grabRandomNew(rand, allCorrDsts, dst, reverseIndexDst)//grabRandom(rand, N, dst, allCorrDsts, None)
        negs.append(negDst)
      }

    }

    negs
  }

  def getAllDsts(currSrc:Seq[Int], src:Seq[Seq[Int]], dst:Seq[Seq[Int]]):Seq[Seq[Int]] = {
    val out = new ArrayBuffer[Seq[Int]]

    // The indexes where the src appears exactly
    val relevantIndexes = src.zipWithIndex.filter(elem => elem._1 == currSrc).unzip._2
    for (i <- relevantIndexes) {
      out.append(dst(i))
    }

    out.toSeq
  }

  // Keep grabbing random destination sequences until you find one with no lexical overlap, return that one
  def grabRandom(rand:Random, N:Int, dst:Seq[Seq[Int]], checkAgainst:Seq[Seq[Int]], potential:Option[Seq[Int]]): Seq[Int] = {
    if (potential.isDefined) {
      if (lexicalOverlap(potential.get, checkAgainst) == 0) {
        return potential.get
      }
    }
    //if (potential.isDefined) println ("There was overlap, so trying again...")
    val next = rand.nextInt(N)
    val newPotential = dst(next)
    grabRandom(rand, N, dst, checkAgainst, Some(newPotential))
  }

  def lexicalOverlap(potential:Seq[Int], checkAgainst:Seq[Seq[Int]]):Int = {
    //println ("Finding lexical overlap...")
    val overlaps = for {
      check <- checkAgainst
      intersection = check.toSet.intersect(potential.toSet)
    } yield intersection.size

    //println ("overlaps.max = " + overlaps.max)
    overlaps.max
  }

  // Find the max length of any row in any of the inputs
  def findMax (s:Seq[Seq[Seq[Int]]]):Int = {
    // Find the individual maxes of each of the seq[seq[int]]s
    val seqMaxes = for {
      seq <- s  // for each of the seq[seq[int]] in s
      rowLengths = seq.map(r=>r.length)  // reduce each of the inner seq[int] to its length
    } yield rowLengths.max    // return the largest of these row lengths

    // Return the global max
    seqMaxes.max
  }

  // Pad a sequence to a given length, using the reserved lexicon item, 0
  def addPadding (s:Seq[Int], max:Int):Seq[Int] = {
    val lenDiff:Int = max - s.length
    s ++ Seq.fill[Int](lenDiff)(0)
  }

  // Save the data to a csv for loading with numpy in keras
  def saveAll (prefix:String, src:Seq[Seq[Int]], dst:Seq[Seq[Int]], neg:Seq[Seq[Int]], negLabel:Int, nNegPerSrc:Int): Unit = {
    println (s"Saving src, dst, and label files to: $prefix.source.csv, $prefix.dest.csv, and $prefix.labels.csv,")

    val allItems = new ArrayBuffer[(Seq[Int], Seq[Int], Int)]
    // Add in the positive examples
    for (i <- src.indices) {
      allItems.append((src(i), dst(i), 1))
    }
    // Add in the negative samples
    // Handle the imbalance in lengths
    for (i <- src.indices) {
      for (j <- 0 until nNegPerSrc) {
        allItems.append((src(i), neg(i*nNegPerSrc + j), negLabel))
      }
    }
    println (s"With positive and negative examples, there are ${allItems.length} total examples")

    // Shuffle the data
    val rand = new Random(426)
    val shuffled = rand.shuffle(allItems).toArray

    // Write the keras input to files
    val pwSrc = new PrintWriter(prefix + ".source.csv")
    val pwDst = new PrintWriter(prefix + ".dest.csv")
    val pwLabels = new PrintWriter(prefix + ".labels.csv")

    for (row <- shuffled) {
      val s = row._1.mkString(",")
      val d = row._2.mkString(",")
      val label = row._3
      pwSrc.println(s)
      pwDst.println(d)
      pwLabels.println(label)
    }

    // Housekeeping
    pwSrc.close()
    pwDst.close()
    pwLabels.close()
  }


  def main(arg:Array[String]): Unit = {

    val lenCutoff:Int = 0
    val negLabel:Int = 0

    // Load the src file
    //val srcFile = "/lhome/bsharp/keras/causal/testScala.src"
    val srcFile = "/lhome/bsharp/causal/nuc_src.txt"
    val srcTokens = getLines(srcFile, lenCutoff)
    // create a src lexicon with 0 reserved
    val (srcTokensIndexes, srcLex) = toIndexes(srcTokens)
    //srcLex.saveTo("/lhome/bsharp/keras/causal/testScala.src.lex")
    srcLex.saveTo(s"/lhome/bsharp/keras/causal/agiga_causal_mar30_lemmas_lenCutoff$lenCutoff.src.lex")

    // Load the dst file
    //val dstFile = "/lhome/bsharp/keras/causal/testScala.dst"
    val dstFile = "/lhome/bsharp/causal/sat_dst.txt"
    val dstTokens = getLines(dstFile, lenCutoff)
    val (dstTokensIndexes, dstLex) = toIndexes(dstTokens)
    //dstLex.saveTo("/lhome/bsharp/keras/causal/testScala.dst.lex")
    dstLex.saveTo(s"/lhome/bsharp/keras/causal/agiga_causal_mar30_lemmas_lenCutoff$lenCutoff.dst.lex")

    // DEBUG:
//    val srcSlice = srcTokensIndexes.slice(0,1000)
//    val dstSlice = dstTokensIndexes.slice(0,1000)

    // Negative sampling
    val nNegPerSrc = 1
    val negDst = negSamples(srcTokensIndexes, dstTokensIndexes, dstLex.size, nNegPerSrc)
    //val negDst = negSamples(srcSlice, dstSlice, nNegPerSrc)

    // Padding?
    val max = findMax(Seq(srcTokensIndexes, dstTokensIndexes, negDst))
    val paddedSrc = srcTokensIndexes.map(row => addPadding(row, max))
    val paddedDst = dstTokensIndexes.map(row => addPadding(row, max))
    val paddedNeg = negDst.map(row => addPadding(row, max))

    // Save everything to files
    val csv = s"/lhome/bsharp/keras/causal/agiga_causal_mar30_lemmas.keras.lenCutoff$lenCutoff.negLab$negLabel.nNeg$nNegPerSrc.padTo$max"
    saveAll(csv, paddedSrc, paddedDst, paddedNeg, negLabel, nNegPerSrc)

//    val csv = s"/lhome/bsharp/keras/causal/SLICE_agiga_causal_mar30_lemmas.keras.lenCutoff$lenCutoff.negLab$negLabel.nNeg$nNegPerSrc.noPadding"
//    saveAll(csv, srcTokensIndexes, dstTokensIndexes, negDst, negLabel)
//    //saveAll(csv, srcSlice, dstSlice, negDst, negLabel)

  }

}

object createWeightsForKeras extends App {

  val kerasDir = "/lhome/bsharp/keras/causal/"
  val rand = new Random(seed = 426)

  val lexFile = kerasDir + "agiga_causal_mar30_lemmas.src.lex"
  //val lexFile = kerasDir + "agiga_causal_mar30_lemmas.dst.lex"
  val lexicon = Lexicon.loadFrom[String](lexFile)

  val w2vDir = "/lhome/bsharp/yoavgo-word2vecf-90e299816bcd/agiga+wiki_mar30_AllLemmas_causal/"
  val embedFile = w2vDir + "agiga+wiki_mar30_AllLemmas_causal_threshold0.dim200vecs"
  //val embedFile = w2vDir + "agiga+wiki_mar30_AllLemmas_causal_threshold0.dim200context-vecs"

  val w2v = new Word2vec(embedFile)
  val dims = w2v.dimensions

  val embedOutFile = kerasDir + "embeddings_src_mar30.csv"
  //val embedOutFile = kerasDir + "embeddings_dst_mar30.csv"
  val embedOut = new PrintWriter(embedOutFile)

  var oov = 0
  for (wordID <- 0 until lexicon.size) {
    val word = lexicon.get(wordID)
    println (s"Word $wordID: $word")
    val vector = w2v.matrix.get(word.toLowerCase)
    if (vector.isDefined) embedOut.println(vector.get.mkString(","))
    else {
      println (s"$word is out of vocabulary, using random weights")
      oov += 1
      val randWeights = new Array[Double](dims)
      for (i <- 0 until dims) randWeights(i) = rand.nextDouble()
      embedOut.println(randWeights.mkString(","))
    }
    embedOut.flush()
  }

  println (s"Out of ${lexicon.size} words, $oov are OOV")

  embedOut.close()

}

