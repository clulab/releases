package edu.arizona.sista.embeddings.postprocessing

import java.io.PrintWriter
import java.util.concurrent.ConcurrentHashMap

import edu.arizona.sista.struct.Counter
import edu.arizona.sista.utils.{Serializer, StringUtils}
import preprocessing.CausalPreprocessingUtils
import preprocessing.agiga.ProcessAgiga

import scala.collection.JavaConverters._
import scala.collection.parallel.ForkJoinTaskSupport

/**
  * Created by bsharp on 5/13/16.
  */
object FindCausalWordPairCounts extends App {

  // Step 1: Load the files
  val causalDir = "/lhome/bsharp/causal/causalOut_mar30/"
  val causalFiles = ProcessAgiga.findFiles(causalDir, "argsC")

  // Step 2: Count the co-occurrence of (c,e) pairs in the files
  val causalCounter = new Counter[String]
  for {
    file <- causalFiles
    pairs = CausalPreprocessingUtils.parseFile(file.getAbsolutePath, lenTreshold = 0)
    (c,e) <- pairs
    cTok <- c.split(" ")
    eTok <- e.split(" ")
    cePairAsString = s"${cTok}__${eTok}"
  } causalCounter.incrementCount(cePairAsString)
  println (s"Finished counting the (c,e) pairs, found ${causalCounter.size} pairs.")

  // Step 3: Make a set of words which occurred in causal constructions
  val wordSet = scala.collection.mutable.Set[String]()
  for (ce <- causalCounter.keySet) {
    val split = ce.split("__")
    val c = split(0)
    val e = split(1)
    wordSet.add(ce)
    //wordSet.add(e + "__" + c)
  }
  println (s"Finished making a set of the c/e words, found ${wordSet.size} words.")

  // Step 4: Store the counter, and a set of the word pairs in the counter
  // Save Counter
  val counterFile = "/lhome/bsharp/causal/pmi/causalCounter_mar30_new.txt"
  val pw = new PrintWriter(counterFile)
  causalCounter.saveTo(pw)
  pw.close()

  // Save Word Set
  val causalSetFile = "/lhome/bsharp/causal/pmi/causalWordSet_mar30_new.ser"
  Serializer.save[scala.collection.mutable.Set[String]](wordSet, causalSetFile)
  println ("Finished saving the counter and serialized word set.")

}

object FindAgigaWordPairCounts {

  val agigaCounter = new ConcurrentHashMap[String, Int]()

  def updateCount(s:String): Unit = {
     val count = agigaCounter.getOrDefault(s, 0)
     agigaCounter.put(s, count + 2)

  }

  def convertToCounter(chm:ConcurrentHashMap[String, Int]): Counter[String] = {
    val c = new Counter[String]
    for {
      e <- chm.entrySet().asScala
      k = e.getKey
      v = e.getValue
    } c.setCount(k, v)
    c
  }

  def main(args:Array[String]): Unit = {
    val props = StringUtils.argsToProperties(args)

    // Step 1: Load the set of words which occurred in causal constructions
    val wordSetFile = props.getProperty("wordset")
    val wordSet = Serializer.load[scala.collection.mutable.Set[String]](wordSetFile)
    println (s"Loaded wordset with ${wordSet.size} words")

    // Step 2: Find the files
    val agigaDir = props.getProperty("agiga.dir")
    val agigaFiles = ProcessAgiga.findFiles(agigaDir, "xml.gz").par

    // Limit the parallelization
    val nthreads = StringUtils.getInt(props, "nthreads", 1)
    agigaFiles.tasksupport = new ForkJoinTaskSupport(new scala.concurrent.forkjoin.ForkJoinPool(nthreads))

    // Step 3: Count the lemma pair occurrences
    //val agigaCounter = new Counter[String]



    val agigaCounterOut = props.getProperty("counter.out")
    val agigaCounterPW = new PrintWriter(agigaCounterOut)

    for (file <- agigaFiles) {

      val doc = ProcessAgiga.agigaDocToDocument(file.getAbsolutePath)
      println(s"Loaded doc from ${file.getAbsolutePath} with ${doc.sentences.length} sentences.")

      // Get the lemmas
      for (s <- doc.sentences) {
        // Retrieve the lemmas which:
        // a) are either NN or VB
        val filteredByPOS = s.lemmas.get.zip(s.tags.get).filter(lemmaAndTag => lemmaAndTag._2.startsWith("NN") || lemmaAndTag._2.startsWith("VB"))
        val lemmas = filteredByPOS.unzip._1
        // b) are found in the causal set
        //val lemmas = lemmasFiltered.filter(lemma => wordSet.contains(lemma))

        // If there are any left after filtering
        if (!lemmas.isEmpty) {
          for (i <- 0 until lemmas.length - 1) {
            for (j <- i + 1 until lemmas.length) {
              // Count both directions
              val forward = s"${lemmas(i)}__${lemmas(j)}"
              val backward = s"${lemmas(j)}__${lemmas(i)}"
              if (wordSet.contains(forward)) updateCount(forward)
              else if (wordSet.contains(backward)) updateCount(backward)

            }
          }
        }

      }
      // Save the counter -- overwriting as you go in case an agiga file dies
      //        if (filesDone % 50 == 0) {
      //          agigaCounter.saveTo(agigaCounterPW)
      //          println ("Counter saved.")
      //        }


    }

    val counter = convertToCounter(agigaCounter)
    counter.saveTo(agigaCounterPW)
    agigaCounterPW.close()
  }



}
