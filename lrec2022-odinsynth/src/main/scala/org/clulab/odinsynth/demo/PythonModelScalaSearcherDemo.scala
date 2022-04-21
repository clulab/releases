package org.clulab.odinsynth.demo

import org.clulab.odinsynth.scorer.DynamicWeightScorer
import edu.cmu.dynet.Initialize
import scala.collection.mutable
import org.clulab.odinsynth.Spec
import org.clulab.odinsynth.evaluation.DocumentFromSentences
import org.clulab.processors.fastnlp.FastNLPProcessor
import org.clulab.odinsynth.Searcher
import java.{util => ju}
import org.clulab.dynet.Utils.initializeDyNet

/**
  * A runnable objects which allows for a rapid demo of the system
  * It should serve as an example of how to put everything together and run the system
  * sbt -J-Xmx16g 'runMain org.clulab.odinsynth.demo.PythonModelScalaSearcherDemo'
  */
object PythonModelScalaSearcherDemo extends App {
  
  // Initialization of processors with DyNet
  val p = {
    initializeDyNet()
    new FastNLPProcessor
  }



  // We are using this object to score states using its score(sentences, specs, nextStates, currentState) method
  // We are using a dynamic weight scorer, which means that we are making a call to the python model using REST endpoints
  // Specifically, we are making a POST using the data provided in the score method and waiting for a response
  val scorer = DynamicWeightScorer("http://localhost:8001")

  val sentences = Seq(
    Seq("International", "Federation", ",", "the", "problems", "faced", "by", "children", "living", "with", "HIV", "/", "AIDS", "are", "one", "of", "the", "most", "significant", "issues", "confronting", "it", "and", "its", "membership", "."),
    Seq("Some", "also", "hope", "that", "such", "a", "shocking", "death", "will", "impact", "the", "public", "\"s", "opinion", "of", "what", "has", "become", "one", "of", "the", "most", "controversial", "issues", "in", "law", "enforcement", "."),
  )
  val specsWithoutDoc = Seq(
    Spec("", 0, 15, 18),
    Spec("", 1, 19, 22),
  )

  private lazy val odinsonDocumentCache = "/home/rvacareanu/projects/odinsynth_cache3/"
  val doc = synchronized { DocumentFromSentences.documentFromSentencesAndCache(sentences, p, odinsonDocumentCache, false) }
  val specs = specsWithoutDoc.map(_.copy(docId = doc.id))

  val searcher = new Searcher(docs = Seq(doc), specs = specs.toSet, fieldNames = Set("word", "tag", "lemma"), maxSteps = Some(1000), writer = None, scorer = scorer, withReward = false)  
  println(ju.Calendar.getInstance().getTime())
  println(searcher.findFirst())
  println(ju.Calendar.getInstance().getTime())

}
