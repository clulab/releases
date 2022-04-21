package org.clulab.odinsynth

import scala.io.StdIn.readLine
import scala.util.control.Breaks._
import ai.lum.odinson._
import org.clulab.processors.clu.CluProcessor
import _root_.org.clulab.processors.{Document => CluDocument}
import org.clulab.processors.fastnlp.FastNLPProcessor

import ai.lum.odinson.extra.ProcessorsUtils
import scala.collection.mutable
import edu.cmu.dynet.Initialize
import org.clulab.odinsynth.evaluation.PandasLikeDataset
import java.io.PrintWriter
import java.io.File
import org.clulab.odinsynth.evaluation.DocumentFromSentences
import org.clulab.odinsynth.scorer.{
  DynamicWeightScorer,
  StaticWeightScorer
}

object SearchOriginal extends App {
  val outputFile = new PrintWriter(new File("test"))
  val scorer = StaticWeightScorer()

  val json =
    """{ "id": "test", "metadata": [], "sentences": [ {"numTokens":21,"fields":[{"$type":"ai.lum.odinson.TokensField","name":"raw","tokens":["SmartPlant","Markup","utilizes","Internet","Explorer","as","the","display","container",",","but","the","product","components","are","installed","on","the","client","PC","."],"store":true},{"$type":"ai.lum.odinson.TokensField","name":"word","tokens":["SmartPlant","Markup","utilizes","Internet","Explorer","as","the","display","container",",","but","the","product","components","are","installed","on","the","client","PC","."]},{"$type":"ai.lum.odinson.TokensField","name":"tag","tokens":["NNP","NNP","VBZ","NNP","NNP","IN","DT","NN","NN",",","CC","DT","NN","NNS","VBP","VBN","IN","DT","NN","NN","."]},{"$type":"ai.lum.odinson.TokensField","name":"lemma","tokens":["SmartPlant","Markup","utilize","Internet","Explorer","as","the","display","container",",","but","the","product","component","be","install","on","the","client","pc","."]},{"$type":"ai.lum.odinson.TokensField","name":"entity","tokens":["O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O"]},{"$type":"ai.lum.odinson.TokensField","name":"chunk","tokens":["B-NP","I-NP","B-VP","B-NP","I-NP","B-PP","B-NP","I-NP","I-NP","O","O","B-NP","I-NP","I-NP","B-VP","I-VP","B-PP","B-NP","I-NP","I-NP","O"]},{"$type":"ai.lum.odinson.GraphField","name":"dependencies","edges":[[1,0,"compound"],[2,1,"nsubj"],[2,4,"dobj"],[2,8,"nmod_as"],[2,9,"punct"],[2,10,"cc"],[2,15,"conj_but"],[2,20,"punct"],[4,3,"compound"],[8,5,"case"],[8,6,"det"],[8,7,"compound"],[13,11,"det"],[13,12,"compound"],[15,13,"nsubjpass"],[15,14,"auxpass"],[15,19,"nmod_on"],[19,16,"case"],[19,17,"det"],[19,18,"compound"]],"roots":[2]}]},{"numTokens":23,"fields":[{"$type":"ai.lum.odinson.TokensField","name":"raw","tokens":["However",",","SmartPlant","Markup","provides","both","initial","installation","and","subsequent","automated","product","updates","from","a","Web","Server",",","simplifying","deployment","and","maintenance","."],"store":true},{"$type":"ai.lum.odinson.TokensField","name":"word","tokens":["However",",","SmartPlant","Markup","provides","both","initial","installation","and","subsequent","automated","product","updates","from","a","Web","Server",",","simplifying","deployment","and","maintenance","."]},{"$type":"ai.lum.odinson.TokensField","name":"tag","tokens":["RB",",","NNP","NNP","VBZ","DT","JJ","NN","CC","JJ","JJ","NN","NNS","IN","DT","NN","NN",",","VBG","NN","CC","NN","."]},{"$type":"ai.lum.odinson.TokensField","name":"lemma","tokens":["however",",","SmartPlant","Markup","provide","both","initial","installation","and","subsequent","automated","product","update","from","a","web","server",",","simplify","deployment","and","maintenance","."]},{"$type":"ai.lum.odinson.TokensField","name":"entity","tokens":["O","O","ORGANIZATION","ORGANIZATION","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O"]},{"$type":"ai.lum.odinson.TokensField","name":"chunk","tokens":["B-ADVP","O","B-NP","I-NP","B-VP","B-NP","I-NP","I-NP","O","B-NP","I-NP","I-NP","I-NP","B-PP","B-NP","I-NP","I-NP","O","B-VP","B-NP","I-NP","I-NP","O"]},{"$type":"ai.lum.odinson.GraphField","name":"dependencies","edges":[[3,2,"compound"],[4,0,"advmod"],[4,1,"punct"],[4,3,"nsubj"],[4,7,"dobj"],[4,12,"dobj"],[4,16,"nmod_from"],[4,17,"punct"],[4,18,"advcl"],[4,22,"punct"],[7,5,"cc:preconj"],[7,6,"amod"],[7,8,"cc"],[7,12,"conj_and"],[12,9,"amod"],[12,10,"amod"],[12,11,"compound"],[16,13,"case"],[16,14,"det"],[16,15,"compound"],[18,19,"dobj"],[18,21,"dobj"],[19,20,"cc"],[19,21,"conj_and"]],"roots":[4]}]} ] }"""
  val doc = Document.fromJson(json)
  val specs = Set(Spec("test", 0, 0, 3), Spec("test", 1, 2, 5))
  // val specs = Set(Spec(doc.id, 0, 7, 10), Spec(doc.id, 1, 6, 9))

  val fieldNames = Set("word", "tag", "lemma")
  val searcher = new Searcher(
    Seq(doc),
    specs.toSet,
    fieldNames,
    Some(1000000),
    Some(outputFile),
    scorer,
    withReward = true
  )

  breakable {
    for (q <- searcher.findAll()) {
      println(s"\nFOUND: ${q.rule.pattern}")
      val response = readLine("Keep looking? [Y/n] ")
      if (response equalsIgnoreCase "n") {
        break
      }
    }
  }
  outputFile.close()

}
