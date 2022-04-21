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
import org.clulab.odinsynth.scorer.DynamicWeightScorer
import org.clulab.dynet.Utils.initializeDyNet


object Search extends App {

  /**
   * 'tokens': ['SmartPlant', 'Markup', 'utilizes', 'Internet', 'Explorer', 'as', 'the', 'display', 'container', ',', 'but', 'the', 'product', 'components', 'are', 'installed', 'on', 'the', 'client', 'PC', '.']
   * 'tokens': ['However', ',', 'SmartPlant', 'Markup', 'provides', 'both', 'initial', 'installation', 'and', 'subsequent', 'automated', 'product', 'updates', 'from', 'a', 'Web', 'Server', ',', 'simplifying', 'deployment', 'and', 'maintenance', '.']
   */
  val p = {
    initializeDyNet()
    new FastNLPProcessor
  }

  val outputFile = new PrintWriter(new File("output2"))
  val scorer = DynamicWeightScorer()
  //// Example cluster 1
  val text = Seq(
    Seq("Ohio", "Republican", "Rep.", "Gillmor", "found", "dead", "in", "PERSON", "apartment", "DATE", ",", "Republican", "aide", "says"),
    Seq("Ohio", "Rep.", "Gillmor", "found", "dead", "in", "PERSON", "apartment", "DATE", ",", "Republican", "aide", "says"),
    Seq("Ohio", "Rep.", "Gillmor", "found", "dead", "in", "PERSON", "house", "DATE", ",", "Republican", "aide", "says"),
    // Seq("This", "thing", "should", "not", "match", "because", "it", "is", "a", "negative", "example"),
  )
  val doc = DocumentFromSentences.documentFromSentences(text, p)
  println(doc.toJson)
  val specs = Seq(
                  Spec(doc.id, 0, 7, 10, Set()), 
                  Spec(doc.id, 1, 6, 9, Set()), 
                  // Spec(doc.id, 2, -1, -1, Set()),
                  // Spec(doc.id, , 0, 0, Set()),
              )



  //// Example 2
  // val text = Seq(
  //   Seq("In", "the", "U.S.", "\"s", "lone", "congressional", "race", ",", "in", "northwest", "Ohio", ",", "Democrat", "Robin", "Weirauch", "and", "Republican", "state", "Rep.", "Bob", "Latta", "won", "their", "primaries", "in", "the", "race", "to", "succeed", "Rep.", "PERSON", ",", "who", "died", "in", "DATE", "from", "a", "fall", "at", "his", "Washington", "apartment", "."),
  //   Seq("PERSON", ",", "who", "died", "in", "DATE", ",", "was", "a", "black", "woman", "who", "worked", "in", "a", "plant", "that", "made", "World", "War", "II", "bombers", "."),
  //   Seq("PERSON", ",", "who", "died", "on", "DATE", "at", "78", ",", "was", "a", "popular", "figure", "in", "the", "opera", "world", "and", "society", "at", "large", ",", "and", "no", "less", "so", "among", "the", "security", "guards", ",", "press", "officers", ",", "high-level", "executives", ",", "secretaries", "and", "many", "others", "who", "work", "at", "Lincoln", "Center", ".")
  // ).map(_.map(_.toLowerCase()))
  // val doc = DocumentFromSentences.documentFromSentences(test, p)
  // val specs = Set(Spec(doc.id, 0, 30, 36), Spec(doc.id, 1, 0, 6), Spec(doc.id, 2, 0, 7))



  //// Example 3 


  // val doc = ProcessorsUtils.convertDocument(p.annotateFromTokens(Seq(Seq("Ohio", "Republican", "Rep.", "Gillmor", "found", "dead", "in", "PERSON", "apartment", "DATE", ",", "Republican", "aide", "says"), Seq("Ohio", "Rep.", "Gillmor", "found", "dead", "in", "PERSON", "apartment", "DATE", ",", "Republican", "aide", "says"))))
  


  // val json = """{ "id": "test", "metadata": [], "sentences": [ {"numTokens":21,"fields":[{"$type":"ai.lum.odinson.TokensField","name":"raw","tokens":["SmartPlant","Markup","utilizes","Internet","Explorer","as","the","display","container",",","but","the","product","components","are","installed","on","the","client","PC","."],"store":true},{"$type":"ai.lum.odinson.TokensField","name":"word","tokens":["SmartPlant","Markup","utilizes","Internet","Explorer","as","the","display","container",",","but","the","product","components","are","installed","on","the","client","PC","."]},{"$type":"ai.lum.odinson.TokensField","name":"tag","tokens":["NNP","NNP","VBZ","NNP","NNP","IN","DT","NN","NN",",","CC","DT","NN","NNS","VBP","VBN","IN","DT","NN","NN","."]},{"$type":"ai.lum.odinson.TokensField","name":"lemma","tokens":["SmartPlant","Markup","utilize","Internet","Explorer","as","the","display","container",",","but","the","product","component","be","install","on","the","client","pc","."]},{"$type":"ai.lum.odinson.TokensField","name":"entity","tokens":["O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O"]},{"$type":"ai.lum.odinson.TokensField","name":"chunk","tokens":["B-NP","I-NP","B-VP","B-NP","I-NP","B-PP","B-NP","I-NP","I-NP","O","O","B-NP","I-NP","I-NP","B-VP","I-VP","B-PP","B-NP","I-NP","I-NP","O"]},{"$type":"ai.lum.odinson.GraphField","name":"dependencies","edges":[[1,0,"compound"],[2,1,"nsubj"],[2,4,"dobj"],[2,8,"nmod_as"],[2,9,"punct"],[2,10,"cc"],[2,15,"conj_but"],[2,20,"punct"],[4,3,"compound"],[8,5,"case"],[8,6,"det"],[8,7,"compound"],[13,11,"det"],[13,12,"compound"],[15,13,"nsubjpass"],[15,14,"auxpass"],[15,19,"nmod_on"],[19,16,"case"],[19,17,"det"],[19,18,"compound"]],"roots":[2]}]},{"numTokens":23,"fields":[{"$type":"ai.lum.odinson.TokensField","name":"raw","tokens":["However",",","SmartPlant","Markup","provides","both","initial","installation","and","subsequent","automated","product","updates","from","a","Web","Server",",","simplifying","deployment","and","maintenance","."],"store":true},{"$type":"ai.lum.odinson.TokensField","name":"word","tokens":["However",",","SmartPlant","Markup","provides","both","initial","installation","and","subsequent","automated","product","updates","from","a","Web","Server",",","simplifying","deployment","and","maintenance","."]},{"$type":"ai.lum.odinson.TokensField","name":"tag","tokens":["RB",",","NNP","NNP","VBZ","DT","JJ","NN","CC","JJ","JJ","NN","NNS","IN","DT","NN","NN",",","VBG","NN","CC","NN","."]},{"$type":"ai.lum.odinson.TokensField","name":"lemma","tokens":["however",",","SmartPlant","Markup","provide","both","initial","installation","and","subsequent","automated","product","update","from","a","web","server",",","simplify","deployment","and","maintenance","."]},{"$type":"ai.lum.odinson.TokensField","name":"entity","tokens":["O","O","ORGANIZATION","ORGANIZATION","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O"]},{"$type":"ai.lum.odinson.TokensField","name":"chunk","tokens":["B-ADVP","O","B-NP","I-NP","B-VP","B-NP","I-NP","I-NP","O","B-NP","I-NP","I-NP","I-NP","B-PP","B-NP","I-NP","I-NP","O","B-VP","B-NP","I-NP","I-NP","O"]},{"$type":"ai.lum.odinson.GraphField","name":"dependencies","edges":[[3,2,"compound"],[4,0,"advmod"],[4,1,"punct"],[4,3,"nsubj"],[4,7,"dobj"],[4,12,"dobj"],[4,16,"nmod_from"],[4,17,"punct"],[4,18,"advcl"],[4,22,"punct"],[7,5,"cc:preconj"],[7,6,"amod"],[7,8,"cc"],[7,12,"conj_and"],[12,9,"amod"],[12,10,"amod"],[12,11,"compound"],[16,13,"case"],[16,14,"det"],[16,15,"compound"],[18,19,"dobj"],[18,21,"dobj"],[19,20,"cc"],[19,21,"conj_and"]],"roots":[4]}]} ] }"""
  // val doc = Document.fromJson(json)
  // val specs = Set(Spec("test", 0, 0, 3), Spec("test", 1, 2, 5))
  // val specs = Set(Spec(doc.id, 0, 7, 10), Spec(doc.id, 1, 6, 9))

  println("\n")
  println("-"*100)
  val fieldNames = Set("word", "tag", "lemma")
  val searcher = new Searcher(doc, specs.toSet, fieldNames, Some(1000000), Some(outputFile), scorer, false)
  val apiCaller = new ApiCaller("http://localhost:8000/score")
  // println(searcher.isSolution(query = Parser.parseBasicQuery("[tag=/N.*/] [tag=\",\"] [tag=WP] [tag=VBD] [tag=IN] [tag=/N.*/]")))
  // println(searcher.isSolution(query = Parser.parseBasicQuery("[tag=NN] [tag=NN] [word=DATE]")))

  // println(searcher.isSolution(query = Parser.parseBasicQuery("[tag=NNP] [tag=NNP] [word=utilizes | word=provides]")))

  // println(searcher.isSolution(query = Parser.parseBasicQuery("""[tag=NN] [tag=","] [word=who] [tag=VBD] [tag=IN] [tag=NN]""")))
  // println(searcher.executeQuery(query = Parser.parseBasicQuery("[tag=NNP] [tag=NNP] [word=utilizes | word=provides]"), false))
  // println(searcher.executeQuery(query = Parser.parseBasicQuery("[tag=NNP] [tag=NNP] [word=utilizes | word=provides]"), true))
  // println(searcher.isSolution(query = Parser.parseBasicQuery("[tag=NNP] [word=Markup] [word=provides | word=utilizes]")))

  // val response = apiCaller.getScores(doc, specs, Seq("[word=SmartPlant] [word=Markup] [□]", "[word=SmartPlant] [word=Markup] □ □"), "[word=SmartPlant] [word=Markup] □")
  // System.exit(1)
  // println(s"response ${response}")


  
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
