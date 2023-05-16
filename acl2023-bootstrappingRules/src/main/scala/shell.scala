import scala.util.parsing.json._
import org.clulab.odin.{ExtractorEngine, Mention}
import org.clulab.processors.{Document, Sentence}
import org.clulab.struct.{DirectedGraph, Edge, GraphMap}
import scala.collection.mutable.{ListBuffer, HashSet, HashMap}
import org.clulab.processors.clu.tokenizer.EnglishLemmatizer
import utils._


object shell extends App {
  val input_file = io.Source.fromURL(getClass.getResource(args(0)))
  val jsonString = input_file.mkString
  input_file.close()
  val list:List[Map[String, Any]] = JSON.parseFull(jsonString).get.asInstanceOf[List[Map[String, Any]]]
  val lemmatizer = new EnglishLemmatizer

  var start = 0
  val sentences = new ListBuffer[Sentence]
  for (d<-list){
    val tokens = d.get("token").get.asInstanceOf[List[String]].toArray
    val ner = d.get("stanford_ner").get.asInstanceOf[List[String]].toArray
    val pos = d.get("stanford_pos").get.asInstanceOf[List[String]].toArray
    val head = d.get("stanford_head").get.asInstanceOf[List[Double]]
    val deprel = d.get("stanford_deprel").get.asInstanceOf[List[String]]
    var label = d.get("relation").get.asInstanceOf[String]
    val start_offests = new Array[Int](tokens.length)
    val end_offests = new Array[Int](tokens.length)
    val edges = new ListBuffer[Edge[String]]
    val roots = new HashSet[Int]
    val subj_type = "SUBJ_"+d.get("subj_type").get.asInstanceOf[String]
    val obj_type = "OBJ_"+d.get("obj_type").get.asInstanceOf[String]
    val subj_start = d.get("subj_start").get.asInstanceOf[Double].toInt
    val subj_end = d.get("subj_end").get.asInstanceOf[Double].toInt
    val obj_start = d.get("obj_start").get.asInstanceOf[Double].toInt
    val obj_end = d.get("obj_end").get.asInstanceOf[Double].toInt
    val lemmas = new Array[String](tokens.length)
    for (i<-0 to tokens.length-1){
      lemmas(i) = lemmatizer.lemmatizeWord(tokens(i))
      start_offests(i) = start
      end_offests(i) = start_offests(i) + tokens(i).length
      if (i<= subj_end && i>= subj_start){
        ner(i)= subj_type
      }
      if (i<=obj_end && i>=obj_start){
        ner(i)= obj_type
      }
      start = end_offests(i) + 1
      if (head(i).toInt!=0){
        val edge = Edge(source = head(i).toInt-1, destination = i, relation = deprel(i))
        edges += edge
      }else{
        roots.add(i)
      }
    }
    val dg = new DirectedGraph(edges = edges.toList, roots = roots.toSet)
    val sentence:Sentence = new Sentence(raw = tokens, startOffsets = start_offests, endOffsets = end_offests, words = tokens)
    sentence.entities = Some(ner)
    sentence.tags = Some(pos)
    sentence.lemmas = Some(lemmas)
    sentence.setDependencies(depType = GraphMap.UNIVERSAL_BASIC, deps = dg)
    sentence.setDependencies(depType = GraphMap.UNIVERSAL_ENHANCED, deps = dg)
    sentences += sentence
  }

  println(s"read in all ${sentences.size} sentences")

  val t1 = System.nanoTime
  val doc = new Document(sentences.toArray)
  val source = io.Source.fromURL(getClass.getResource(args(1)))
  val rules = source.mkString
  source.close()
  val extractor = ExtractorEngine(rules)
  val mentions = extractor.extractFrom(doc).sortBy(m => (m.sentence, m.getClass.getSimpleName))
  displayMentions(mentions, doc, args(2))
  val duration = (System.nanoTime - t1) / 1e9d
  println(s"""finish relation extraction in $duration seconds""")

}
