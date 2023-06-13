
import SentenceObjectModifier.determiners
import com.github.tototoshi.csv.CSVWriter
import org.clulab.processors.Processor
import org.clulab.processors.corenlp.CoreNLPProcessor
import org.clulab.processors.fastnlp.FastNLPProcessor
import org.clulab.struct.Interval
import org.clulab.utils.DependencyUtils.findHead
import org.clulab.utils.FileUtils
import org.scalatest.{FunSuite, Succeeded, concurrent}
import org.scalatestplus.play._
import org.scalatest.Matchers.convertToAnyShouldWrapper
import org.scalatest.MustMatchers.convertToAnyMustWrapper

import java.io.File
import java.nio.file.{Files, Paths}
import scala.Console.in

class TestSentenceObjectModifier extends FunSuite {
  val determiners = Array("every", "some", "exactly one", "all but one", "no")
  val adjectives = Array("green", "happy", "sad", "good", "bad")
  val adj_spl = Array("an abnormal", "an elegant")

  test("Empty list") {
    assert(List.empty.size == 0)
  }

  test("Test if POS Tags with tokens are saved as CSV file:") {
    val proc: Processor = new FastNLPProcessor()
    val f = new File("pos_tags_sick_20.csv")
    val writer = CSVWriter.open(f)
    writer.writeRow(List("Premise", "Premise POS Tags","Hypothess","Hypothesis POS Tags"))

      FileUtils.getTextFromResource("/sick-20.tsv").split("\n").drop(1) map { line =>
        val tokens = line.split("\t")
        val premise = proc.annotate(tokens(0))
        val words = premise.sentences(0).words
        val tags = premise.sentences(0).tags

        val hypothesis = proc.annotate(tokens(1))
        val hwords = hypothesis.sentences(0).words
        val htags = hypothesis.sentences(0).tags
        println(String.format(words.toList.mkString(" ") + " " + tags.toList(0).mkString(" ") + "," + hwords.toList.mkString(" ") + " " + htags.toList(0).mkString(" ")+"\n"))
        writer.writeRow(List(words.toList.mkString(" "), tags.toList(0).mkString(" "), hwords.toList.mkString(" "), htags.toList(0).mkString(" ")))
      }
  }
  test("Test Modifying sentence pairs"){
    val sentenceObjectModifier = SentenceObjectModifier
    SentenceObjectModifier.main(Array("",""))
    assert(sentenceObjectModifier != None) mustEqual Succeeded
    //assert(Files.exists(Paths.get("../../../../modified_obj_sick20_fin.csv"))) mustEqual(Succeeded)
  }
  test("Test one sentence ") {
    val sentence = "A blond girl is riding the waves"

    val proc: Processor = new FastNLPProcessor() //uses dependency parser only
    val premise = proc.annotate(sentence)
    val tokens = premise.sentences(0).words
    val tags = premise.sentences(0).tags.toList(0)
    val graphs = premise.sentences(0).graphs.tail.toList(0)._2
    val len = premise.sentences(0).words.length
    graphs.allEdges.foreach(tuple => println(tuple._1, tuple._2, tuple._3))
    val ind = findHead(Interval(0, graphs.allEdges.length), graphs)

    val start = graphs.allEdges(ind)._1-1 //one before
    val end = graphs.allEdges(ind)._2-1
    println(graphs.allEdges(ind)._1, graphs.allEdges(ind)._2, graphs.allEdges(ind)._3)
    tags.slice(start,end).foreach(println)
    val res = tags.sliding(3).toList(0).indexWhere( element => element.canEqual(Array("IN","DT", "NN" )))
    tokens.slice(start, end).foreach(println)
  }
  test("Test one more sentence ") {
    val sentence = "A cramped black train is on the bag of a girl"
    var sentences: Map[String, String] = Map() //mutable

    val proc: Processor = new CoreNLPProcessor() //uses dependency parser only
    val premise = proc.annotate(sentence)
    val tokens = premise.sentences(0).words
    val tags = premise.sentences(0).tags.toList(0)
    val graphs = premise.sentences(0).graphs.tail.toList(0)._2
    val len = premise.sentences(0).words.length
    graphs.allEdges.foreach(tuple => println(tuple._1, tuple._2, tuple._3))
    val ind = findHead(Interval(0, graphs.allEdges.length), graphs)

    val start = graphs.allEdges(ind)._1 - 1
    val end = graphs.allEdges(ind)._2 - 1
    println(graphs.allEdges(ind)._1, graphs.allEdges(ind)._2, graphs.allEdges(ind)._3)
    tags.foreach(println)
    tokens.slice(start, end).foreach(println)

    var restOfSentence = tokens.slice(0, start).toList
    var thisObject = tokens.slice(start, tokens.length).toList
    println(tokens.toList.mkString(" "), start)
    determiners.foreach(modifier => modifier -> {
      //replace only Determiner at first location of this slice
      thisObject = thisObject.updated(0, modifier)
      println((restOfSentence ++ thisObject).mkString(" "))
      sentences += (modifier -> (restOfSentence ++ thisObject).mkString(" "))
    })
    val thisWord = tokens(start).toString
    adjectives.foreach(modifier => sentences += (modifier -> tokens.updated(start, thisWord + " " + modifier).mkString(" ")))
    adj_spl.foreach(modifier => sentences += (modifier -> tokens.updated(start, modifier).mkString(" ")))
    sentences.foreach(tuple => println(tuple._1, tuple._2))
  }

  test("Test sentence pair modification for obj (nominal modifier") {
    val proc: Processor = new FastNLPProcessor()
    //val f = new File("out.csv")
    //val writer = CSVWriter.open(f)
    //writer.writeRow(Seq("Premise", "Hypothess"))
    val f = new File("modified_obj_sick20.csv")
    val writer = CSVWriter.open(f)
    writer.writeRow(List("Modifier", "Premise/Hypothesis/Both", "Part of Premise/Hypothesis Modified", "Premise", "Hypothesis"))

    FileUtils.getTextFromResource("/sick-20.tsv").split("\n").drop(1) map { line =>
      val sentences = line.split("\t")
      val premise = proc.annotate(sentences(0))
      val tokens = premise.sentences(0).words
      val tags = premise.sentences(0).tags
      val graphs = premise.sentences(0).graphs.tail.toList(0)._2
      val ind = findHead(Interval(0, graphs.allEdges.length), graphs)
      val start = graphs.allEdges(ind)._1
      val end = graphs.allEdges(ind)._2
      println("nmod or not " +graphs.allEdges(ind)._3)
      //println(premise.)
      tokens.slice(start, end+1).foreach(println)
      tags.toList(0).slice(start, end+1).foreach(println)
      var p: Map[String, String] = Map()

      List((start, end)).foreach(tuple => {
        val res = tags.toList(0).slice(tuple._1, tuple._2+1).indexWhere(_.toString == "DT")
        if (res > -1) {
          determiners.foreach(modifier => modifier -> {
            p += (modifier -> tokens.updated(res, modifier).mkString(" "))
          })
          adj_spl.foreach(modifier => modifier -> {
            p += (modifier -> tokens.updated(res, modifier).mkString(" "))
          })
          adjectives.foreach(modifier => modifier -> {
            p += (modifier -> tokens.patch(res, tokens(res) :+ modifier, 2).mkString(" "))
          })
        }
        else{
          val pattern = tags.toList(0).slice(start, end+1)
          println("premises -1 " + tuple._1.toString + (tuple._2+1).toString + tags.toList(0).mkString(" "))
        }
      })

      val hypothesis = proc.annotate(sentences(1))
      val htokens = hypothesis.sentences(0).words
      val htags = hypothesis.sentences(0).tags
      val hgraphs = hypothesis.sentences(0).graphs.tail.toList(0)._2
      val ind1 = findHead(Interval(0, hgraphs.allEdges.length), hgraphs)
      val start1 = hgraphs.allEdges(ind1)._1
      val end1 = hgraphs.allEdges(ind1)._2
      println("nmod or not " +hgraphs.allEdges(ind1)._3)
      htokens.slice(start1, end1+1).foreach(println)
      htags.toList(0).slice(start1, end1+1).foreach(println)
      var h: Map[String, String] = Map()

      List((start1, end1)).foreach(tuple => {
        val res = htags.toList(0).slice(tuple._1, tuple._2+1).indexWhere(_.toString == "DT")
        if (res > -1){
          determiners.foreach(modifier => modifier -> {
            h += (modifier -> htokens.updated(res, modifier).mkString(" "))
          })
          adj_spl.foreach(modifier => modifier -> {
            h += (modifier -> htokens.updated(res, modifier).mkString(" "))
          })
          adjectives.foreach(modifier => modifier -> {
            h += (modifier -> htokens.patch(res, htokens(res) :+ modifier, 2).mkString(" "))
          })
        }
        else {
          println("hypothesis -1"+ tuple._1.toString+ (tuple._2+1).toString + htags.toList(0).mkString(" "))

        }
      })

      writer.writeRow(List(ObjTagSequence.NONE.toString,  ObjTagSequence.NONE.toString,  ObjTagSequence.NONE.toString,  sentences(0), sentences(1)))
      p.flatMap(premiseTuple =>
        h.map { hypothesisTuple =>
          premiseTuple._1 match {
            case hypothesisTuple._1 => {
              writer.writeRow(List(premiseTuple._1, "Premise", "Object", premiseTuple._2, sentences(1)))
              writer.writeRow(List(premiseTuple._1, "Hypothesis", "Object", sentences(0), hypothesisTuple._2))
              writer.writeRow(List(premiseTuple._1, "Both", "Object", premiseTuple._2, hypothesisTuple._2))
            }
            case _ => {}
          }
        }
      )
      //writer.writeRow(Seq(words.toList.mkString(" ") + " " + tags.toList(0).mkString(" "), hwords.toList.mkString(" ") + " " + htags.toList(0).mkString(" ") + "\n"))
    }
  }
  test("test"){
    val tags = Array("DT", "NNS", "VBZ", "VBG", "RP", "DT", "NN")
    val res = tags.slice(3, tags.length).indexWhere(_.toString == "DT")
    print(res, tags.slice(3, tags.length)(res))
  }
}
