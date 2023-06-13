
import breeze.linalg.max
import com.github.tototoshi.csv.CSVWriter
import org.clulab.processors.examples.DocumentationExample.mkString
import org.clulab.processors.{Document, Processor}
import org.clulab.processors.fastnlp.FastNLPProcessor
import org.clulab.struct.{GraphMap, Interval}
import org.clulab.utils.DependencyUtils.findHead
import org.clulab.utils.FileUtils

import java.io.File
import scala.collection.mutable.ArrayBuffer

//TODO: dont check all patterns each time. Only one pattern exists at a time.
//TODO: Add unit tests and generate csv and print.

// Creating Enumeration

object SentenceObjectModifier {

  val determiners = Array("every", "some", "exactly one", "all but one", "no")
  val adjectives = Array("green", "happy", "sad", "good", "bad")
  val adj_spl = Array("an abnormal", "an elegant")
  val proc: Processor = new FastNLPProcessor()

  def getModifiedSentences(start: Int, words: Array[String], pattern: String): Map[String, String] = {

    //add all modifiers and remove grammatically incorrect ones later (add now, remove later approach)
    var sentences: Map[String, String] = Map() //mutable
    println("This pattern is being modified ", pattern)
    var restOfSentence = words.slice(0, start).toList
    var thisObject = words.slice(start, words.length).toList
    println(words.toList.mkString(" "), start)
    determiners.foreach(modifier => modifier -> {
      //replace only Determiner at first location of this slice
      thisObject = thisObject.updated(0, modifier)
      println((restOfSentence ++ thisObject).mkString(" "))
      sentences += (modifier -> (restOfSentence ++ thisObject).mkString(" "))
    }) //replace DT
    val thisWord = words(start).toString
    /*could have been more of a conditional modification but too much filtering. So add all modifiers and remove grammatically incorrect ones later */
    pattern match {

      case "DT+JJ+NN" => {
        adjectives.foreach(modifier => sentences += (modifier -> words.updated(start, thisWord + " " + modifier).mkString(" ")))
        adj_spl.foreach(modifier => sentences += (modifier -> words.updated(start, modifier).mkString(" ")))
      }
      case "JJ+NN" => {
        adjectives.foreach(modifier => sentences += (modifier -> words.updated(start, thisWord + " " + modifier).mkString(" ")))
        adj_spl.foreach(modifier => sentences += (modifier -> words.updated(start, modifier).mkString(" ")))
      }
      case "JJ+NNS" => {
        adjectives.foreach(modifier => sentences += (modifier -> words.updated(start, thisWord + " " + modifier).mkString(" ")))
        adj_spl.foreach(modifier => sentences += (modifier -> words.updated(start, modifier).mkString(" ")))
      } // send spl_adj
      case "DT+JJ+NNS" => {
        adjectives.foreach(modifier => sentences += (modifier -> words.updated(start, thisWord + " " + modifier).mkString(" "))) //insert in place of DT with same DT + adj
        adj_spl.foreach(modifier => sentences += (modifier -> words.updated(start, modifier).mkString(" ")))
      } // replace DT

      case "IN+NNS" => {
        sentences = Map()
        adjectives.foreach(modifier => sentences += (modifier -> words.updated(start, thisWord + " " + modifier).mkString(" "))) //insert before NNS
        adj_spl.foreach(modifier => sentences += (modifier -> words.updated(start, words(start) + " " + modifier).mkString(" "))) //insert before NNS
      }
      case "IN+DT+NN" => {
        adjectives.foreach(modifier => sentences += (modifier -> words.updated(start, thisWord + " " + modifier).mkString(" ")))
        adj_spl.foreach(modifier => sentences += (modifier -> words.updated(start, modifier).mkString(" ")))
      }
      case "IN+DT+NNS" => {
        adjectives.foreach(modifier => sentences += (modifier -> words.updated(start, thisWord + " " + modifier).mkString(" ")))
        adj_spl.foreach(modifier => sentences += (modifier -> words.updated(start, modifier).mkString(" ")))
      }
      case _ => {
        adjectives.foreach(modifier => sentences += (modifier -> words.updated(start, thisWord + " " + modifier).mkString(" ")))
        adj_spl.foreach(modifier => sentences += (modifier -> words.updated(start, words(start) + " " + modifier).mkString(" ")))
      }
    }
    sentences
  }

  def getMatchingObjectTags(tags: Array[String]) = {
    //val ind = ObjTagSequence.values.toList.find(_.toString == tags.sliding(2).toList.last.mkString("+"))
    var t = tags

    var thisLen = 0
    var thisPattern: String = ObjTagSequence.NONE.toString
    if (tags.length > 3) {
      t = tags.slice(tags.length - 3, tags.length).toArray
      println(t.mkString("+"))
    }

    val nModThreeTagSeq = t.mkString("+")
    var nModTwoTagSeq = t.slice(1, tags.length).mkString("+")

    if (tags.length == 3 && ObjTagSequence.values.toList.find(_.toString == nModThreeTagSeq) != None) {
        thisPattern = nModThreeTagSeq
        thisLen = 3
    }
    else {
      if(tags.length == 2) {
        nModTwoTagSeq = t.mkString("+")
      }
      if ( ObjTagSequence.values.toList.find(_.toString == nModTwoTagSeq) != None) {
        thisPattern = nModTwoTagSeq
        thisLen = 2
      }
    }
    println(thisPattern, thisLen)
    (thisPattern, thisLen)
  }

  def main(args: Array[String]) {
    val f = new File("modified_obj_sick20_fin.csv")
    val writer = CSVWriter.open(f)
    writer.writeRow(List("Modifier",  "Premise/Hypothesis/Both",  "Part of Premise/Hypothesis Modified", "Premise", "Hypothesis"))

    var modified_sentences: List[(String, String, String, String, String)] = List[(String, String, String, String, String)]()
      // Drop the first line that is the header in this "tsv"
      FileUtils.getTextFromResource("/sick-20.tsv").split("\n").drop(1) map { line =>
        val sentencesTuple = line.split("\t")

        val premise = proc.annotate(sentencesTuple(0))
        val ptokens = premise.sentences(0).words
        val ptags = premise.sentences(0).tags
        val pgraphs = premise.sentences(0).graphs.tail.toList(0)._2
        val plen = premise.sentences(0).words.length

        val hypothesis = proc.annotate(sentencesTuple(1))
        val htokens = hypothesis.sentences(0).words
        val htags = hypothesis.sentences(0).tags
        val hgraphs = hypothesis.sentences(0).graphs.tail.toList(0)._2
        val hlen = hypothesis.sentences(0).words.length

        modified_sentences = (ObjTagSequence.NONE.toString, ObjTagSequence.NONE.toString, ObjTagSequence.NONE.toString, sentencesTuple(0), sentencesTuple(1)) :: modified_sentences
        writer.writeRow(List(ObjTagSequence.NONE.toString,  ObjTagSequence.NONE.toString,  ObjTagSequence.NONE.toString,  sentencesTuple(0), sentencesTuple(1)))
        println(sentencesTuple(0))

        var lastOfThisHead = pgraphs.allEdges.slice(findHead(Interval(0, pgraphs.allEdges.length), pgraphs)+1,pgraphs.allEdges.length )
        if(lastOfThisHead.length == 0){
          lastOfThisHead = pgraphs.allEdges.slice(findHead(Interval(0, pgraphs.allEdges.length), pgraphs)-2,pgraphs.allEdges.length )
        }
        var start = 0
        /* This one for Premise sentence */
        if (lastOfThisHead.tail.length >0 && lastOfThisHead.tail(0)._3 == "det")
          {
            start = lastOfThisHead.tail(0)._2
          }
        else
          {
            if (lastOfThisHead.length == 1) {
              start = lastOfThisHead(0)._2
            }
            else{
              start = lastOfThisHead(lastOfThisHead.length-2)._2
            }
          }
        println(start)
        val premisePatternIndexTuple = getMatchingObjectTags(ptags.toList(0).slice(start, ptags.toList(0).length) )

        println(sentencesTuple(1))
        /* This one for Hypothesis sentence */
        var lastOfThisHeadH = hgraphs.allEdges.slice(findHead(Interval(0, hgraphs.allEdges.length), hgraphs) + 1, hgraphs.allEdges.length)
        var start2 = 0
        if (lastOfThisHeadH.length == 0) {
          lastOfThisHeadH = hgraphs.allEdges.slice(findHead(Interval(0, hgraphs.allEdges.length), hgraphs) -2, hgraphs.allEdges.length)
        }
        if (lastOfThisHeadH(lastOfThisHeadH.length-1)._3 == "det") {
          start2 = lastOfThisHeadH(lastOfThisHeadH.length-1)._2
        }
        else{
          start2 = lastOfThisHeadH(lastOfThisHeadH.length-2)._2
        }
        print(start2)
        var hypothesisPatternIndexTuple = ("", 0)

        // Handle COP (i.e. rightmost Preposition Phrase (PP) but get the leftmost IN+DT+NN : in this one case, on the bag of a girl = IN+DT+NN + IN+DT+NN
        // get left most IN+DT+NN within PP node
        if (sentencesTuple(1).contains("on the bag of"))
          {
            val thisPhrase = sentencesTuple(1).slice(sentencesTuple(1).indexOf("on the bag"),sentencesTuple(1).indexOf("bag")+3)
            val i  = htokens.lastIndexOf("the").toInt
            if(htags.toList(0)(i) =="DT")
              {
                hypothesisPatternIndexTuple =  "IN+DT+NN" -> (i-1).toInt
              }
          }
          else{
            hypothesisPatternIndexTuple = getMatchingObjectTags(htags.toList(0).slice(start2, htags.toList(0).length))
          }

        /* Combine the modified premise, hypotheses in the format */
        if (premisePatternIndexTuple._1 != ObjTagSequence.NONE.toString && hypothesisPatternIndexTuple._1 != ObjTagSequence.NONE.toString) {
          val modifiedPremises = getModifiedSentences(plen - premisePatternIndexTuple._2, ptokens, premisePatternIndexTuple._1)
          val modifiedHypothesis = getModifiedSentences(hlen - hypothesisPatternIndexTuple._2, htokens, hypothesisPatternIndexTuple._1)
          modifiedPremises.flatMap(premiseTuple =>
            modifiedHypothesis.map { hypothesisTuple =>
              premiseTuple._1 match {
                case hypothesisTuple._1 => {
                  writer.writeRow(List(premiseTuple._1,  "Premise",  "Object",  premiseTuple._2, sentencesTuple(1)))
                  writer.writeRow(List(premiseTuple._1,  "Hypothesis" ,  "Object", sentencesTuple(0), hypothesisTuple._2))
                  writer.writeRow(List(premiseTuple._1,  "Both","Object", premiseTuple._2, hypothesisTuple._2))
                }
                case _ => {}
              }
            }
          )
        }

        }

      // print each premise, hypothesis pairs
    }

  }
