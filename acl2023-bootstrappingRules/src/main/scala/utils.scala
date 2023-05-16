import org.clulab.odin._
import org.clulab.processors.{Document, Sentence}

import java.io._

package object utils {

  def displayMentions(mentions: Seq[Mention], doc: Document, out:String): Unit = {
    val mentionsBySentence = mentions groupBy (_.sentence) mapValues (_.sortBy(_.start)) withDefaultValue Nil
    val pw = new PrintWriter(new File(out ))
    for ((s, i) <- doc.sentences.zipWithIndex) {
//      println(s"sentence #$i")
//      println(s.getSentenceText)
//      println("Tokens: " + (s.words.indices, s.words, s.tags.get).zipped.mkString(", "))
//      printSyntacticDependencies(s)
//      println

      val sortedMentions = mentionsBySentence(i).sortBy(_.label)
      val (events, entities) = sortedMentions.partition(_ matches "Relation")
      val (tbs, rels) = entities.partition(_.isInstanceOf[TextBoundMention])
      val sortedEntities = tbs ++ rels.sortBy(_.label)
      if (events.size!=0){
        for (i<- 0 to events.size-1){
          if (i==events.size-1) {
            displayMention(events(i), true, pw)
          }else {
            displayMention(events(i), false, pw)
          }
        }
      }else{
        pw.write("no_relation\tNone\n")
//        println("no_relation\tNone")
      }
    }
    pw.close
  }

  def printSyntacticDependencies(s:Sentence): Unit = {
    if(s.dependencies.isDefined) {
      println(s.dependencies.get.toString)
    }
  }

  def displayMention(mention: Mention, last: Boolean, pw: PrintWriter) {
    val boundary = s"\t${"-" * 30}"
//    println(s"${mention.labels} => ${mention.text}")
//    println(boundary)
//    println(s"\tRule => ${mention.foundBy}")
    val mentionType = mention.getClass.toString.split("""\.""").last
//    println(s"\tType => $mentionType")
//    println(boundary)
    mention match {
      case tb: TextBoundMention =>
        println(s"\t${tb.labels.mkString(", ")} => ${tb.text}")
      case em: EventMention =>
        if (last) {
          pw.write(s"${mention.labels(0)}\t(${em.trigger.start},${em.trigger.end})\t${em.foundBy}\n")
        } else {
          pw.write(s"${mention.labels(0)}\t(${em.trigger.start},${em.trigger.end})\t${em.foundBy}|")
        }
      //        displayArguments(em)
      case rel: RelationMention =>
        if (last) {
          pw.write(s"${mention.labels(0)}\t(${rel.start},${rel.end})\t${rel.foundBy}\n")
//          println(s"${mention.labels(0)}\t(${rel.start},${rel.end})\t${rel.foundBy}")
        } else
          pw.write(s"${mention.labels(0)}\t(${rel.start},${rel.end})\t${rel.foundBy}|")
      case _ => ()
    }
//    println(s"$boundary\n")
  }


  def displayArguments(b: Mention): Unit = {
    b.arguments foreach {
      case (argName, ms) =>
        ms foreach { v =>
          println(s"\t$argName ${v.labels.mkString("(", ", ", ")")} => ${v.text}")
        }
    }
  }
}