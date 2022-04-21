package org.clulab.odinsynth

import ai.lum.common.Interval
import ai.lum.odinson._
import scala.util.parsing.json.JSON

case class Spec(
    docId: String,
    sentId: Int,
    start: Int,
    end: Int,
    captures: Set[(String, Spec)]
) {
  val interval = Interval.open(start, end)
}
object Spec {

  def apply(docId: String, sentId: Int, start: Int, end: Int): Spec = {
    Spec(docId, sentId, start, end, Set.empty)
  }

  def fromOdinsonMentions(ms: Seq[Mention]): Set[Spec] = {
    ms.map(fromOdinsonMention).toSet
  }

  def fromOdinsonMention(m: Mention): Spec = {
    fromOdinsonMatch(m.odinsonMatch, m.docId, m.sentenceId.toInt)
  }

  def fromOdinsonMatch(m: OdinsonMatch, docId: String, sentId: Int): Spec = {
    val captures = m.namedCaptures.map(c =>
      (c.name, fromOdinsonMatch(c.capturedMatch, docId, sentId))
    )
    // FIXME: shouldn't captures -> mask?
    Spec(docId, sentId, m.start, m.end, captures.toSet)
  }

  def fromString(str: String): Seq[Spec] = {
    val numbersAsInteger = { input: String => Integer.parseInt(input) }
    JSON.globalNumberParser = numbersAsInteger
    return JSON.parseFull(str) match {
      case Some(x: Map[String, List[Map[String, Any]]]) => {
        x("specs").map(it =>
          it.values.toSeq match {
            case Seq(docId: String, sentId: Int, start: Int, end: Int) =>
              Spec(docId, sentId, start, end)
          }
        )
      }
      case _ => throw new RuntimeException("Unexpected json format")
    }
  }
}