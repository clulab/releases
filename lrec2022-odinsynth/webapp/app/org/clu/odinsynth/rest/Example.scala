package org.clulab.odinsynth.rest

import ai.lum.odinson.{
  Sentence => OdinsonSentence,
  NamedCapture,
  _
}
import org.clulab.odinsynth.Spec
import play.api.libs.json._
import play.api.libs.functional.syntax._

/**
 * Denotes a start and end point in some sequence.  Used to construct an [[org.clulab.odinsynth.rest.PositiveExample]].
 */
case class Span(
  start: Int,
  end: Int
)

/**
 * Represents a span that should be matched. Translates to an [[org.clulab.odinsynth.Span]].
 */
case class PositiveExample(
  val docId: String,
  val sentId: Int,
  val span: Span,
  val captures: Seq[Span] = Nil // context
  //val captures: 
) extends Example {

  // FIXME: add optional named captures
  def toSpec: Spec = Spec(
    docId, 
    sentId, 
    span.start, 
    span.end,
    // Set[(String, Spec)]
    // FIXME: why is this a set?
    captures.map{ capture => 
      ("CONTEXT" -> Spec(docId, sentId, capture.start, capture.end))
    }.toSet
  )

  def toEntitySpec: Seq[Spec] = {
    assert(captures.size == 1)
    Seq(
      Spec(docId, sentId, span.start, span.end),
      Spec(docId, sentId, captures.head.start, captures.head.end),
    ).sortBy(_.start)
  }

  def maskOrder: Seq[Boolean] = {
    ((span.start, false) :: captures.map(it => (it.start, true)).toList).sortBy(_._1).map(_._2)
  }

}

/**
 * Represents something that should **not** be matched.
 */
case class NegativeExample(
  val docId: String,
  val sentId: Int,
) extends Example

/**
  * Data model for abbreviated search results.
  */
sealed trait Example {
  val docId: String
  val sentId: Int
}

object ExampleUtils {
  def fromPlayJson(json: JsValue): Example = {
    try {
      val docId = (json \ "docId").as[String]
      val sentId = (json \ "sentId").as[Int]
      json.as[JsObject].keys.contains("span") match {
        case false =>
          NegativeExample(
            docId = docId, 
            sentId = sentId
          )
        case true => 
          val start = (json \ "span" \ "start").as[Int]
          val end = (json \ "span" \ "end").as[Int]
          PositiveExample(
            docId = docId, 
            sentId = sentId,
            span = Span(start = start, end = end),
            // FIXME: what is the frontend **actually** sending us???
            captures = (json \ "captures").as[JsArray].value.map{ entry =>
              Span(
                start = (entry \ "span" \ "start").as[Int],
                end = (entry \ "span" \ "end").as[Int]
              )
            }
          )
      }
    } catch {
      case e : Throwable =>
        throw e
    }
  }

}