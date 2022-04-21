package org.clulab.odinsynth.rest

import ai.lum.odinson.{
  NamedCapture,
  OdinsonMatch
}
import org.clulab.odinsynth.SearchResult
import play.api.libs.json._
import play.api.libs.functional.syntax._

object JsonUtils {

  /**
   * Provides methods for generating Play JSON from an [[org.clulab.odinsynth.SearchResult]].
   */
  implicit class SearchResultJsonOps(sr: SearchResult) {
    // rather than serializing the case class verbatim, 
    // we're returning a custom JSON format
    def playJson: JsValue = {
      val ncs = namedCapturesToPlayJson(sr.namedCaptures)
      // format: off
      val res = Json.obj(
        "text"         -> sr.text,
        "docId"        -> sr.documentId,
        "sentId"       -> sr.sentenceId,
        // match fields
        "matchStart"   -> sr.matchStart,
        "matchEnd"     -> sr.matchEnd,
        // TODO: add sentence?
        // Enrique: Added total number of hits
        "totalHits"    -> sr.totalHits,
      )
      // format: on
      // only include named captures if we have at least one
      ncs.value.size match {
        case 0 => res
        case nonzero =>
          res + ("namedCaptures" -> ncs)
      }
    }

    def namedCapturesToPlayJson(ncs: Seq[NamedCapture]): JsArray = {
      Json.arr(ncs.map(namedCaptureToPlayJson): _*)
    }

    def namedCaptureToPlayJson(nc: NamedCapture): Json.JsValueWrapper = {
      //OdinsonNamedCapture(name: String, label: Option[String], capturedMatch: OdinsonMatch)
      Json.obj(nc.name -> odinsonMatchToPlayJson(nc.capturedMatch))
    }

    def odinsonMatchToPlayJson(m: OdinsonMatch): JsValue = {
      Json.obj(
        "span" -> Json.obj("start" -> m.start, "end" -> m.end),
        "captures" -> Json.arr(m.namedCaptures.map(namedCaptureToPlayJson): _*)
      )
    }
  }

}