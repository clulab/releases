package org.clulab.odinsynth

import ai.lum.odinson.{
  Sentence => OdinsonSentence,
  NamedCapture,
  _
}

/**
  * Data model for search results.
  */
case class SearchResult(
  text: String,
  documentId: String,
  sentenceId: Int,
  matchStart: Int,
  matchEnd: Int,
  namedCaptures: List[NamedCapture] = Nil,
  sentence: OdinsonSentence,
  totalHits: Int,
)