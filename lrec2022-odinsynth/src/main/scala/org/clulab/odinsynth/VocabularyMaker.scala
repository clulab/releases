package org.clulab.odinsynth

import scala.collection.mutable
import ai.lum.odinson._

class VocabularyMaker(val fieldNames: Set[String]) {

  def apply(document: Document): Map[String, Array[String]] = {
    apply(Seq(document))
  }

  def apply(documents: Seq[Document]): Map[String, Array[String]] = {
    val vocabs = mutable.Map.empty[String, mutable.Set[String]]
    for {
      d <- documents
      s <- d.sentences
      f <- s.fields.collect { case f: TokensField => f }
      if fieldNames contains f.name
    } vocabs.getOrElseUpdate(f.name, mutable.Set.empty[String]) ++= f.tokens
    vocabs.mapValues(_.toArray).toMap
  }

  def apply(documents: Seq[Document], specs: Seq[Set[Spec]]): Map[String, Array[String]] = {
    val vocabs = mutable.Map.empty[String, mutable.Set[String]]
    for {
      (doc, specs) <- documents.zip(specs.map{ it => it.toSeq.sortBy(_.sentId) })
      (sentence, spec) <- doc.sentences.zip(specs)
      field <- sentence.fields.collect { case f: TokensField => f }
      if fieldNames.contains(field.name)
    } {
      vocabs.getOrElseUpdate(field.name, mutable.Set.empty[String]) ++= field.tokens.slice(spec.start, spec.end)
    }
    
    vocabs.mapValues(_.toArray).toMap
  }

}
