package org.clulab.clint

import ai.lum.common.StringUtils._

sealed trait PatternElement
case class Entity(id: Int) extends PatternElement
case class Word(id: Int) extends PatternElement

class Pattern(val elements: Seq[PatternElement]) {

  def withEntityIds: String = {
    elements.map {
      case w: Word => w.id.toString
      case e: Entity => s"@${e.id}"
    }.mkString(" ")
  }

  def withoutEntityIds: String = {
    elements.map {
      case w: Word => w.id.toString
      case e: Entity => "@"
    }.mkString(" ")
  }

  def entityIds: Seq[Int] = {
    for (Entity(id) <- elements) yield id
  }

  def patternString(wordLexicon: IndexToLexeme): String = {
    elements.map {
      case w: Word => wordLexicon.get(w.id).get
      case e: Entity => "@ENTITY"
    }.mkString(" ")

  }

}

object Pattern {
  def apply(string: String): Pattern = {
    val elements = for (e <- string.splitOnWhitespace) yield {
      if (e startsWith "@") {
        Entity(e.drop(1).toInt)
      } else {
        Word(e.toInt)
      }
    }
    new Pattern(elements)
  }
}
