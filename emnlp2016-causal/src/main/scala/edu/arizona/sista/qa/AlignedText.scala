package edu.arizona.sista.qa

/**
 * Provide an unified interface to CQAQuestion and Question that can be used by MakeTranslationMatrix
 * Created by dfried on 6/12/14.
 */
trait AlignedText {

  def questionText: String

  def answersTexts: Iterable[String]

}
