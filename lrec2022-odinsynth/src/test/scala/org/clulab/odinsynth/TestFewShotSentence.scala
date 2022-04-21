package org.clulab.odinsynth

import org.scalatest._
import ai.lum.odinson._
import org.clulab.odinsynth.evaluation.fstacred.FewShotSentence
import org.clulab.odinsynth.evaluation.tacred.SubjObjDirection
import org.clulab.odinsynth.evaluation.tacred.ObjSubjDirection

class TestFewShotSentence extends FlatSpec with Matchers {
  val fss = FewShotSentence(
          "id",
          "docId",
          Seq("In", "high", "school", "and", "at", "Southern", "Methodist", "University", ",", "where", ",", "already", "known", "as", "Dandy", "Don", "(", "a", "nickname", "bestowed", "on", "him", "by", "his", "brother", ")", ",", "Meredith", "became", "an", "all-American", "."),
          27,
          27,
          "PERSON",
          14,
          15,
          "PERSON",
        )
  
  "getHighlightedPart" should "return a sequence with only the words in between the two available entities" in {
    val highlighted = fss.getHighlightedPart()
    assert(highlighted.mkString(" ") == "( a nickname bestowed on him by his brother ) ,")
  }

  "getHighlightedPartWithTypes" should "return a sequence with only the words between the two available entities and the entities (their types)" in {
    val highlighted = fss.getHighlightedPartWithTypes()
    assert(highlighted.mkString(" ") == "PERSON ( a nickname bestowed on him by his brother ) , PERSON")
  }

  "getDirectionality" should "return the direction; that is, wherher it is SUBJ <..> OBJ or OBJ <..> SUBJ" in {
    val direction = fss.getDirectionality
    assert(direction == ObjSubjDirection)
  }

  "getSpecWithoutTypes" should "return the specification without types" in {
    val specWithoutTypes = fss.getSpecWithoutTypes(0)
    assert(specWithoutTypes == Spec(fss.docId, 0, 16, 27))
  }

  "getSentenceWithTypesAndSpecWithTypes" should "return sentence with types and the specification with types" in {
    val (sentence, spec) = fss.getSentenceWithTypesAndSpecWithTypes(0)
    assert(sentence.mkString(" ") == "In high school and at Southern Methodist University , where , already known as PERSON ( a nickname bestowed on him by his brother ) , PERSON became an all-American .")
    assert(spec == Spec(fss.docId, 0, 14, 27))
  }

  "getSentenceWithTypesAndSpecWithoutTypes" should "return sentence with types and the specification without types" in {
    val (sentence, spec) = fss.getSentenceWithTypesAndSpecWithoutTypes(0)
    assert(sentence.mkString(" ") == "In high school and at Southern Methodist University , where , already known as PERSON ( a nickname bestowed on him by his brother ) , PERSON became an all-American .")
    assert(spec == Spec(fss.docId, 0, 15, 26))
  }

}
