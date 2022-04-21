package org.clulab.odinsynth.evaluation

import org.clulab.odinsynth.Spec
import org.clulab.odinsynth.evaluation.tacred.{PatternDirection, SubjObjDirection, ObjSubjDirection}
import upickle.default.{ReadWriter => RW, macroRW, write, read}
import ujson.Value
import scala.collection.mutable
import ai.lum.odinson.{ExtractorEngine, Document, TokensField, GraphField}

package object fstacred {

  /**
    * For example:
    * val fss = FewShotSentence(
    *             "id",
    *             "docId",
    *             Seq("In", "high", "school", "and", "at", "Southern", "Methodist", "University", ",", "where", ",", "already", "known", "as", "Dandy", "Don", "(", "a", "nickname", "bestowed", "on", "him", "by", "his", "brother", ")", ",", "Meredith", "became", "an", "all-American", "."),
    *             27,
    *             27,
    *             "PERSON",
    *             14,
    *             15,
    *             "PERSON",
    *           )
    * The highligted part is: Seq("(", "a", "nickname", "bestowed", "on", "him", "by", "his", "brother", ")", ",")
    *
    * @param id        - unique identifier
    * @param docId     - doc identifier
    * @param tokens    - the words as a sequence of strings
    * @param subjStart - where the subject starts
    * @param subjEnd   - where the subjects ends (includes the token at subjEnd as part of the subject)
    * @param subjType  - the type of the subject
    * @param objStart  - where the object starts
    * @param objEnd    - where the object ends (includes the token at objEnd as part of the object)
    * @param objType   - the type of the object
    */
  case class FewShotSentence(
    id: String,
    docId: String,
    tokens: Seq[String],
    subjStart: Int,
    subjEnd: Int,
    subjType: String,
    objStart: Int,
    objEnd: Int,
    objType: String,
  ) {

    /**
      * There are two methods needed when getting the sentence with the specifications,
      * with some code duplicate, because we need access to the number of tokens
      * before the first entity and the number of tokens in between the entities
      * 
      * After we replace the entities with their types we lose track of their position
      * An alternative would be to replace the sentence with the entity types and
      * return a new object of this type with the positions replaced, but that would
      * mean a new object creation
      *
      * @param sentenceId the sentence id to use in the specification
      * @return a tuple with the sentence and the specification
      */
    def getSentenceWithTypesAndSpecWithTypes(sentenceId: Int): (Seq[String], Spec) = {
      val (firstType, firstStart, firstEnd) = if (subjStart < objStart) {
        (subjType, subjStart, subjEnd)
      } else {
        (objType, objStart, objEnd)
      }
      val (secondType, secondStart, secondEnd) = if (objStart < subjStart) {
        (subjType, subjStart, subjEnd)
      } else {
        (objType, objStart, objEnd)
      }
      val firstPart: Seq[String] = tokens.take(firstStart)
      val between:   Seq[String] = tokens.slice(firstEnd + 1, secondStart)
      val lastPart:  Seq[String] = tokens.drop(secondEnd + 1)
      (
        firstPart ++ Seq(firstType) ++ between ++ Seq(secondType) ++ lastPart, 
        Spec(docId, sentenceId, firstPart.length, (firstPart.length + between.length + 1) + 1)
      )
    }

    def getSentenceWithTypesAndSpecWithoutTypes(sentenceId: Int): (Seq[String], Spec) = {
      val (firstType, firstStart, firstEnd) = if (subjStart < objStart) {
        (subjType, subjStart, subjEnd)
      } else {
        (objType, objStart, objEnd)
      }
      val (secondType, secondStart, secondEnd) = if (objStart < subjStart) {
        (subjType, subjStart, subjEnd)
      } else {
        (objType, objStart, objEnd)
      }
      val firstPart: Seq[String] = tokens.take(firstStart)
      val between:   Seq[String] = tokens.slice(firstEnd + 1, secondStart)
      val lastPart:  Seq[String] = tokens.drop(secondEnd + 1)
      (
        firstPart ++ Seq(firstType) ++ between ++ Seq(secondType) ++ lastPart, 
        Spec(docId, sentenceId, firstPart.length + 1, (firstPart.length + between.length + 1))
      )
    }

    def getSpecWithoutTypes(sentenceId: Int): Spec = {
        Spec(docId, sentenceId, math.min(subjEnd, objEnd) + 1, math.max(subjStart, objStart))
    }

    // This does not include the entities
    def getHighlightedPart(): Seq[String] = {
      val between: Seq[String] = tokens.slice(math.min(subjEnd, objEnd) + 1, math.max(subjStart, objStart))
      between
    }
    
    def getHighlightedPartWithTypes(): Seq[String] = {
      val (firstType, firstStart, firstEnd) = if (subjStart < objStart) {
        (subjType, subjStart, subjEnd)
      } else {
        (objType, objStart, objEnd)
      }
      val (secondType, secondStart, secondEnd) = if (objStart < subjStart) {
        (subjType, subjStart, subjEnd)
      } else {
        (objType, objStart, objEnd)
      }
      val between:   Seq[String] = tokens.slice(firstEnd + 1, secondStart)

      Seq(firstType) ++ between ++ Seq(secondType)
    }
    
    def getSentenceWithTypes: Seq[String] = {
      val (firstType, firstStart, firstEnd) = if (subjStart < objStart) {
        (subjType, subjStart, subjEnd)
      } else {
        (objType, objStart, objEnd)
      }
      val (secondType, secondStart, secondEnd) = if (objStart < subjStart) {
        (subjType, subjStart, subjEnd)
      } else {
        (objType, objStart, objEnd)
      }
      val firstPart: Seq[String] = tokens.take(firstStart)
      val between:   Seq[String] = tokens.slice(firstEnd + 1, secondStart)
      val lastPart:  Seq[String] = tokens.drop(secondEnd + 1)
      firstPart ++ Seq(firstType) ++ between ++ Seq(secondType) ++ lastPart
    }

    def getFirstType: String = if (subjStart < objStart) subjType else objType 

    def getSecondType: String = if (objStart < subjStart) subjType else objType 

    def getDirectionality: PatternDirection = if (subjStart < objStart) SubjObjDirection else ObjSubjDirection

    def getSubjTokens: Seq[String] = tokens.slice(subjStart, subjEnd + 1)
    def getObjTokens: Seq[String] = tokens.slice(objStart, objEnd + 1)

    def getFirstTypeTokens: Seq[String] = if (subjStart < objStart) getSubjTokens else getObjTokens 
    def getSecondTypeTokens: Seq[String] = if (subjStart < objStart) getObjTokens else getSubjTokens 


    // def getSpecificationWithTypes(sentenceId: Int): Spec = {
      // Spec(docId, sentenceId,)
    // }

  }
  object FewShotSentence {
    implicit val rw: RW[FewShotSentence] = macroRW

    // Create itself from a json value
    // Does not handle exceptions. Expects the json to be good
    def fromJson(jsonValue: Value): FewShotSentence = {
      FewShotSentence(
        id        = jsonValue("id").str,
        docId     = jsonValue("docid").str,
        tokens    = jsonValue("token").arr.map(_.str),
        subjStart = jsonValue("subj_start").num.toInt,
        subjEnd   = jsonValue("subj_end").num.toInt,
        subjType  = jsonValue("subj_type").str,
        objStart  = jsonValue("obj_start").num.toInt,
        objEnd    = jsonValue("obj_end").num.toInt,
        objType   = jsonValue("obj_type").str,
      )
    }
  }

  case class SupportSentence(
    relation: String,
    sentence: FewShotSentence,
  )
  object SupportSentence {
    implicit val rw: RW[SupportSentence] = macroRW
  }
  case class Pattern(
    val rule: String,
    val relation: String
  )
  object Pattern {
    implicit val rw: RW[Pattern] = macroRW
  }

  case class Episode(
    supportSentences: Seq[SupportSentence],
    querySentence:    FewShotSentence,
    queryRelation:    String
  )
  case class EpisodeWithPattern(
    supportSentences: Seq[SupportSentence],
    patterns:         Seq[Pattern],
    querySentence:    FewShotSentence,
    queryRelation:    String
  )

  case class MultiQueryEpisode(
    supportSentences: Seq[SupportSentence],
    querySentences:   Seq[FewShotSentence],
    queryRelations:   Seq[String]
  ) {
    def unroll: Seq[Episode] = querySentences.zip(queryRelations).map { case (queryS, queryR) => Episode(supportSentences, queryS, queryR) }
  }

  /**
    * Hold the confusion matrix as a class with four fields
    *
    * @param tp true positives
    * @param tn true negatives
    * @param fp false positives
    * @param fn false negatives
    */
  case class ConfusionMatrix(
    tp: Int,
    tn: Int,
    fp: Int,
    fn: Int,
  ) {

    /**
      * Override the base toString method to provide the name of the fields
      *
      * @return
      */
    override def toString(): String = f"ConfusionMatrix(tp=$tp,tn=$tn,fp=$fp,fn=$fn)"

    /**
      * Computes the precision as tp/(tp+fp)
      * In case of division by 0, return 0.0
      *
      * @return the precision as a double
      */
    def precision: Double = if((tp + fp) == 0) 0 else tp.toDouble/(tp + fp)

    /**
      * Computes the recall as tp/(tp+fn)
      * In case of division by 0, return 0.0
      *
      * @return the recall as a double
      */
    def recall: Double = if((tp + fn) == 0) 0 else tp.toDouble/(tp + fn)
    
    /**
      * Computes the f1 as (2 * precision * recall) / (precision + recall)
      * In case of division by 0, return 0.0
      *
      * @return the f1 score as a double
      */
    def f1: Double = if(precision + recall == 0) 0 else (2 * precision * recall) / (precision + recall)

    /**
      * Computes the f1 as (2 * precision * recall) / (precision + recall)
      * In case of division by 0, return 0.0
      *
      * @return the f1 score as a double
      */
    def accuracy: Double = if(tp + tn + fp + fn == 0) 0 else (tp + tn) / (tp + tn + fp + fn)


  }


  def allPaths(graph: Map[Int, Seq[Int]], from: Int, to: Int, pathSoFar: mutable.LinkedHashSet[Int]): Seq[Seq[Int]] = {
    val path = pathSoFar ++ Seq(from)
    if (from == to) {
      return Seq(path.toSeq)
    } else {
      val paths = mutable.ListBuffer.empty[Seq[Int]]
      graph(from).foreach { node =>
        if(!path.contains(node)) {
          val newPaths = allPaths(graph, node, to, path)
          newPaths.foreach(it => paths.append(it))
        }
      }
      paths.toSeq
    }
  }

  def shortestPaths(graph: Seq[(Int, Int, String)], from: Int, to: Int): Seq[Int] = {
    val adjacencyMap = mutable.Map.empty[Int, mutable.LinkedHashSet[Int]]//.withDefaultValue(new mutable.LinkedHashSet())
    // Undirected for our purpose
    graph.foreach { case (x, y, _) => 
      if (adjacencyMap.contains(x)) adjacencyMap(x).add(y) else adjacencyMap(x) = mutable.LinkedHashSet(y)
      if (adjacencyMap.contains(y)) adjacencyMap(y).add(x) else adjacencyMap(y) = mutable.LinkedHashSet(x)
    }

    allPaths(adjacencyMap.toMap.mapValues(_.toSeq), from, to, mutable.LinkedHashSet()).minBy(_.length)
  }
  

  /* Consider the dependency graph stored in the Document (@param doc) With this function we obtain 
    * the words (excluding the edges, which are the dependency types) between @param firstWord and @param secondWord 
    * The end result will contain @param firstWord (as .head) and @param secondWord (as .last)
    *
    * @param doc        the Odinson Document containing one sentence
    * @param firstWord  the first word (takes the first index of)
    * @param secondWord the second word (takes the last index of)
    * @return           the words between (including the firstWord and secodnWord) as a Seq
    */
  def wordsOnSyntacticPathBetween(doc: Document, firstWord: String, secondWord: String): Seq[String] = {

    val graph = doc.sentences(0).fields.collect { case GraphField("dependencies", edges, roots) => edges}.head // Unsafe; Dependencies should be there, otherwise we cannot complete
    val tokens = doc.sentences(0).fields.collect { case TokensField("word", tokens) => tokens }.head

    shortestPaths(graph, tokens.indexOf(firstWord.toLowerCase()), tokens.lastIndexOf(secondWord.toLowerCase())).map(tokens)
  }

}