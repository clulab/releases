package org.clulab

import ai.lum.odinson.Document
import ai.lum.odinson.TokensField
import scala.collection.mutable

package object odinsynth {

  val holeGlyph = "\u25a1" // WHITE SQUARE
  val emptySetGlyph = "\u2205" // EMPTY SET

  implicit class EnhancedType[A](a: A) {
    def let[B](f: A => B) = f(a)
  }

  /**
  * Extension function on any type with an implicit conversion to a collection
  *
  * @param coll: any collection-like
  *             It must have an implicit conversion to Seq[T]
  */
  implicit class EnhancedColl[T, CT](coll: CT)(implicit viewCollAsSeq: CT => Seq[T]) {
    /**
    * NOTE This method is in the API starting from scala 2.13
    * NOTE Also, this implementation will not work starting from 2.13 due to changes in CanBuildFrom 
    * 
    * @param f: function that maps an element of the collection to a new type B, on which the distinction should be made
    * @param cbf: implicit parameter for returning the same type of collection as the one on which this method is called
    */
    def distinctBy[B](f: T => B)(implicit cbf: scala.collection.generic.CanBuildFrom[CT, T, CT]): CT = {
      val builder = cbf()
      // Keep only first AND maintain the order
      val seen = scala.collection.mutable.LinkedHashSet.empty[B]
      coll.foreach { el => f(el).let { fel => if(!seen.contains(fel)) { builder += el; seen += fel } } }
      builder.result()
    }

  }


  implicit class EnhancedResource[A <: { def close(): Unit }](r: A) {
    def use[B](f: (A) => B): B = {
      try {
        f(r)
      }
      finally {
        r.close()
      }
    }
  }

  def using[A <: { def close(): Unit }, B](closeable: A)(f: A => B): B = closeable.use(f)

  def extractSentencesFromDoc(od: Document): Seq[Seq[String]] = {
    val result = mutable.ListBuffer.empty[Seq[String]]
    for {
      s <- od.sentences
      f <- s.fields.collect { case f: TokensField => f }
      if f.name == "word"
    } result += f.tokens
    result.toSeq
  }

  def generalizedCross[CCT, CT, T](colls: CCT)(implicit viewCCollAsSeq: CCT => Seq[CT], viewCollAsSeq: CT => Seq[T]): Seq[Seq[T]] = {
    if(colls.isEmpty) {
      Seq(Seq.empty)
    } else {
      colls.head.flatMap { it => generalizedCross(colls.tail).map { partialResult => Seq(it) ++ partialResult } }
    }
  }
  
}
