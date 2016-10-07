package edu.arizona.sista.qa.index

import collection.mutable.ListBuffer
import DependencyBuilder._
import edu.arizona.sista.struct.{DirectedGraphEdgeIterator, DirectedGraph}

/**
 * 
 * User: mihais
 * Date: 4/3/13
 */
class DependencyBuilder(val termFilter:TermFilter) {
  def buildDependencies(doc:edu.arizona.sista.processors.Document, start:Int, end:Int):Iterable[Dependency] = {
    val deps = new ListBuffer[Dependency]
    var i = start
    while(i < end) {
      doc.sentences(i).dependencies.foreach(ds => {
        addDependencies(ds,
          doc.sentences(i).words,
          doc.sentences(i).lemmas.getOrElse(doc.sentences(i).words),
          deps)
      })
      i += 1
    }
    deps.toList
  }

  private def addDependencies(
    deps:DirectedGraph[String],
    words:Array[String],
    lemmas:Array[String],
    result:ListBuffer[Dependency]) {
    val it = new DirectedGraphEdgeIterator[String](deps)
    while(it.hasNext) {
      val (h,m,l) = it.next()
      if (termFilter.validToken(words(h), lemmas(h)) &&
          termFilter.validToken(words(m), lemmas(m))) {
        result += new Dependency(lemmas(h), lemmas(m), l)
        result += new Dependency(lemmas(h), WILDCARD, l)
        result += new Dependency(WILDCARD, lemmas(m), l)
        result += new Dependency(lemmas(h), lemmas(m), WILDCARD)
      }
    }
  }

  def buildDependencies(doc:edu.arizona.sista.processors.Document):Iterable[Dependency] =
    buildDependencies(doc, 0, doc.sentences.length)
}

object DependencyBuilder {
  // better not use a Lucene special char here: + - && || ! ( ) { } [ ] ^ " ~ * ? : \
  val WILDCARD = "w"
}

class Dependency (
  val head:String,
  val modifier:String,
  val label:String) {

  override def toString:String = {
    val os = new StringBuilder
    os.append(head)
    os.append(">")
    os.append(label)
    os.append(">")
    os.append(modifier)
    os.toString()
  }
}
