package org.clulab.odinsynth

import ai.lum.odinson._

class Oracle(
  val _src: AstNode,
  val _dst: AstNode,
  val vocabularies: Map[String, Array[String]],
) {

  val src = _src.toBinary
  val dst = _dst.toBinary

  def this(dst: AstNode, vocabularies: Map[String, Array[String]]) = {
    this(HoleQuery, dst, vocabularies)
  }

  def this(dst: String, vocabularies: Map[String, Array[String]]) = {
    this(Parser.parseBasicQuery(dst), vocabularies)
  }

  def this(src: String, dst: String, vocabularies: Map[String, Array[String]]) = {
    this(Parser.parseBasicQuery(src), Parser.parseBasicQuery(dst), vocabularies)
  }

  def transitions = new Iterable[Transition] {
    def iterator = new Iterator[Transition] {
      var curr: AstNode = src
      var result: Option[AstNode] = None
      var index = -1
      def hasNext: Boolean = {
        if (result.isEmpty) {
          val resultIndex = pick(curr)
          result = resultIndex._1
          index  = resultIndex._2
        }
        result.isDefined
      }
      def next: Transition = result match {
        case None => null
        case Some(node) =>
          val start = curr
          curr = node
          result = None
          Transition(start, node.toBinary, index)
      }
    }
  }

  def pick(node: AstNode): (Option[AstNode], Int) = {
    // maybe we are already done?
    if (node != dst) {
      // get all candidates
      val candidates = node.nextNodes(vocabularies)
      // find the correct candidate
      for ((candidate, index) <- candidates.zipWithIndex) {
        val c = candidate.toBinary
        if (isNextState(c, node, dst)) {
          return (Some(c), index)
        }
      }
    }
    // how can this be?
    (None, -1)
  }

  private def isNextState(c: AstNode, curr: AstNode, dst: AstNode): Boolean = {
    val cNodes = c.preOrderTraversal
    val currNodes = curr.preOrderTraversal
    val dstNodes = dst.preOrderTraversal
    val i = currNodes.indexWhere(_.isHole)
    i >= 0 && confirmMatch(cNodes(i), dstNodes(i))
  }
  
  private def confirmMatch(c: AstNode, dst: AstNode): Boolean = {
    (c, dst) match {
      case ((c: StringMatcher), (dst:StringMatcher)) =>
        c == dst
      case ((c:RepeatQuery), (dst:RepeatQuery)) =>
        c.min == dst.min && c.max == dst.max
      case (c, dst) =>
        c.getClass == dst.getClass
    }
  }

}
