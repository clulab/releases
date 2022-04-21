package org.clulab.odinsynth

import ai.lum.common.itertools.product
import ai.lum.odinson.utils.QueryUtils._
import ai.lum.odinson._
import AstNode._
import org.clulab.odinsynth.scorer.AstCost



// everything will be an AstNode
sealed trait AstNode {

  /** returns a string representation of the pattern */
  def pattern: String

  /** returns the number of holes in the tree */
  def numHoles: Int

  /** returns the number of nodes in the tree */
  def numNodes: Int

  /** distance between the root and the deepest leaf */
  def height: Int

  /** returns true if this node is a hole */
  def isHole: Boolean = false

  /** returns true if this node is a hole,
   *  or if any of its descendents is a hole
   */
  def hasHole: Boolean

  /** the states that can be reached from the current state
   *  by following valid transitions that expand the leftmost hole
   */
  def nextNodes(vocabularies: Map[String, Array[String]]): Array[AstNode]

  /** returns true if the pattern is a valid odinson query */
  def isValidQuery: Boolean = false

  /** returns a string representation that is easier to read */
  def pretty: String = pprint.apply(this, width = 72).toString

  /** return (an estimate of) the cost of the node + children nodes on the tree */
  def cost: Float

  /** return the cost of the node + children nodes on the tree using the cost define by @see AstCost */
  def cost(astCost: AstCost): Float

  /** returns the over-approximation of this query */
  def overApproximation: AstNode

  def unroll: AstNode

  def split: Seq[AstNode]

  def preOrderTraversal: Array[AstNode] = Array(this)

  def toBinary: AstNode = this
  def toFlat: AstNode = this

  var isBinary: Boolean = false

  def execute(ee: ExtractorEngine, disableMatchSelector: Boolean = false): Set[Spec] = {
    executeQuery(this, ee, disableMatchSelector)
  }

  def checkOverApproximation(ee: ExtractorEngine, specs: Set[Spec]): Boolean = {
    // generate query over-approximation and retrieve all possible matches
    val results = executeQuery(overApproximation, ee, disableMatchSelector = true)
    // all specs should be included in the possible results
    // or else this subtree is dead
    (specs -- results).isEmpty
  }

  def checkUnderApproximation(ee: ExtractorEngine, specs: Set[Spec]): Boolean = {
    val q = this match {
      case q: Query if q.isValidQuery => Some(q)
      case c: TokenConstraint if c.numHoles == 0 => Some(TokenQuery(c))
      case _ => None
    }
    if (q.isDefined) {
      val results = executeQuery(q.get, ee, disableMatchSelector = false).map(_.copy(captures = Set.empty))
      (results -- specs).isEmpty
    } else {
      true
    }
  }

  def hasRedundancy(ee: ExtractorEngine, specs: Set[Spec]): Boolean = {
    nodeHasRedundancy(this, ee, specs)
  }

}

object AstNode {

  val emptyNodeArray: Array[AstNode] = new Array[AstNode](0)

  def addHoleQuery(nodes: Vector[Query], i: Int): Vector[Query] = {
    val (lhs, rhs) = nodes.splitAt(i)
    lhs ++ Vector(HoleQuery) ++ rhs
  }

  def addHoleConstraint(nodes: Vector[TokenConstraint], i: Int): Vector[TokenConstraint] = {
    val (lhs, rhs) = nodes.splitAt(i)
    lhs ++ Vector(HoleConstraint) ++ rhs
  }

  def fillHole(nodes: Vector[Query], i: Int, node: Query): Vector[Query] = {
    nodes.updated(i, node)
  }

  def fillHole(nodes: Vector[TokenConstraint], i: Int, node: TokenConstraint): Vector[TokenConstraint] = {
    nodes.updated(i, node)
  }

  def subsumedClause(queries: Vector[Query], ee: ExtractorEngine): Boolean = {
    val results = queries.map(_.execute(ee, disableMatchSelector = false))
    var i = 0
    var j = 0
    while (i < results.length) {
      j = i + 1
      while (j < results.length) {
        if ((results(i) union results(j)).size == math.max(results(i).size, results(j).size)) {
          return true
        }
        j += 1
      }
      i += 1
    }
    false
  }

  def executeQuery(q: AstNode, ee: ExtractorEngine, disableMatchSelector: Boolean): Set[Spec] = {
    if (!q.isValidQuery) return Set.empty
    // make a rule object
    val rule = Rule(name = "Searcher_executeQuery", label = None, ruletype = "basic", pattern = q.pattern, priority = "1")
    // convert rule object into an extractor
    val extractors = ee.ruleReader.mkExtractors(Seq(rule))
    // use extractor to find matches
    val mentions = ee.extractMentions(extractors, disableMatchSelector = disableMatchSelector).toSeq
    // convert mentions to Spec objects
    Spec.fromOdinsonMentions(mentions)
  }

  def nodeHasRedundancy(node: AstNode, ee: ExtractorEngine, specs: Set[Spec]): Boolean = {
    val q = node match {
      case q: Query => q
      case c: TokenConstraint => TokenQuery(c)
      case _ => ???
    }
    if (q.isValidQuery) {
      val results = executeQuery(q, ee, disableMatchSelector = false)
      results.forall { r =>
        specs.forall { s =>
          r.docId != s.docId || r.sentId != s.sentId || !(r.interval subset s.interval)
        }
      }
    } else {
      false
    }
  }

  def splitQueries(node: AstNode): Seq[Query] = node match {
    case h: Hole => Nil
    case c: OrConstraint => c.constraints.map(TokenQuery)
    case c: TokenConstraint => Seq(TokenQuery(c))
    case q: TokenQuery => splitQueries(disjunctiveNormalForm(q.constraint))
    case q: OrQuery => q.queries.flatMap(splitQueries)
    case q: ConcatQuery =>
      val queries = q.queries.map(splitQueries).takeWhile(_.nonEmpty)
      if (queries.isEmpty) Nil
      else product(queries).map(qs => ConcatQuery(qs.toVector)).filter(_.isValidQuery)
    case q: RepeatQuery => splitQueries(q.query).map(x => RepeatQuery(x, q.min, q.max))
    case q: NamedCaptureQuery => splitQueries(q.query).map(x => NamedCaptureQuery(x, q.argName))
  }

  def disjunctiveNormalForm(c: TokenConstraint): TokenConstraint = {
    def convert(c: TokenConstraint): TokenConstraint = c match {
      case AndConstraint(Vector(lhs, rhs:OrConstraint)) =>
        val a = convert(lhs)
        val b = convert(rhs.constraints(0))
        val c = convert(rhs.constraints(1))
        OrConstraint(Vector(
          AndConstraint(Vector(a, b)),
          AndConstraint(Vector(a, c))))
      case AndConstraint(Vector(lhs:OrConstraint, rhs)) =>
        val a = convert(lhs.constraints(0))
        val b = convert(lhs.constraints(1))
        val c = convert(rhs)
        OrConstraint(Vector(
          AndConstraint(Vector(a, c)),
          AndConstraint(Vector(b, c))))
      case AndConstraint(cs) =>
        AndConstraint(cs.map(convert))
      case OrConstraint(cs) =>
        OrConstraint(cs.map(convert))
      case NotConstraint(c) =>
        NotConstraint(convert(c))
      case c => c
    }
    convert(negationNormalForm(c.toBinary)).toFlat
  }

  def negationNormalForm(c: TokenConstraint): TokenConstraint = c match {
    case OrConstraint(cs) =>
      OrConstraint(cs.map(negationNormalForm))
    case AndConstraint(cs) =>
      AndConstraint(cs.map(negationNormalForm))
    case NotConstraint(OrConstraint(cs)) => // De Morgan
      AndConstraint(cs.map(c => negationNormalForm(NotConstraint(c))))
    case NotConstraint(AndConstraint(cs)) => // De Morgan
      OrConstraint(cs.map(c => negationNormalForm(NotConstraint(c))))
    case NotConstraint(NotConstraint(c)) =>
      negationNormalForm(c)
    case c => c
  }

}



object Costs {
  val holeMatcher = 2
  val holeConstraint = 3
  val holeQuery = 2
  val stringMatcher = 1
  val fieldConstraint = 0
  val notConstraint = 10
  val andConstraint = 4
  val orConstraint = 4
  val tokenQuery = 0
  val concatQuery = 4
  val orQuery = 4
  val repeatQuery = 15
}



// a Hole is a placeholder that must be replaced with a valid AstNode
sealed trait Hole extends AstNode {
  val pattern: String = holeGlyph
  val numHoles: Int = 1
  val numNodes: Int = 1
  val height: Int = 1
  override val isHole: Boolean = true
  val hasHole: Boolean = true
}



// A Matcher is used to match a string to some value specified ahead of time
// (during the Matcher construction).
sealed trait Matcher extends AstNode {
  def overApproximation: Matcher
  def unroll: Matcher = this
  def split: Seq[Matcher] = Seq(this)
}

case object MatchAllMatcher extends Matcher {
  val pattern: String = "/.*/"
  val numHoles: Int = 0
  val numNodes: Int = 1
  val height: Int = 1
  val hasHole: Boolean = false
  def cost: Float = 0
  def cost(cost: AstCost): Float = cost.matchAllMatcher
  val overApproximation: Matcher = this
  def nextNodes(vv: Map[String, Array[String]]): Array[AstNode] = emptyNodeArray
}

case object HoleMatcher extends Matcher with Hole {
  val cost: Float = Costs.holeMatcher
  def cost(cost: AstCost): Float = cost.holeMatcher
  val overApproximation: Matcher = MatchAllMatcher
  def nextNodes(vocabularies: Map[String, Array[String]]): Array[AstNode] = {
    emptyNodeArray
  }
}

case class StringMatcher(s: String) extends Matcher {
  def pattern: String = maybeQuoteWord(s)
  val numHoles: Int = 0
  val numNodes: Int = 1
  val height: Int = 1
  val hasHole: Boolean = false
  val cost: Float = Costs.stringMatcher
  def cost(cost: AstCost): Float = cost.stringMatcher
  val overApproximation: Matcher = this
  def nextNodes(vocabularies: Map[String, Array[String]]): Array[AstNode] = {
    emptyNodeArray
  }
}



// A TokenConstraint is used to match a single token.
sealed trait TokenConstraint extends AstNode {
  def overApproximation: TokenConstraint
  def unroll: TokenConstraint = this
  def split: Seq[TokenConstraint] = Seq(this)
  override def toBinary: TokenConstraint = this
  override def toFlat: TokenConstraint = this
  def getValidQuery: Option[TokenConstraint] = Some(this)
}

case object MatchAllConstraint extends TokenConstraint {
  val pattern: String = "" // will be surrounded with brackets by TokenQuery
  val numHoles: Int = 0
  val numNodes: Int = 1
  val height: Int = 1
  val hasHole: Boolean = false
  def cost: Float = 0
  def cost(cost: AstCost): Float = cost.matchAllConstraint
  val overApproximation: TokenConstraint = this
  def nextNodes(vv: Map[String, Array[String]]): Array[AstNode] = emptyNodeArray
}

case object HoleConstraint extends TokenConstraint with Hole {
  def cost: Float = Costs.holeConstraint
  def cost(cost: AstCost): Float = cost.holeConstraint
  val overApproximation: TokenConstraint = MatchAllConstraint
  def nextNodes(vocabularies: Map[String, Array[String]]): Array[AstNode] = {
    Array[AstNode](
      FieldConstraint(HoleMatcher, HoleMatcher),
      NotConstraint(HoleConstraint),
      AndConstraint(Vector(HoleConstraint, HoleConstraint)),
      OrConstraint(Vector(HoleConstraint, HoleConstraint)),
    )
  }
  override def getValidQuery: Option[TokenConstraint] = None
}

case class FieldConstraint(name: Matcher, value: Matcher) extends TokenConstraint {

  def pattern: String = s"${name.pattern}=${value.pattern}"
  def numHoles: Int = name.numHoles + value.numHoles
  def numNodes: Int = 1 + name.numNodes + value.numNodes
  def height: Int = 1 + math.max(name.height, value.height)
  def hasHole: Boolean = name.hasHole || value.hasHole
  val cost: Float = Costs.fieldConstraint + name.cost + value.cost
  def cost(cost: AstCost): Float = cost.fieldConstraint + name.cost(cost) + value.cost(cost)

  override def preOrderTraversal: Array[AstNode] = {
    super.preOrderTraversal ++ name.preOrderTraversal ++ value.preOrderTraversal
  }

  def overApproximation: TokenConstraint = {
    if (name.isHole || value.isHole) MatchAllConstraint else this
  }

  def nextNodes(vocabularies: Map[String, Array[String]]): Array[AstNode] = {
    val fieldNames = vocabularies.keys.toArray
    if (name.isHole) {
      // If the name is a hole then we need to fill it first.
      // Make an array with one element for each fieldname.
      val results = new Array[AstNode](fieldNames.length)
      var i = 0
      while (i < results.length) {
        // make a copy of this node replacing the hole in name
        // with a fieldname
        results(i) = copy(name = StringMatcher(fieldNames(i)))
        i += 1
      }
      // return results
      results
    } else if (value.isHole) {
      // If the name is set and the value is a hole
      // then fill the hole with the terms associated
      // with the fieldname.
      // First retrieve the vocabulary.
      val terms = vocabularies(name.pattern)
      // make an array with one element for each term
      // in the vocabulary
      val results = new Array[AstNode](terms.length)
      var i = 0
      while (i < results.length) {
        // make a copy of this node replacing the hole in value
        // with a term in the vocabulary
        results(i) = copy(value = StringMatcher(terms(i)))
        i += 1
      }
      // return results
      results
    } else {
      // there are no next nodes
      emptyNodeArray
    }
  }

  override def getValidQuery(): Option[TokenConstraint] = {
    if (name.isHole || value.isHole) None else Some(this)
  }
}

case class NotConstraint(constraint: TokenConstraint) extends TokenConstraint {

  def pattern: String = constraint match {
    case c: OrConstraint  => s"!(${c.pattern})"
    case c: AndConstraint => s"!(${c.pattern})"
    case c => s"!${c.pattern}"
  }

  def numHoles: Int = constraint.numHoles
  def numNodes: Int = 1 + constraint.numNodes
  def height: Int = 1 + constraint.height
  def hasHole: Boolean = constraint.hasHole
  val cost: Float = Costs.notConstraint + constraint.cost
  def cost(cost: AstCost): Float = cost.notConstraint + constraint.cost(cost)

  override def preOrderTraversal: Array[AstNode] = {
    super.preOrderTraversal ++ constraint.preOrderTraversal
  }

  def overApproximation: TokenConstraint = {
    val over = constraint.overApproximation
    if (over == MatchAllConstraint) MatchAllConstraint else NotConstraint(over)
  }

  def nextNodes(vocabularies: Map[String, Array[String]]): Array[AstNode] = {
    if (!constraint.hasHole) return emptyNodeArray
    // avoid nesting negations
    val nodes = constraint.nextNodes(vocabularies).filterNot(_.isInstanceOf[NotConstraint])
    val results = new Array[AstNode](nodes.length)
    var i = 0
    while (i < results.length) {
      results(i) = nodes(i) match {
        case c: TokenConstraint => copy(constraint = c)
        case _ => ???
      }
      i += 1
    }
    results
  }

  override def getValidQuery(): Option[TokenConstraint] = {
    if( constraint.getValidQuery.isDefined ) {
      Some(this)
    } else {
      None
    }
  }
}

case class AndConstraint(constraints: Vector[TokenConstraint]) extends TokenConstraint {

  def pattern: String = {
    constraints.map{
      case c: OrConstraint => s"(${c.pattern})"
      case c => c.pattern
    }.mkString(" & ")
  }

  def numHoles: Int = constraints.map(_.numHoles).sum
  def numNodes: Int = 1 + constraints.map(_.numNodes).sum
  def height: Int = 1 + constraints.map(_.height).max
  def hasHole: Boolean = constraints.exists(_.hasHole)
  val cost: Float = Costs.andConstraint + constraints.map(_.cost).sum
  def cost(cost: AstCost): Float = cost.andConstraint + constraints.map(it => it.cost(cost)).sum

  override def toBinary: TokenConstraint = {
    constraints.map(_.toBinary).reduceRight[TokenConstraint] { case (a, b) =>
      val c = AndConstraint(Vector(a, b))
      c.isBinary = true
      c
    }
  }

  override def toFlat: TokenConstraint = {
    val cs = constraints.map(_.toFlat).flatMap {
      case c: AndConstraint => c.constraints.map(_.toFlat)
      case c => Array(c)
    }
    AndConstraint(cs)
  }

  override def preOrderTraversal: Array[AstNode] = {
    super.preOrderTraversal ++ constraints.flatMap(_.preOrderTraversal)
  }

  def overApproximation: TokenConstraint = {
    val over = constraints
      .map(_.overApproximation)
      .filter(_ != MatchAllConstraint)
    if (over.isEmpty) MatchAllConstraint else AndConstraint(over)
  }

  override def hasRedundancy(ee: ExtractorEngine, specs: Set[Spec]): Boolean = {
    if (super.hasRedundancy(ee, specs)) return true
    if (constraints.size > 3) return true // FIXME don't hardcode values
    val qs = constraints.filter(_.numHoles == 0).map(TokenQuery)
    qs.length != qs.distinct.length || subsumedClause(qs, ee) || constraints.exists(_.hasRedundancy(ee, specs))
  }

  def nextNodes(vocabularies: Map[String, Array[String]]): Array[AstNode] = {
    // find the leftmost clause with a hole
    val i = constraints.indexWhere(_.hasHole)
    if (i < 0) return emptyNodeArray
    val constraint = constraints(i)
    val nodes = constraint.nextNodes(vocabularies)
    val results = new Array[AstNode](nodes.length)
    var j = 0
    while (j < results.length) {
      // make new constraints
      val newConstraints = nodes(j) match {
        case c: AndConstraint if !isBinary => addHoleConstraint(constraints, i)
        case c: TokenConstraint => fillHole(constraints, i, c)
        case _ => ???
      }
      results(j) = copy(constraints = newConstraints)
      j += 1
    }
    results
  }
  
  override def getValidQuery(): Option[TokenConstraint] = {
    val nonHoleConstraints = constraints.flatMap(_.getValidQuery)
    if(nonHoleConstraints.size > 1) {
      Some(AndConstraint(nonHoleConstraints))
    } else if(nonHoleConstraints.size > 0) {
      Some(nonHoleConstraints.head)
    } else {
      None
    }
  }

}

case class OrConstraint(constraints: Vector[TokenConstraint]) extends TokenConstraint {

  def pattern: String = constraints.map(_.pattern).mkString(" | ")
  def numHoles: Int = constraints.map(_.numHoles).sum
  def numNodes: Int = 1 + constraints.map(_.numNodes).sum
  def height: Int = 1 + constraints.map(_.height).max
  def hasHole: Boolean = constraints.exists(_.hasHole)
  val cost: Float = Costs.orConstraint + constraints.map(_.cost).sum
  def cost(cost: AstCost): Float = cost.orConstraint + constraints.map(it => it.cost(cost)).sum

  override def toBinary: TokenConstraint = {
    constraints.map(_.toBinary).reduceRight[TokenConstraint] { case (a, b) =>
      val c = OrConstraint(Vector(a, b))
      c.isBinary = true
      c
    }
  }

  override def toFlat: TokenConstraint = {
    val cs = constraints.map(_.toFlat).flatMap {
      case c: OrConstraint => c.constraints.map(_.toFlat)
      case c => Array(c)
    }
    OrConstraint(cs)
  }

  override def preOrderTraversal: Array[AstNode] = {
    super.preOrderTraversal ++ constraints.flatMap(_.preOrderTraversal)
  }

  def overApproximation: TokenConstraint = {
    val overConstraints = constraints.map(_.overApproximation)
    if (overConstraints contains MatchAllConstraint) {
      MatchAllConstraint
    } else {
      OrConstraint(overConstraints)
    }
  }

  override def split: Seq[TokenConstraint] = {
    Seq(this) ++ constraints.flatMap(_.split)
  }

  override def hasRedundancy(ee: ExtractorEngine, specs: Set[Spec]): Boolean = {
    if (super.hasRedundancy(ee, specs)) return true
    if (constraints.size > 3) return true // FIXME don't hardcode values
    val qs = constraints.filter(_.numHoles == 0).map(TokenQuery)
    qs.length != qs.distinct.length || subsumedClause(qs, ee) || constraints.exists(_.hasRedundancy(ee, specs))
  }

  def nextNodes(vocabularies: Map[String, Array[String]]): Array[AstNode] = {
    // find the leftmost clause with a hole
    val i = constraints.indexWhere(_.hasHole)
    if (i < 0) return emptyNodeArray
    val constraint = constraints(i)
    val nodes = constraint.nextNodes(vocabularies)
    val results = new Array[AstNode](nodes.length)
    var j = 0
    while (j < results.length) {
      val newConstraints = nodes(j) match {
        case c: OrConstraint if !isBinary => addHoleConstraint(constraints, i)
        case c: TokenConstraint => fillHole(constraints, i, c)
        case _ => ???
      }
      results(j) = copy(constraints = newConstraints)
      j += 1
    }
    results
  }

  override def getValidQuery(): Option[TokenConstraint] = {
    val nonHoleConstraints = constraints.flatMap(_.getValidQuery)
    if(nonHoleConstraints.size > 1) {
      Some(OrConstraint(nonHoleConstraints))
    } else if(nonHoleConstraints.size > 0) {
      Some(nonHoleConstraints.head)
    } else {
      None
    }
  }

}



// A Query represents a valid query.
sealed trait Query extends AstNode {
  def overApproximation: Query
  def unroll: Query = this
  def split: Seq[Query] = Seq(this)
  override def isValidQuery: Boolean = !hasHole
  override def toBinary: Query = this
  override def toFlat: Query = this

  def getValidQuery: Option[Query] = Some(this)
}

case object MatchAllQuery extends Query {
  val pattern: String = "[]*"
  val numHoles: Int = 0
  val numNodes: Int = 1
  val height: Int = 1
  val hasHole: Boolean = false
  def cost: Float = 0
  def cost(cost: AstCost): Float = cost.matchAllQuery
  val overApproximation: Query = this
  def nextNodes(vv: Map[String, Array[String]]): Array[AstNode] = emptyNodeArray
}

case object HoleQuery extends Query with Hole {
  def cost: Float = Costs.holeQuery
  def cost(cost: AstCost): Float = cost.holeQuery
  val overApproximation: Query = MatchAllQuery
  def nextNodes(vocabularies: Map[String, Array[String]]): Array[AstNode] = {
    Array[AstNode](
      TokenQuery(HoleConstraint),
      ConcatQuery(Vector(HoleQuery, HoleQuery)),
      OrQuery(Vector(HoleQuery, HoleQuery)),
      RepeatQuery(HoleQuery, 0, 1),
      RepeatQuery(HoleQuery, 0, None),
      RepeatQuery(HoleQuery, 1, None),
    )
  }

  override def getValidQuery: Option[Query] = None
}

case class TokenQuery(constraint: TokenConstraint) extends Query {

  def pattern: String = s"[${constraint.pattern}]"
  def numHoles: Int = constraint.numHoles
  def numNodes: Int = 1 + constraint.numNodes
  def height: Int = 1 + constraint.height
  def hasHole: Boolean = constraint.hasHole
  val cost: Float = Costs.tokenQuery + constraint.cost
  def cost(cost: AstCost): Float = cost.tokenQuery + constraint.cost(cost)

  override def toBinary: Query = {
    val tq = TokenQuery(constraint.toBinary)
    tq.isBinary = true
    tq
  }

  override def preOrderTraversal: Array[AstNode] = {
    super.preOrderTraversal ++ constraint.preOrderTraversal
  }

  def overApproximation: Query = {
    TokenQuery(constraint.overApproximation)
  }

  override def split: Seq[Query] = {
    Seq(this) ++ constraint.split.map(TokenQuery)
  }

  override def hasRedundancy(ee: ExtractorEngine, specs: Set[Spec]): Boolean = {
    disjunctiveNormalForm(constraint).hasRedundancy(ee, specs)
  }

  def nextNodes(vocabularies: Map[String, Array[String]]): Array[AstNode] = {
    if (!constraint.hasHole) return emptyNodeArray
    val nodes = constraint.nextNodes(vocabularies)
    val results = new Array[AstNode](nodes.length)
    var i = 0
    while (i < results.length) {
      results(i) = nodes(i) match {
        case c: TokenConstraint => copy(constraint = c)
        case _ => ???
      }
      i += 1
    }
    results
  }

  override def getValidQuery(): Option[Query] = {
    constraint match {
      case HoleConstraint => None
      case c: OrConstraint =>
        if(c.getValidQuery.isDefined) {
          Option( TokenQuery(c.getValidQuery.get))
        } else {
          None
        }
      case c: AndConstraint => {
        if(c.getValidQuery.isDefined) {
          Option( TokenQuery(c.getValidQuery.get))
        } else {
          None
        }
      }
      case c: FieldConstraint => 
        if(c.getValidQuery.isDefined) {
          Option( TokenQuery(c.getValidQuery.get))
        } else {
          None
        }
      case c: NotConstraint => 
        if(c.getValidQuery.isDefined) {
          Option( TokenQuery(c.getValidQuery.get))
        } else {
          None
        }
      case _ => ???
    }
  }

}

case class ConcatQuery(queries: Vector[Query]) extends Query {

  def pattern: String = {
    queries.map{
      case q: OrQuery => s"(${q.pattern})"
      case q => q.pattern
    }.mkString(" ")
  }

  def numHoles: Int = queries.map(_.numHoles).sum
  def numNodes: Int = 1 + queries.map(_.numNodes).sum
  def height: Int = 1 + queries.map(_.height).max
  def hasHole: Boolean = queries.exists(_.hasHole)
  val cost: Float = Costs.concatQuery + queries.map(_.cost).sum
  def cost(cost: AstCost): Float = cost.concatQuery + queries.map(it => it.cost(cost)).sum

  override def toBinary: Query = {
    queries.map(_.toBinary).reduceRight[Query] { case (a, b) =>
      val q = ConcatQuery(Vector(a, b))
      q.isBinary = true
      q
    }
  }

  override def toFlat: Query = {
    val qs = queries.map(_.toFlat).flatMap {
      case q: ConcatQuery => q.queries.map(_.toFlat)
      case q => Array(q)
    }
    ConcatQuery(qs)
  }

  override def preOrderTraversal: Array[AstNode] = {
    super.preOrderTraversal ++ queries.flatMap(_.preOrderTraversal)
  }

  def overApproximation: Query = {
    var over = queries.map(_.overApproximation)
    // collapse consecutive MatchAllQuery into a single one
    over = over.head +: over.sliding(2)
      .collect { case Vector(a,b) if a != b || b != MatchAllQuery => b }
      .toVector
    ConcatQuery(over)
  }

  override def unroll: Query = {
    ConcatQuery(queries.map(_.unroll))
  }

  override def split: Seq[Query] = {
    product(queries.map(_.split)).map(qs => ConcatQuery(qs.toVector))
  }

  override def hasRedundancy(ee: ExtractorEngine, specs: Set[Spec]): Boolean = {
    if (super.hasRedundancy(ee, specs)) return true
    if (queries.count(_.isHole) > 3) return true
    splitQueries(this).exists(q => nodeHasRedundancy(q, ee, specs)) || queries.exists(_.hasRedundancy(ee, specs))
  }

  def nextNodes(vocabularies: Map[String, Array[String]]): Array[AstNode] = {
    val i = queries.indexWhere(_.hasHole)
    if (i < 0) return emptyNodeArray
    val query = queries(i)
    val nodes = query.nextNodes(vocabularies)
    val results = new Array[AstNode](nodes.length)
    var j = 0
    while (j < results.length) {
      val newQueries = nodes(j) match {
        case q: ConcatQuery if !isBinary => addHoleQuery(queries, i)
        case q: Query => fillHole(queries, i, q)
        case _ => ???
      }
      results(j) = copy(queries = newQueries)
      j += 1
    }
    results
  }

  override def getValidQuery(): Option[Query] = {
    val nonHoleQueries = queries.flatMap(_.getValidQuery)
    if(nonHoleQueries.size > 1) {
      Some(ConcatQuery(nonHoleQueries))
    }
    else if(nonHoleQueries.size > 0) {
      // there is only one element
      Some(nonHoleQueries.head)
    } else {
      None
    }
  }

}

case class OrQuery(queries: Vector[Query]) extends Query {

  def pattern: String = queries.map(_.pattern).mkString(" | ")
  def numHoles: Int = queries.map(_.numHoles).sum
  def numNodes: Int = 1 + queries.map(_.numNodes).sum
  def height: Int = 1 + queries.map(_.height).max
  def hasHole: Boolean = queries.exists(_.hasHole)
  def cost: Float = Costs.orQuery + queries.map(_.cost).sum
  def cost(cost: AstCost): Float = cost.orQuery + queries.map(it => it.cost(cost)).sum

  override def toBinary: Query = {
    queries.map(_.toBinary).reduceRight[Query] { case (a, b) =>
      val q = OrQuery(Vector(a, b))
      q.isBinary = true
      q
    }
  }

  override def toFlat: Query = {
    val qs = queries.map(_.toFlat).flatMap {
      case q: OrQuery => q.queries.map(_.toFlat)
      case q => Array(q)
    }
    OrQuery(qs)
  }

  override def preOrderTraversal: Array[AstNode] = {
    super.preOrderTraversal ++ queries.flatMap(_.preOrderTraversal)
  }

  def overApproximation: Query = {
    val overQueries = queries.map(_.overApproximation)
    if (overQueries contains MatchAllQuery) {
      MatchAllQuery
    } else {
      OrQuery(overQueries)
    }
  }

  override def checkUnderApproximation(ee: ExtractorEngine, specs: Set[Spec]): Boolean = {
    queries.forall(_.checkUnderApproximation(ee, specs))
  }

  override def unroll: Query = {
    OrQuery(queries.map(_.unroll))
  }

  override def split: Seq[Query] = {
    Seq(this) ++ queries ++ queries.flatMap(_.split)
  }

  override def hasRedundancy(ee: ExtractorEngine, specs: Set[Spec]): Boolean = {
    if (super.hasRedundancy(ee, specs)) return true
    queries.length != queries.distinct.length || subsumedClause(queries, ee) || queries.exists(_.hasRedundancy(ee, specs))
  }

  def nextNodes(vocabularies: Map[String, Array[String]]): Array[AstNode] = {
    val i = queries.indexWhere(_.hasHole)
    if (i < 0) return emptyNodeArray
    val query = queries(i)
    val nodes = query.nextNodes(vocabularies)
    val results = new Array[AstNode](nodes.length)
    var j = 0
    while (j < results.length) {
      val newQueries = nodes(j) match {
        case q: OrQuery if !isBinary => addHoleQuery(queries, i)
        case q: Query => fillHole(queries, i, q)
        case _ => ???
      }
      results(j) = copy(queries = newQueries)
      j += 1
    }
    results
  }

  override def getValidQuery(): Option[Query] = {
    val nonHoleQueries = queries.flatMap(_.getValidQuery)
    if(nonHoleQueries.size > 1) {
      Some(OrQuery(nonHoleQueries)) 
    } else if(nonHoleQueries.size > 0) {
      Some(nonHoleQueries.head)
    } else {
      None
    }
  }
}

case class RepeatQuery(query: Query, min: Int, max: Option[Int]) extends Query {

  require(
    max.isEmpty || min <= max.get,
    s"min=$min must be less than or equal to max=${max.get}"
  )

  def pattern: String = query match {
    case q: ConcatQuery => s"(${q.pattern})${quantifier(min, max)}"
    case q: OrQuery => s"(${q.pattern})${quantifier(min, max)}"
    case q => s"${q.pattern}${quantifier(min, max)}"
  }
  def numHoles: Int = query.numHoles
  def numNodes: Int = 1 + query.numNodes
  def height: Int = 1 + query.height
  def hasHole: Boolean = query.hasHole
  val cost: Float = Costs.repeatQuery + query.cost
  def cost(cost: AstCost): Float = cost.repeatQuery + query.cost(cost)

  override def toBinary: Query = {
    val rq = RepeatQuery(query.toBinary, min, max)
    rq.isBinary = true
    rq
  }

  override def preOrderTraversal: Array[AstNode] = {
    super.preOrderTraversal ++ query.preOrderTraversal
  }

  def overApproximation: Query = {
    val over = query.overApproximation
    if (over == MatchAllQuery) over else RepeatQuery(over, min, max)
  }

  override def unroll: Query = {
    if (hasHole) return this
    val q = query.unroll
    if (min < 2 && max.isEmpty) {
      ConcatQuery(Vector(q, q, RepeatQuery(q, min, max)))
    } else {
      RepeatQuery(q, min, max)
    }
  }

  override def split: Seq[Query] = {
    this +: query.split
  }

  override def hasRedundancy(ee: ExtractorEngine, specs: Set[Spec]): Boolean = {
    super.hasRedundancy(ee, specs) || query.hasRedundancy(ee, specs)
  }

  def nextNodes(vocabularies: Map[String, Array[String]]): Array[AstNode] = {
    if (!query.hasHole) return emptyNodeArray
    // avoid nesting repetitions directly
    val nodes = query.nextNodes(vocabularies).filterNot(_.isInstanceOf[RepeatQuery])
    val results = new Array[AstNode](nodes.length)
    var i = 0
    while (i < results.length) {
      results(i) = nodes(i) match {
        case q: Query => copy(query = q)
        case _ => ???
      }
      i += 1
    }
    results
  }

  override def getValidQuery(): Option[Query] = {
    query.getValidQuery.map { q => if (q.isInstanceOf[RepeatQuery]) q else RepeatQuery(q, min, max) }
  }

}

object RepeatQuery {

  def apply(query: Query, n: Int): RepeatQuery = {
    RepeatQuery(query, n, Some(n))
  }

  def apply(query: Query, min: Int, max: Int): RepeatQuery = {
    RepeatQuery(query, min, Some(max))
  }

}


case class NamedCaptureQuery(query: Query, argName: String = "arg") extends Query {
  def pattern: String = f"(?<$argName> (${query.pattern}))"

  def cost: Float = 0f
  def cost(astCost: AstCost): Float = 0f

  def hasHole: Boolean = query.hasHole
  def height: Int = 1 + query.height

  def nextNodes(vocabularies: Map[String,Array[String]]): Array[AstNode] = ??? //Array.empty

  def numHoles: Int = query.numHoles
  def numNodes: Int = 1 + query.numNodes

  def overApproximation: Query = NamedCaptureQuery(query.overApproximation, argName) // query.overApproximation

  override def toBinary: Query = {
    val ncq = NamedCaptureQuery(query.toBinary, argName)
    ncq.isBinary = true
    ncq
  }

  override def toFlat: Query = NamedCaptureQuery(query.toFlat, argName)

  override def unroll: Query = {
    NamedCaptureQuery(query.unroll, argName)
  }

}


object TestTest extends App {
  // val q = TokenQuery(HoleConstraint)
  // val q = OrQuery(Vector(HoleQuery, HoleQuery))
  val q = ConcatQuery(Vector(HoleQuery, HoleQuery))
  println(q.pattern)
  q.nextNodes(
    Map(
      "tag" -> Array("NN", "VBD"),
      "word" -> Array("car", "is"),
    )
  )
  .map(_.pattern)
  .foreach(println)
}