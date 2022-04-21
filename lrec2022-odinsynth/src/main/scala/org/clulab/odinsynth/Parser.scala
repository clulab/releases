package org.clulab.odinsynth

import fastparse._
import ScriptWhitespace._
import ai.lum.odinson.compiler.Literals

object Parser {

  case class Quantifier(min: Int, max: Option[Int])

  def parseBasicQuery(pattern: String) = parse(pattern, basicQuery(_)).get.value

  def basicQuery[_: P]: P[Query] = {
    P(Start ~ surfaceQuery ~ End)
  }

  def surfaceQuery[_: P]: P[Query] = {
    P(orQuery)
  }

  def orQuery[_: P]: P[Query] = {
    P(concatQuery.rep(min = 1, sep = "|")).map {
      case Seq(query) => query
      case queries    => OrQuery(queries.toVector)
    }
  }

  def concatQuery[_: P]: P[Query] = {
    P(repeatQuery.rep(1)).map {
      case Seq(query) => query
      case queries    => ConcatQuery(queries.toVector)
    }
  }

  def repeatQuery[_: P]: P[Query] = {
    P(atomicQuery ~ quantifier.?).map {
      case (query, None) => query
      case (query, Some(Quantifier(min, max))) => RepeatQuery(query, min, max)
    }
  }

  def namedCaptureQuery[_: P]: P[Query] = {
    P(
      "(?<" ~ Literals.string ~ ">" ~ atomicQuery ~ ")"
    ).map {
      case (name, pattern) =>
        NamedCaptureQuery(pattern, name)
    }
  }

  def atomicQuery[_: P]: P[Query] = {
    P(holeQuery | tokenQuery | "(" ~ orQuery ~ ")" | namedCaptureQuery)
  }

  def holeQuery[_: P]: P[Query] = {
    P(holeGlyph).map(_ => HoleQuery)
  }

  def tokenQuery[_: P]: P[Query] = {
    P(tokenConstraint).map(TokenQuery)
  }

  def tokenConstraint[_: P]: P[TokenConstraint] = {
    P("[" ~ orConstraint ~ "]")
  }

  def orConstraint[_: P]: P[TokenConstraint] = {
    P(andConstraint.rep(min = 1, sep = "|")).map {
      case Seq(constraint) => constraint
      case constraints     => OrConstraint(constraints.toVector)
    }
  }

  def andConstraint[_: P]: P[TokenConstraint] = {
    P(notConstraint.rep(min = 1, sep = "&")).map {
      case Seq(constraint) => constraint
      case constraints     => AndConstraint(constraints.toVector)
    }
  }

  def notConstraint[_: P]: P[TokenConstraint] = {
    P("!".!.? ~ atomicConstraint).map {
      case (None,    constraint) => constraint
      case (Some(_), constraint) => NotConstraint(constraint)
    }
  }

  def atomicConstraint[_: P]: P[TokenConstraint] = {
    P(fieldConstraint | holeConstraint | "(" ~ orConstraint ~ ")")
  }

  def holeConstraint[_: P]: P[TokenConstraint] = {
    P(holeGlyph).map(_ => HoleConstraint)
  }

  def fieldConstraint[_: P]: P[TokenConstraint] = {
    P(matcher ~ "=" ~ matcher).map {
      case (name, matcher) => FieldConstraint(name, matcher)
    }
  }

  def matcher[_: P]: P[Matcher] = {
    P(holeMatcher | stringMatcher)
  }

  def holeMatcher[_: P]: P[Matcher] = {
    P(holeGlyph).map(_ => HoleMatcher)
  }

  def stringMatcher[_: P]: P[Matcher] = {
    P(Literals.string).map(StringMatcher)
  }

  def quantifier[_: P]: P[Quantifier] = {
    P(quantOperator | range | repetition)
  }

  def quantOperator[_: P]: P[Quantifier] = {
    P(StringIn("?", "*", "+")).!.map {
      case "?" => Quantifier(0, Some(1))
      case "*" => Quantifier(0, None)
      case "+" => Quantifier(1, None)
    }
  }

  def range[_: P]: P[Quantifier] = {
    P("{" ~ Literals.unsignedInt.? ~ "," ~ Literals.unsignedInt.? ~ "}").flatMap {
      case (Some(min), Some(max)) if min > max => Fail
      case (Some(min), maxOption) => Pass(Quantifier(min, maxOption))
      case (None,      maxOption) => Pass(Quantifier(0,   maxOption))
    }
  }

  def repetition[_: P]: P[Quantifier] = {
    P("{" ~ Literals.unsignedInt ~ "}").map {
      case n => Quantifier(n, Some(n))
    }
  }

}
