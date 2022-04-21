from collections import namedtuple
from pyparsing import *
from queryast import *
import config


inf = config.INFINITY
hole = config.HOLE_GLYPH


class QueryParser:

    def __init__(self):
        self.parser = make_parser()

    def parse(self, pattern: str) -> AstNode:
        return self.parser.parseString(pattern)[0]


def make_parser():

    # literal values
    number = Word(nums).setParseAction(lambda t: int(t[0]))
    identifier = Word(alphas + '_', alphanums + '_')
    quoted_string = QuotedString('"', unquoteResults=True, escChar='\\')
    string = identifier | quoted_string

    # number to the left of the comma {n,}
    quant_range_left = Literal('{').suppress() + number + Literal(',').suppress() + Literal('}').suppress()
    quant_range_left.setParseAction(lambda t: (t[0], inf))
    # number to the right of the comma {,m}
    quant_range_right = Literal('{').suppress() + Literal(',').suppress() + number + Literal('}').suppress()
    quant_range_right.setParseAction(lambda t: (0, t[0]))
    # numbers on both sides of the comma {n,m}
    quant_range_both = Literal('{').suppress() + number + Literal(',').suppress() + number + Literal('}').suppress()
    quant_range_both.setParseAction(lambda t: (t[0], t[1]))
    # no number either side of the comma {,}
    quant_range_neither = Literal('{').suppress() + Literal(',').suppress() + Literal('}').suppress()
    quant_range_neither.setParseAction(lambda t: (0, inf))
    # range {n,m}
    quant_range = quant_range_left | quant_range_right | quant_range_both | quant_range_neither
    # repetition {n}
    quant_rep = Literal('{').suppress() + number + Literal('}').suppress()
    quant_rep.setParseAction(lambda t: (t[0], t[0]))
    # quantifier operator
    quant_op = oneOf('? * +')
    quant_op.setParseAction(lambda t: (0, 1) if t[0] == '?' else (0, inf) if t[0] == '*' else (1, inf))
    # any quantifier
    quantifier = quant_range | quant_rep | quant_op

    # a hole that can take the place of a matcher
    hole_matcher = Literal(hole).setParseAction(lambda t: HoleMatcher())
    # a matcher that compares tokens to a string (t[0])
    string_matcher = string.setParseAction(lambda t: StringMatcher(t[0]))
    # any matcher
    matcher = hole_matcher | string_matcher

    # a hole that can take the place of a token constraint
    hole_constraint = Literal(hole).setParseAction(lambda t: HoleConstraint())

    # a constraint of the form `f=v` means that only tokens
    # that have a field `f` with a corresponding value of `v`
    # can be accepted
    field_constraint = matcher + Literal('=').suppress() + matcher
    field_constraint.setParseAction(lambda t: FieldConstraint(*t))

    # forward declaration, defined below
    or_constraint = Forward()

    # an expression that represents a single constraint
    atomic_constraint = field_constraint | hole_constraint | Literal('(').suppress() + or_constraint + Literal(')').suppress()

    # a constraint that may or may not be negated
    not_constraint = Optional('!') + atomic_constraint
    not_constraint.setParseAction(lambda t: NotConstraint(t[1]) if len(t) > 1 else t[0])

    # one or more constraints ANDed together
    and_constraint = not_constraint + ZeroOrMore(Literal('&').suppress() + not_constraint)
    and_constraint.setParseAction(lambda t: AndConstraint(t) if len(t) > 1 else t[0])

    # one or more constraints ORed together
    or_constraint << (and_constraint + ZeroOrMore(Literal('|').suppress() + and_constraint))
    or_constraint.setParseAction(lambda t: OrConstraint(t) if len(t) > 1 else t[0])

    # a hole that can take the place of a query
    hole_query = Literal(hole).setParseAction(lambda t: HoleQuery())

    # a token constraint surrounded by square brackets
    token_query = Literal('[').suppress() + or_constraint + Literal(']').suppress()
    token_query.setParseAction(lambda t: TokenQuery(t[0]))

    # forward declaration, defined below
    or_query = Forward()

    # an expression that represents a single query
    atomic_query = hole_query | token_query | Literal('(').suppress() + or_query + Literal(')').suppress()

    # a query with an optional quantifier
    repeat_query = atomic_query + Optional(quantifier)
    repeat_query.setParseAction(lambda t: RepeatQuery(t[0], *t[1]) if len(t) > 1 else t[0])

    # one or more queries that must match consecutively
    concat_query = OneOrMore(repeat_query)
    concat_query.setParseAction(lambda t: ConcatQuery(t) if len(t) > 1 else t[0])

    # one or more queries ORed together
    or_query << (concat_query + ZeroOrMore(Literal('|').suppress() + concat_query))
    or_query.setParseAction(lambda t: OrQuery(t) if len(t) > 1 else t[0])

    # the top symbol of our grammar
    basic_query = LineStart() + or_query + LineEnd()

    return basic_query
