from __future__ import annotations
import re
import json
from typing import List, Union
from utils import filter_query_holes
import config


hole = config.HOLE_GLYPH
inf = config.INFINITY


class AstNode:

    def pattern(self) -> str:
        raise NotImplementedError

    def num_holes(self) -> int:
        raise NotImplementedError

    def has_holes(self) -> bool:
        raise NotImplementedError

    def is_valid_query(self) -> bool:
        return False

    def to_binary_right_heavy(self) -> AstNode:
        raise NotImplementedError

    def get_tokens(self) -> List[str]:
        raise NotImplementedError 

    def next_nodes(self, vocabularies: dict[str, List[str]]) -> List[AstNode]:
        raise NotImplementedError 

    def get_leftmost_hole(self) -> Union[HoleMatcher, HoleConstraint, HoleQuery]:
        raise NotImplementedError 

    # [#(Hole Query), #(Hole Constraint), #(Hole Matcher)]
    def num_holes_by_type(self) -> list:
        raise NotImplementedError 

    def next_nodes_filtered(self, vocabulary: dict):
        return list(filter(lambda l: filter_query_holes(l), self.next_nodes(vocabulary)))

##########


class HoleMatcher(AstNode):

    def pattern(self):
        return hole

    def num_holes(self):
        return 1

    def has_holes(self):
        return True

    def to_binary_right_heavy(self):
        return self

    def next_nodes(self, vocabularies: dict[str, List[str]]) -> List[AstNode]:
        return []

    def get_leftmost_hole(self) -> Union[HoleMatcher, HoleConstraint, HoleQuery]:
        return self

    # [#(Hole Query), #(Hole Constraint), #(Hole Matcher)]
    def num_holes_by_type(self) -> list:
        return [0, 0, 1]

class StringMatcher(AstNode):

    # TODO Investigate when there is a '"' inside
    # ([word=prosperity]) ([word="\",\""])
    # ([word=prosperity]) ([word=","]) ## query.pattern().replace("\\\"", "")
    # construct_query result:
    # {'query': '([word=prosperity]) ([word=","])', 'sentences': [['Every', 'man', 'is', 'taught', 'to', 'consider', 'his', 'own', 'happiness', ',', 'as', 'combined', 'with', 'the', 'publick', 'prosperity', ',', 'and', 'to', 'think', 'himself', 'great', 'and', 'powerful', ',', 'in', 'proportion', 'to', 'the', 'greatness', 'and', 'power', 'of', 'his', 'governours', '.'], ['So', 'perhaps', ',', 'President', 'Bush', 'and', 'these', 'historians', 'are', 'in', 'consensus', 'on', 'the', 'correlation', 'of', 'liberty', ',', 'innovation', 'and', 'prosperity', '.']], 'specs': [{'docId': 'test', 'sentId': 0, 'start': 15, 'end': 17}, {'docId': 'test', 'sentId': 1, 'start': 19, 'end': 21}]}
    def __init__(self, string: str):
        if string.isalpha():
            self.string = string
        else:
            self.string = f'"{string}"'
            

    def pattern(self):
        if re.match(r'^[\w_][\w\d_]*$', self.string):
            return self.string
        else:
            return json.dumps(self.string)

    def num_holes(self):
        return 0

    def has_holes(self):
        return False

    def to_binary_right_heavy(self):
        return self

    def next_nodes(self, vocabularies: dict[str, List[str]]) -> List[AstNode]:
        return []

    def get_leftmost_hole(self) -> Union[HoleMatcher, HoleConstraint, HoleQuery]:
        return None

    def num_holes_by_type(self) -> list:
        return [0, 0, 0]

##########

class TokenConstraint(AstNode):
    pass

class HoleConstraint(TokenConstraint):

    def pattern(self):
        return hole

    def num_holes(self):
        return 1

    def has_holes(self):
        return True

    def to_binary_right_heavy(self):
        return self

    def get_tokens(self):
        return ['AST-HoleConstraint']

    def next_nodes(self, vocabularies: dict[str, List[str]]) -> List[AstNode]:
        return [
            FieldConstraint(HoleMatcher(), HoleMatcher()),
            NotConstraint(HoleConstraint()),
            AndConstraint([HoleConstraint(), HoleConstraint()]),
            OrConstraint([HoleConstraint(), HoleConstraint()]),
        ]

    def get_leftmost_hole(self) -> Union[HoleMatcher, HoleConstraint, HoleQuery]:
        return self

    def num_holes_by_type(self) -> list:
        return [0, 1, 0]

class FieldConstraint(TokenConstraint):
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def pattern(self):
        return f'{self.name.pattern()}={self.value.pattern()}'

    def num_holes(self):
        return self.name.num_holes() + self.value.num_holes()

    def has_holes(self):
        return self.name.has_holes() or self.value.has_holes()

    def to_binary_right_heavy(self):
        name = self.name.to_binary_right_heavy()
        value = self.value.to_binary_right_heavy()
        return FieldConstraint(name, value)

    def get_tokens(self):
        if isinstance(self.name, HoleMatcher):
            return [
                'AST-FieldConstraint-start',
                'AST-HoleMatcher',
                'AST-HoleMatcher',
                'AST-FieldConstraint-end'
            ]
        elif isinstance(self.value, HoleMatcher):
            return [
                'AST-FieldConstraint-start',
                f'fieldname-{self.name.string}',
                'AST-HoleMatcher',
                'AST-FieldConstraint-end'
             ]
        else:
            return [
                'AST-FieldConstraint-start',
                f'fieldname-{self.name.string}',
                f'{self.value.string}',
                'AST-FieldConstraint-end'
            ]

    def next_nodes(self, vocabularies: dict[str, List[str]]) -> List[AstNode]:
        # Two holes means both are holes, so we fill name first
        if self.num_holes() == 2:
            return [FieldConstraint(StringMatcher(x), self.value) for x in vocabularies.keys()]
        # One hole means that we have to fill value (otherwise error)
        elif self.num_holes() == 1:
            if self.name.num_holes == 1:
                raise ValueError("The left hole (name) should have been filled first. Something went wrong")
            return [FieldConstraint(self.name, StringMatcher(x)) for x in vocabularies[self.name.pattern()]]
        # No hole means that there is nothing to fill. Return empty
        else:
            return []

    def get_leftmost_hole(self) -> Union[HoleMatcher, HoleConstraint, HoleQuery]:
        return self.name.get_leftmost_hole() or self.value.get_leftmost_hole()

    def num_holes_by_type(self) -> list:
        n = self.name.num_holes_by_type()
        v = self.value.num_holes_by_type()
        return [n[0] + v[0], n[1] + v[1], n[2] + v[2]]

class NotConstraint(TokenConstraint):
    def __init__(self, constraint):
        self.constraint = constraint

    def pattern(self):
        return f'!({self.constraint.pattern()})'

    def num_holes(self):
        return self.constraint.num_holes()

    def has_holes(self):
        return self.constraint.has_holes()

    def to_binary_right_heavy(self):
        constraint = self.constraint.to_binary_right_heavy()
        return NotConstraint(constraint)

    def get_tokens(self):
        return [
            'AST-NotConstraint-start',
            *self.constraint.get_tokens(),
            'AST-NotConstraint-end',
        ]
    
    def next_nodes(self, vocabularies: dict[str, List[str]]) -> List[AstNode]:
        if self.num_holes() == 0:
            return []
        
        nodes = filter(lambda x: not isinstance(x, NotConstraint), self.constraint.next_nodes(vocabularies))
        result = []

        for n in nodes:
            if isinstance(n, TokenConstraint):
                result.append(NotConstraint(n))
            else:
                raise ValueError(f"The expansion of {self.constraint} was expected to consist of TokenConstraint only (HoleConstraint or FieldConstraint)")

        return result

    def get_leftmost_hole(self) -> Union[HoleMatcher, HoleConstraint, HoleQuery]:
        return self.constraint.get_leftmost_hole()

    def num_holes_by_type(self) -> list:
        return self.constraint.num_holes_by_type()

class AndConstraint(TokenConstraint):
    def __init__(self, constraints):
        self.constraints = list(constraints)

    def pattern(self):
        return ' & '.join(f'({c.pattern()})' for c in self.constraints)

    def num_holes(self):
        return sum(c.num_holes() for c in self.constraints)

    def has_holes(self):
        return any(c.has_holes() for c in self.constraints)

    def to_binary_right_heavy(self):
        lhs = self.constraints[0].to_binary_right_heavy()
        if len(self.constraints) == 2:
            rhs = self.constraints[1].to_binary_right_heavy()
        else:
            rhs = AndConstraint(self.constraints[1:]).to_binary_right_heavy()
        return AndConstraint([lhs, rhs])

    def get_tokens(self):
        tokens = ['AST-AndConstraint-start']
        tokens.extend(self.constraints[0].get_tokens())
        for c in self.constraints[1:]:
            tokens.append('AST-AndConstraint-sep')
            tokens.extend(c.get_tokens())
        tokens.append('AST-AndConstraint-end')
        return tokens

    def next_nodes(self, vocabularies: dict[str, List[str]]) -> List[AstNode]:
        if len([x for x in self.constraints if x.has_holes]) == 0:
            return []
        
        i = self.constraints.index(next(x for x in self.constraints if x.has_holes()))
        constraint = self.constraints[i]
        nodes = constraint.next_nodes(vocabularies)
        result = []

        for n in nodes:
            if isinstance(n, AndConstraint) and self != self.to_binary_right_heavy():
                newConstraints = self.constraints[:i] + [HoleConstraint()] + self.constraints[(i+1):]
            elif isinstance(n, TokenConstraint):
                newConstraints = self.constraints[:i] + [n] + self.constraints[(i+1):]
            else:
                raise ValueError(f"This type of node ({n}) is unexpected")
            result.append(AndConstraint(newConstraints))
        return result

    def get_leftmost_hole(self) -> Union[HoleMatcher, HoleConstraint, HoleQuery]:
        return [c.get_leftmost_hole() for c in self.constraints if c is not None][0]

    def num_holes_by_type(self) -> list:
        result = [x.num_holes_by_type() for x in self.constraints]
        return [sum([x[0] for x in result]), sum([x[1] for x in result]), sum([x[2] for x in result])]

class OrConstraint(TokenConstraint):
    def __init__(self, constraints):
        self.constraints = list(constraints)

    def pattern(self):
        return ' | '.join(f'({c.pattern()})' for c in self.constraints)

    def num_holes(self):
        return sum(c.num_holes() for c in self.constraints)

    def has_holes(self):
        return any(c.has_holes() for c in self.constraints)

    def to_binary_right_heavy(self):
        lhs = self.constraints[0].to_binary_right_heavy()
        if len(self.constraints) == 2:
            rhs = self.constraints[1].to_binary_right_heavy()
        else:
            rhs = OrConstraint(self.constraints[1:]).to_binary_right_heavy()
        return OrConstraint([lhs, rhs])

    def get_tokens(self):
        tokens = ['AST-OrConstraint-start']
        tokens.extend(self.constraints[0].get_tokens())
        for c in self.constraints[1:]:
            tokens.append('AST-OrConstraint-sep')
            tokens.extend(c.get_tokens())
        tokens.append('AST-OrConstraint-end')
        return tokens

    def next_nodes(self, vocabularies: dict[str, List[str]]) -> List[AstNode]:
        if len([x for x in self.constraints if x.has_holes]) == 0:
            return []
        
        i = self.constraints.index(next(x for x in self.constraints if x.has_holes()))
        constraint = self.constraints[i]
        nodes = constraint.next_nodes(vocabularies)
        result = []

        for n in nodes:
            if isinstance(n, OrConstraint) and self != self.to_binary_right_heavy():
                newConstraints = self.constraints[:i] + [HoleConstraint()] + self.constraints[(i+1):]
            elif isinstance(n, TokenConstraint):
                newConstraints = self.constraints[:i] + [n] + self.constraints[(i+1):]
            else:
                raise ValueError(f"This type of node ({n}) is unexpected")
            result.append(OrConstraint(newConstraints))
        return result

    def get_leftmost_hole(self) -> Union[HoleMatcher, HoleConstraint, HoleQuery]:
        return [c.get_leftmost_hole() for c in self.constraints if c is not None][0]

    def num_holes_by_type(self) -> list:
        result = [x.num_holes_by_type() for x in self.constraints]
        return [sum([x[0] for x in result]), sum([x[1] for x in result]), sum([x[2] for x in result])]

##########


class Query(AstNode):
    def is_valid_query(self):
        return not self.has_holes()


class HoleQuery(Query):
    def pattern(self):
        return hole

    def num_holes(self):
        return 1

    def has_holes(self):
        return True

    def to_binary_right_heavy(self):
        return self

    def get_tokens(self):
        return ['AST-HoleQuery']

    def next_nodes(self, vocabularies: dict[str, List[str]]) -> List[AstNode]:
        return [
            TokenQuery(HoleConstraint()),
            ConcatQuery([HoleQuery(), HoleQuery()]),
            OrQuery([HoleQuery(), HoleQuery()]),
            RepeatQuery(HoleQuery(), 0, 1),
            RepeatQuery(HoleQuery(), 0, inf),
            RepeatQuery(HoleQuery(), 1, inf),
        ]

    def get_leftmost_hole(self) -> Union[HoleMatcher, HoleConstraint, HoleQuery]:
        return self

    def num_holes_by_type(self) -> list:
        return [1, 0, 0]

class TokenQuery(Query):
    def __init__(self, constraint):
        self.constraint = constraint

    def pattern(self):
        return f'[{self.constraint.pattern()}]'

    def num_holes(self):
        return self.constraint.num_holes()

    def has_holes(self):
        return self.constraint.has_holes()

    def to_binary_right_heavy(self):
        constraint = self.constraint.to_binary_right_heavy()
        return TokenQuery(constraint)

    def get_tokens(self):
        return [
            'AST-TokenQuery-start',
            *self.constraint.get_tokens(),
            'AST-TokenQuery-end',
        ]

    def next_nodes(self, vocabularies: dict[str, List[str]]) -> List[AstNode]:
        if self.num_holes() == 0:
            return []
        nodes = self.constraint.next_nodes(vocabularies)
        result = []

        for n in nodes:
            if isinstance(n, TokenConstraint):
                result.append(TokenQuery(n))
            else:
                raise ValueError(f"This type of node ({n}) is unexpected")
        return result

    def get_leftmost_hole(self) -> Union[HoleMatcher, HoleConstraint, HoleQuery]:
        return self.constraint.get_leftmost_hole()

    def num_holes_by_type(self) -> list:
        return self.constraint.num_holes_by_type()


class ConcatQuery(Query):
    def __init__(self, queries):
        self.queries = list(queries)

    def pattern(self):
        return ' '.join(f'({q.pattern()})' for q in self.queries)

    def num_holes(self):
        return sum(q.num_holes() for q in self.queries)

    def has_holes(self):
        return any(q.has_holes() for q in self.queries)

    def to_binary_right_heavy(self):
        lhs = self.queries[0].to_binary_right_heavy()
        if len(self.queries) == 2:
            rhs = self.queries[1].to_binary_right_heavy()
        else:
            rhs = ConcatQuery(self.queries[1:]).to_binary_right_heavy()
        return ConcatQuery([lhs, rhs])

    def get_tokens(self):
        tokens = ['AST-ConcatQuery-start']
        tokens.extend(self.queries[0].get_tokens())
        for q in self.queries[1:]:
            tokens.append('AST-ConcatQuery-sep')
            tokens.extend(q.get_tokens())
        tokens.append('AST-ConcatQuery-end')
        return tokens

    def next_nodes(self, vocabularies: dict[str, List[str]]) -> List[AstNode]:
        if len([x for x in self.queries if x.has_holes()]) == 0:
            return []
        
        i = self.queries.index(next(x for x in self.queries if x.has_holes()))
        query = self.queries[i]
        nodes = query.next_nodes(vocabularies)
        result = []
        for n in nodes:
            if isinstance(n, ConcatQuery) and self != self.to_binary_right_heavy():
                new_queries = self.queries[:i] + [HoleQuery()] + self.queries[i:]
            elif isinstance(n, Query):
                new_queries = self.queries[:i] + [n] + self.queries[(i+1):]
            else:
                raise ValueError(f"This type of node ({n}) is unexpected")
            result.append(ConcatQuery(new_queries))
        return result

    def get_leftmost_hole(self) -> Union[HoleMatcher, HoleConstraint, HoleQuery]:
        return list(filter(None, [c.get_leftmost_hole() for c in self.queries]))[0]

    def num_holes_by_type(self) -> list:
        result = [x.num_holes_by_type() for x in self.queries]
        return [sum([x[0] for x in result]), sum([x[1] for x in result]), sum([x[2] for x in result])]

class OrQuery(Query):
    def __init__(self, queries):
        self.queries = list(queries)

    def pattern(self):
        return ' | '.join(f'({q.pattern()})' for q in self.queries)

    def num_holes(self):
        return sum(q.num_holes() for q in self.queries)

    def has_holes(self):
        return any(q.has_holes() for q in self.queries)

    def to_binary_right_heavy(self):
        lhs = self.queries[0].to_binary_right_heavy()
        if len(self.queries) == 2:
            rhs = self.queries[1].to_binary_right_heavy()
        else:
            rhs = OrQuery(self.queries[1:]).to_binary_right_heavy()
        return OrQuery([lhs, rhs])

    def get_tokens(self):
        tokens = ['AST-OrQuery-start']
        tokens.extend(self.queries[0].get_tokens())
        for q in self.queries[1:]:
            tokens.append('AST-OrQuery-sep')
            tokens.extend(q.get_tokens())
        tokens.append('AST-OrQuery-end')
        return tokens

    def next_nodes(self, vocabularies: dict[str, List[str]]) -> List[AstNode]:
        if len([x for x in self.queries if x.has_holes]) == 0:
            return []
        
        i = self.queries.index(next(x for x in self.queries if x.has_holes()))
        query = self.queries[i]
        nodes = query.next_nodes(vocabularies)
        result = []

        for n in nodes:
            if isinstance(n, OrQuery) and self != self.to_binary_right_heavy():
                new_queries = self.queries[:i] + [HoleQuery()] + self.queries[(i+1):]
            elif isinstance(n, Query):
                new_queries = self.queries[:i] + [n] + self.queries[(i+1):]
            else:
                raise ValueError(f"This type of node ({n}) is unexpected")
            result.append(ConcatQuery(new_queries))
        return result

    def get_leftmost_hole(self) -> Union[HoleMatcher, HoleConstraint, HoleQuery]:
        return list(filter(None, [c.get_leftmost_hole() for c in self.queries]))[0]

    def num_holes_by_type(self) -> list:
        result = [x.num_holes_by_type() for x in self.queries]
        return [sum([x[0] for x in result]), sum([x[1] for x in result]), sum([x[2] for x in result])]

class RepeatQuery(Query):
    def __init__(self, query, min, max):
        self.query = query
        self.min = min
        self.max = max

    def pattern(self):
        return f'({self.query.pattern()}){self.quantifier()}'

    def num_holes(self):
        return self.query.num_holes()

    def has_holes(self):
        return self.query.has_holes()

    def to_binary_right_heavy(self):
        query = self.query.to_binary_right_heavy()
        return RepeatQuery(query, self.min, self.max)

    def get_tokens(self):
        return [
            'AST-RepeatQuery-start',
            f'AST-quantifier-{self.quantifier()}',
            *self.query.get_tokens(),
            'AST-RepeatQuery-end',
        ]

    def quantifier(self):
        if self.min == 0 and self.max == 1:
            return '?'
        elif self.min == 0 and self.max == inf:
            return '*'
        elif self.min == 1 and self.max == inf:
            return '+'
        elif self.min == 0 and isinstance(self.max, int):
            return f'{{,{self.max}}}'
        elif self.max == inf and isinstance(self.min, int):
            return f'{{{self.min},}}'
        elif isinstance(self.min, int) and isinstance(self.max, int):
            if self.min == self.max:
                return f'{{{self.min}}}'
            else:
                return f'{{{self.min},{self.max}}}'
        else:
            raise Exception('invalid quantifier')

    def next_nodes(self, vocabularies: dict[str, List[str]]) -> List[AstNode]:
        if self.num_holes() == 0:
            return []
        nodes = filter(lambda x: not isinstance(x, RepeatQuery), self.query.next_nodes(vocabularies))
        result = []

        for n in nodes:
            if isinstance(n, Query):
                result.append(RepeatQuery(n, self.min, self.max))
            else:
                raise ValueError(f"This type of node ({n}) is unexpected")
        return result

    def get_leftmost_hole(self) -> Union[HoleMatcher, HoleConstraint, HoleQuery]:
        return self.query.get_leftmost_hole()

    def num_holes_by_type(self) -> list:
        return self.query.num_holes_by_type()
