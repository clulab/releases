# Towards Realistic Few-Shot Relation Extraction: A New Meta Dataset and Evaluation

# Test Episodes Guide

This is a quick guide to understanding the test episode files released in this repository.

## 1. Basic terms

- **Episode:** One support set together with three queries.
- **Support set:** The examples that demonstrate the five candidate relations.
- **Query:** A sentence and marked entity pair that the model must classify.
- **Head and tail:** The ordered pair of marked entities whose relation is being classified.
- **NOTA:** None of the five candidate relations in the current episode.

## 2. Top-level structure

The JSON root is an array with three parallel parts:

```text
[
  episodes,
  gold_labels,
  auxiliary_metadata
]
```

Items at the same index belong together:

```python
episode = episodes[i]
labels = gold_labels[i]
metadata = auxiliary_metadata[i]
```

| Part | Contents |
|---|---|
| `episodes[i]` | Support examples and three queries |
| `gold_labels[i]` | Correct local label for each query |
| `auxiliary_metadata[i]` | Candidate relation names and original query relation names |

### JSON tree

```text
JSON root
├── episodes
│   ├── meta_train
│   │   ├── id
│   │   ├── relation
│   │   ├── tokens
│   │   ├── h
│   │   ├── t
│   │   ├── head_end
│   │   └── tail_end
│   └── meta_test
│       ├── id
│       ├── relation
│       ├── tokens
│       ├── h
│       ├── t
│       ├── head_end
│       └── tail_end
├── gold_labels
└── auxiliary_metadata
```

## 3. Episode contents

Each stored episode has:

```json
{
  "meta_train": [],
  "meta_test": []
}
```

- **`meta_train`** contains the few-shot support examples. These examples show the model what each of the five candidate relation classes looks like.
- **`meta_test`** contains the three query examples. The model uses the support examples in `meta_train` to predict a relation class for each query.

### Few-shot support examples: `meta_train`

`meta_train` contains five groups:

```text
meta_train[0] = support examples for local class 0
meta_train[1] = support examples for local class 1
meta_train[2] = support examples for local class 2
meta_train[3] = support examples for local class 3
meta_train[4] = support examples for local class 4
```

Each group has one record in a 1-shot file or five records in a 5-shot file.

Despite its name, `meta_train` is not the global model-training split. It is the support set supplied inside that test episode.

### Queries: `meta_test`

`meta_test` contains three query records:

```text
meta_test[0] = query 0
meta_test[1] = query 1
meta_test[2] = query 2
```

The model predicts one label from `0` through `5` for each query.

## 4. Sentence records and entities

A support or query record has this general form:

```json
{
  "id": "example-001",
  "relation": "person:birth_date",
  "token": ["Alex", "was", "born", "in", "1980", "."],
  "tokens": ["Alex", "was", "born", "in", "1980", "."],
  "h": ["alex", null, [[0]]],
  "t": ["1980", null, [[4]]],
  "tokens_with_markers": [
    "[unused1]", "Alex", "[unused2]", "was", "born", "in",
    "[unused3]", "1980", "[unused4]", "."
  ],
  "head_after_bert": 1,
  "tail_after_bert": 7,
  "head_end": 3,
  "tail_end": 9
}
```

The important fields are:

- `relation`: The original annotated relation for the entity pair.
- `token` and `tokens`: Sentence tokens. They are normally identical in these files.
- `h`: Head-entity information.
- `t`: Tail-entity information.
- `tokens_with_markers`: The sentence with entity markers.
- The four position fields store marker positions used by the existing encoder.

The entity format is:

```text
[entity text, optional entity identifier, token-position spans]
```

For example:

```json
["patricia neal", null, [[2, 3]]]
```

This says that the entity is `patricia neal`, its optional identifier is unavailable, and it occupies zero-based token positions `2` and `3`.

The relation is directed:

```text
head --relation--> tail
```

The special markers mean:

```text
[unused1] ... [unused2] = head entity
[unused3] ... [unused4] = tail entity
```

They can be displayed to a person or prompted model as:

```text
<HEAD>Alex</HEAD> was born in <TAIL>1980</TAIL>.
```

The file marks the entities, but it does not mark a separate exact phrase as the relationship. The model infers the relationship from the sentence context.

## 5. Labels and auxiliary metadata

`gold_labels[i]` has one local label for each of the three queries:

```json
[0, 5, 2]
```

This means:

```text
query 0 -> class 0
query 1 -> NOTA
query 2 -> class 2
```

The matching metadata has two lists:

```json
[
  [
    "person:birth_date",
    "person:education",
    "person:residence",
    "organization:member",
    "organization:location"
  ],
  [
    "person:birth_date",
    "organization:alternate_name",
    "person:residence"
  ]
]
```

The first list maps local classes to target relations:

```text
0 -> person:birth_date
1 -> person:education
2 -> person:residence
3 -> organization:member
4 -> organization:location
5 -> NOTA
```

The second list gives the original relations of queries 0, 1, and 2. Therefore, the labels are `[0, 5, 2]`. Query 1 receives NOTA because its relation is not among the five target relations.

NOTA is episode-specific. It can mean a literal no-relation instance or any valid relation that is outside the five current targets.

Local labels are also episode-specific. Class `0` can represent different relations in different episodes.
