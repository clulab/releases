
object ObjTagSequence extends Enumeration
{
  type ObjPattern = Value

  // case where SUB+VERB+OBJ pattern exists. but OBJ is Determiner + Noun (NN (singular noun) or NNS (plural noun)) DT+NN
  // examples: a student, the student
  // Strategy: we should only replace DET with one of the values from determiners

  val DTNN = Value("DT+NN")
  val DTNNS = Value("DT+NNS")

  // case where SUB+VERB+OBJ pattern exists. But OBJ is ADJ + Noun (NN (singular noun) or NNS (plural noun)) : JJ+NN
  // examples: large classroom, large pond
  // Strategy: for "JJ +NN", replace JJ with one of the values from adjectives
  // Strategy 2: but if "DT + JJ +NN" exist, then use adj_spl array i.e. replace a DET/DT with adj_spl i.e.
  // Strategy 2 contd: **an abnormally** large pond, **an elegantly** large pond
  // Resolution: therefore check if "DT","JJ", "NN" -> pattern exists first before considering "JJ", "NN"
  val JJNN = Value("JJ+NN")
  val JJNNS = Value("JJ+NNS")
  val DTJJNN = Value("DT+JJ+NN")
  val DTJJNNS = Value("DT+JJ+NNS")

  // case SUB+VERB+OBJ pattern exists. But OBJ is "DT+NN+NN". Examples: "a school competition", "a wordle game"
  //(wordle : https://en.wiktionary.org/wiki/wordle different from present day "Wordle") but "Wordle" (with capital W) is an NNP and not NN.
  // **perks of coming up with a complex example.**
  // Strategy for adding adjectives: just insert JJ right after DT: a good school competition, a green school competition
  // Strategy for adding special adjectives for : just replace DT with DT+JJ right after determiner: an abnormal school competition
  // Strategy for determiners: every school competition, some school competition
  val DTNNNN = Value("DT+NN+NN")

  // case in point: also where only NP+VP exist but object is the adjective phrase ADJP such as 'of deadlines'.
  // IN+NNS. example: 'today is full of deadlines', 'december is full of exams', 'this coffee is full of sugar', 'life is full of surprises' "
  // so modified sentences can be: life is full of abnormal surprises, life is full of elegant surprises, life is full of good surprises,
  // life is full of bad surprises (very pessimistic)
  // strategy: insert before NNS
  val INNNS = Value("IN+NNS")

  // case in point: where only NP+VP exist i.e. object is the adjective i.e. JJ. example sentences: 'this is good', 'coffee is good' "
  // modified examples: VBZ+JJ -> this is abnormally good, this is bad good, this is green good ? does not make sense but generate anyway
  // note: choose VBZ+JJ but replace this_index+1 i.e. the one after VBZ -> will be replacing the JJ
  val VBZJJ = Value("VBZ+JJ")

  // ************ added as per Mihai's review *****************
  // case where the rightmost head points to an nmod with Preposition Phrase. Then we extract leftmost NP
  // example: on the bag of a girl - nmod in dependency parse i.e  PP in constituency parse.
  // here we would have nominal modifier in dependency parse with a pattern of IN+DT+NN where we modify DT.
  // the pattern for on the bag of a girl -> IN DT NN IN DT NN .
  // So we want the left most Noun Phrase in Constituency parse i.e. the first IN DT NN (regardless of what the rest of pattern is in nmod head)
  val INDTNN = Value("IN+DT+NN")
  val INDTNNS = Value("IN+DT+NNS")

  val NONE = Value("NONE")

}