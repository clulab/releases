package edu.arizona.sista.qa.translation

import collection.mutable.ArrayBuffer
import edu.arizona.sista.processors.Document
import edu.arizona.sista.qa.index.TermFilter
import TransView.termFilter
import org.slf4j.LoggerFactory
import edu.arizona.sista.processors.fastnlp.FastNLPProcessor
import edu.arizona.sista.qa.word2vec.Word2vec

/**
  * Storage class that allows a given document to be transferred into one or more different representations (lemmas, bigrams, dependencies, etc)
  * User: peter, gus
  * Date: 8/2/13, 4/1/14
  */
class TransView(doc: Document,
                val sanitizeForW2V: Boolean = false,
                val sanitizeForW2VKeepNumbers: Boolean = true) {

  lazy val processor = new FastNLPProcessor()
  var features = new ArrayBuffer[String]

  // used by views that join strings
  val delimiter = "__"
  // set upper bound to size of w2v window
  val sentenceBuffer = (for (_ <- 1 to 2) yield "<BOUNDARY>").toArray
  // create an Array of dependency-based relations to ignore
  val toIgnore = for (b <- Array("cop", "det", "expl", "predet", "punct", "quantmod", "poss")) yield s"$delimiter$b$delimiter"
  // lemma flag
  var useLemmas = false
  // filter flags
  var filterRelations = false
  var filterPunct = true

  def empty = features.size == 0

  def mkDepAnnotations(text: String): Document = {
    // minimum annotations; faster than annotate
    val doc = processor.mkDocument(text)
    // parse doc to extract dependencies
    processor.parse(doc)
    // clean up after annotators...
    doc.clear()
    doc
  }

  // allows passing multiple view types, separated by commas, then adds the features for each view type in turn
  def makeView(viewTypes: String) {
    for (view <- viewTypes.split(",").map(_.trim.toLowerCase)) {
      view match {

        // lemmas -- default is content-filtered
        case "lemmas_all" => makeTokenIncludeNoncontentView() // only do stopword and regex filtering
        case "lemmas_content" => makeTokenView() // do stopword, regex, and POS filtering
        case "lemmas_unfiltered" => makeLemmaUnfilteredView() // no filtering at all
        case "lemmas_nounverb" => makeTokenViewOnlyNounVerb()
        case "lemmas_content_nocause" => makeTokenViewNoCause()
        case "lemmas" => makeTokenView() // alias for lemmas_content

        // words -- default is not content-filtered
        case "words_all" => makeWordView() // only do stopword and regex filtering
        case "words_content" => makeWordContentView() // do stopword, regex, and POS filtering
        case "words_unfiltered" => makeWordUnfilteredView() // no filtering at all
        case "words" => makeWordView() // alias for words_all


        //n-gram based views
        case "bigrams" => makeNgramView(2)
        // case "sentences" => makeSentencesView()
        case _ => throw new RuntimeException(s"ERROR: view type $view not supported yet! Add it to TransView.makeView()")
      }
    }
    if (sanitizeForW2V)
      features = features.map(Word2vec.sanitizeWord(_, sanitizeForW2VKeepNumbers)).filter(_ != "")
  }

  // add all words, lowercased, but without stopword or non alpha num removal
  def makeWordUnfilteredView() {
    features ++= (for {
      sent <- doc.sentences
      word <- sent.words
    } yield word.toLowerCase)
  }

  // add all lemmas, lowercased, but without stopword or non alpha num removal
  def makeLemmaUnfilteredView() {
    features ++= (for {
      sent <- doc.sentences
      lemmas <- sent.lemmas.toSeq // convert option to seq so we can use in the for comprehension
      lemma <- lemmas
    } yield lemma.toLowerCase)
  }

  def makeTokenView() {
    // Extract lemmas from document.  Ignore stopwords and lemmas with tags other than NN*, VB*, and JJ*
    val filteredLemmas = termFilter.extractValidContentLemmas(doc, 0, doc.sentences.size)

    // Add lemmas to list of features for this document
    for (lemma <- filteredLemmas) {
      features += lemma.toLowerCase
    }
  }

  def makeTokenViewOnlyNounVerb() {
    // Extract lemmas from document.  Ignore stopwords and lemmas with tags other than NN*, VB*, and JJ*
    val filteredLemmas = termFilter.extractValidNounVerbLemmas(doc, 0, doc.sentences.size)

    // Add lemmas to list of features for this document
    for (lemma <- filteredLemmas) {
      features += lemma.toLowerCase
    }
  }

  def makeTokenViewNoCause() {
    // Extract lemmas from document.  Ignore stopwords and lemmas with tags other than NN*, VB*, and JJ*
    val toRemove = Array("cause", "result", "effect", "affect")
    val filteredLemmas = termFilter.extractValidContentLemmas(doc, 0, doc.sentences.size)
    val removedCause = filteredLemmas.filter(!toRemove.contains(_))

    // Add lemmas to list of features for this document
    for (lemma <- removedCause) {
      features += lemma.toLowerCase
    }
  }

  def makeTokenIncludeNoncontentView() {
    // names between token and word views (content vs include noncontent) are inconsistent for backward compatibility
    // Extract lemmas from document.  Ignore stopwords
    val filteredLemmas = termFilter.extractValidLemmas(doc, 0, doc.sentences.size)

    // Add lemmas to list of features for this document
    for (lemma <- filteredLemmas) {
      features += lemma.toLowerCase
    }
  }

  def makeWordView() {
    // Extract words from document.  Ignore stopwords
    val filteredWords = termFilter.extractValidWords(doc)

    // Add lemmas to list of features for this document
    for (word <- filteredWords) {
      features += word.toLowerCase
    }
  }

  def makeWordContentView() {
    for (word <- termFilter.extractValidContentWords(doc)) {
      features += word.toLowerCase
    }
  }



  // Ngram view
  def makeNgramView(n: Int, use_boundary: Boolean = false) {
    // Iterate over sentences
    for (sentence <- doc.sentences) {
      var ngramBuffer = new ArrayBuffer[String]
      if (sentence.words.size >= n) {
        for (i <- 0 to sentence.words.size - n) ngramBuffer += sentence.words.slice(i, i + n).mkString(delimiter)
        // Should we include sentence boundary tags?
        if (use_boundary) {
          ngramBuffer = ArrayBuffer("<BOUNDARY>_" + sentence.words.slice(0, n - 1).mkString(delimiter)) ++ ngramBuffer ++
            ArrayBuffer(sentence.words.slice(sentence.words.size - n + 1, sentence.words.size).mkString(delimiter) + "_<BOUNDARY>")
        }
      }
      // Add each ngram to the features Array
      for (ngram <- ngramBuffer) if (!hasPunctuation(ngram)) features += ngram.toLowerCase
    }
  }




  // helper method for PoS tag mappings
  def posMatch(tag: String): String = {
    //regex patterns for mappings to simplified tags
    val VERB = "MD|VB.*".r
    val NOUN = "NN.*".r // NOUN - nouns (common and proper)
    val PRON = "(WP|PRP)".r //PRON - pronouns
    val ADJ = "JJ".r //ADJ - adjectives
    val ADV = "W?RB".r //ADV - adverbs
    val ADP = "IN".r //ADP - adpositions (prepositions and postpositions)
    val CONJ = "CC".r //CONJ - conjunctions
    val DET = "W?DT".r //DET - determiners
    val NUM = "CD".r //NUM - cardinal numbers
    val PRT = "POS".r //  PRT - particles or other function words
    val PUNCT = "[^\\w\\s].*".r //  . - punctuation

    var simpleTag = "X" //X - other: foreign words, typos, abbreviations
    tag match {
      case VERB() => simpleTag = "VERB"
      case NOUN() => simpleTag = "NOUN"
      case PRON() => simpleTag = "PRON"
      case ADJ() => simpleTag = "ADJ"
      case ADV() => simpleTag = "ADJ"
      case ADP() => simpleTag = "ADP"
      case CONJ() => simpleTag = "CONJ"
      case DET() => simpleTag = "DET"
      case NUM() => simpleTag = "NUM"
      case PRT() => simpleTag = "PRT"
      case PUNCT() => simpleTag = "."
      case _ =>
    }
    simpleTag
  }

  // test for punctuation
  def hasPunctuation(word: String): Boolean = {
    val punctPattern = ".*[^\\w_].*".r
    // Do we need to filter?
    if (!filterPunct) return false
    word match {
      case punctPattern() => true
      case _ => false
    }
  }

  // for filtering unecessary relations
  def validRelation(representation: String): Boolean = {
    // do we want to filter?
    if (!filterRelations) return true
    else {
      // check to see whether or not the relation should be ignored
      for (i <- toIgnore) if (representation.contains(i)) return false
    }
    // return true if the relation appears valid
    true
  }

  override def toString: String = features.mkString(" ")
}

object TransView {
  val termFilter = new TermFilter()
  val logger = LoggerFactory.getLogger(classOf[TransView])

}
