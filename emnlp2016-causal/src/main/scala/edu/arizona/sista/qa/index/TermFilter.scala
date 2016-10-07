package edu.arizona.sista.qa.index

import collection.mutable
import org.apache.lucene.analysis.standard.StandardAnalyzer
import scala.collection.JavaConversions._
import collection.mutable.ListBuffer
import edu.arizona.sista.qa.segmenter.Segment
import edu.arizona.sista.processors.fastnlp.FastNLPProcessor

/**
 * Maintains the set of stop words in the system
 * User: mihais
 * Date: 3/7/13
 */
class TermFilter(val keepConnectives:Boolean = false) {
  import TermFilter.isContentTag

  val stopWords = mkStopWords(keepConnectives)

  private def mkStopWords(keepConnectives:Boolean):Set[String] = {
    val sw = new mutable.HashSet[String]()
    for (word <- StandardAnalyzer.STOP_WORDS_SET) {
      if (word.isInstanceOf[String]) {
        sw += word.asInstanceOf[String]
      }
    }

    if(keepConnectives) {
      // TODO
    }

    // QA specific stop words
    sw += "how"
    sw += "why"

    sw.toSet
  }

  def validToken(word:String, lemma:String):Boolean = {

    // must contain a letter or digit
    var hasAlpha = false
    var i = 0
    while(i < word.length && ! hasAlpha) {
      val c = word.charAt(i)
      if (Character.isLetterOrDigit(c)) {
        hasAlpha = true
      }
      i += 1
    }
    if(! hasAlpha) return false

    // skip URLS
    val URLCONTENTS = ".*\\.(org|edu|mil|gov|net|com|co\\.[a-z]{2}).*" // see if urls contain common domain extensions
    if(word.startsWith("http") || word.matches(URLCONTENTS)) return false // improved url coverage

    // cannot be a stop word
    if (stopWords.contains(word)) return false
    if (stopWords.contains(lemma)) return false

    true
  }

  def extractValidWords(doc:edu.arizona.sista.processors.Document):List[String] =
    extractValidWords(doc, 0, doc.sentences.length)

  def extractValidWords(doc:edu.arizona.sista.processors.Document, startSent:Int, endSent:Int):List[String] = {
    val words = new ListBuffer[String]
    var i = startSent
    while(i < endSent) {
      val s = doc.sentences(i)
      var j = 0
      while(j < s.size) {
        val w = s.words(j)
        val l = s.lemmas.get(j)
        if (validToken(w, l)) words += w
        j += 1
      }
      i += 1
    }
    words.toList
  }

  def extractValidContentWords(doc:edu.arizona.sista.processors.Document):List[String] = {
    val words = new ListBuffer[String]
    for (s <- doc.sentences) {
      var j = 0
      while(j < s.size) {
        val w = s.words(j)
        val l = s.lemmas.get(j)
        val tag = s.tags.get(j)
        if (validToken(w, l) && (isContentTag(tag))) {
          words += w
        }
        j += 1
      }
    }
    words.toList
  }

  def extractValidNounVerbWords(doc:edu.arizona.sista.processors.Document):List[String] = {
    val words = new ListBuffer[String]
    for (s <- doc.sentences) {
      var j = 0
      while(j < s.size) {
        val w = s.words(j)
        val l = s.lemmas.get(j)
        val tag = s.tags.get(j)
        if (validToken(w, l) && TermFilter.isNounVerbTag(tag)) {
          words += w
        }
        j += 1
      }
    }
    words.toList
  }

  def extractValidLemmas(doc:edu.arizona.sista.processors.Document):List[String] =
    extractValidLemmas(doc, 0, doc.sentences.length)

  def extractValidLemmas(doc:edu.arizona.sista.processors.Document, startSent:Int, endSent:Int):List[String] = {
    val lemmas = new ListBuffer[String]
    var i = startSent
    while(i < endSent) {
      val s = doc.sentences(i)
      //if (s.lemmas.isEmpty) throw new RuntimeException("ERROR: lemmas must exist for TermFilter.extractValidLemmas!")
      if (s.lemmas.isEmpty) {
        println ("ERROR: lemmas must exist for TermFilter.extractValidLemmas!")
        return lemmas.toList
      }
      var j = 0
      while(j < s.size) {
        val w = s.words(j)
        val l = s.lemmas.get(j)
        if (validToken(w, l)) lemmas += l
        j += 1
      }
      i += 1
    }
    lemmas.toList
  }

   def extractValidLemmasFromArray(in:Array[String]):List[String] = {
    val lemmas = new ListBuffer[String]
    for (w <- in) {
      if (validToken(w, w)) lemmas += w
    }
    lemmas.toList
  }

  def extractValidLemmasFromSegment(seg:Segment):List[String] = {
    // TODO: For discourse matching, many of the stop words are on the cue phrases list.  A validToken() method
    // should be created that addresses this.
    val outLemmas = new ListBuffer[String]
    val start = seg.startOffset
    val end = seg.endOffset

    for (s <- start._1 to end._1) {
      val sent = seg.doc.sentences(s)
      //if (sent.lemmas.isEmpty) throw new RuntimeException("ERROR: lemmas must exist for SegmentMatcherBOW.extractValidLemmasFromSegment()!")
      if (sent.lemmas.isEmpty) {
        println("ERROR: lemmas must exist for SegmentMatcherBOW.extractValidLemmasFromSegment()!")
        return outLemmas.toList
      }

      // Determine start/stop offsets for the current sentence in a given segment
      var sStart = 0
      var sEnd = 0
      if (s == start._1 && s == end._1) {
        // Segment spans single sentence
        sStart = start._2
        sEnd = end._2
      } else {
        // Segment spans multiple sentences
        if (s == start._1) {
          // we're on the first sentence
          sStart = start._2
          sEnd = sent.size
        } else if (s == end._1) {
          // we're on the last sentence
          sStart = 0
          sEnd = end._2
        } else {
          // we're on a middle sentence
          sStart = 0
          sEnd = sent.size
        }
      }

      for (i <- sStart until sEnd) {
        val word = sent.words(i)
        val lemma = sent.lemmas.get(i)
        if (validToken(word, lemma)) outLemmas += lemma
      }
    }
    //    println (" *extractValidLemmasFromSegment: " + outLemmas.toList)
    outLemmas.toList
  }

  def extractValidContentLemmas(doc: edu.arizona.sista.processors.Document): List[String] =
    extractValidContentLemmas(doc, 0, doc.sentences.size)

  def extractValidContentLemmas(doc: edu.arizona.sista.processors.Document, startSent: Int, endSent: Int): List[String] = {
    val lemmas = new ListBuffer[String]
    var i = startSent
    while (i < endSent) {
      val s = doc.sentences(i)
      //if (s.lemmas.isEmpty) throw new RuntimeException("ERROR: lemmas must exist for TermFilter.extractValidLemmas!")
      if (s.lemmas.isEmpty) {
        println("ERROR: lemmas must exist for TermFilter.extractValidContentLemmas!")
        return lemmas.toList
      }
      var j = 0
      while (j < s.size) {
        val w = s.words(j)
        val l = s.lemmas.get(j)
        val tag = s.tags.get(j)
        if (validToken(w, l)) {
          if (isContentTag(tag)) {
            lemmas += l
          }
        }
        j += 1
      }
      i += 1
    }

    lemmas.toList
  }

  def extractValidContentLemmasFromSegment(seg:Segment):List[String] = {
    val outLemmas = new ListBuffer[String]
    val start = seg.startOffset
    val end = seg.endOffset

    for (s <- start._1 to end._1) {
      val sent = seg.doc.sentences(s)
      //if (sent.lemmas.isEmpty) throw new RuntimeException("ERROR: lemmas must exist for SegmentMatcherBOW.extractValidLemmasFromSegment()!")
      if (sent.lemmas.isEmpty) {
        println("ERROR: lemmas must exist for SegmentMatcherBOW.extractValidLemmasFromSegment()!")
        return outLemmas.toList
      }

      // Determine start/stop offsets for the current sentence in a given segment
      var sStart = 0
      var sEnd = 0
      if (s == start._1 && s == end._1) {
        // Segment spans single sentence
        sStart = start._2
        sEnd = end._2
      } else {
        // Segment spans multiple sentences
        if (s == start._1) {
          // we're on the first sentence
          sStart = start._2
          sEnd = sent.size
        } else if (s == end._1) {
          // we're on the last sentence
          sStart = 0
          sEnd = end._2
        } else {
          // we're on a middle sentence
          sStart = 0
          sEnd = sent.size
        }
      }

      for (i <- sStart until sEnd) {
        val word = sent.words(i)
        val lemma = sent.lemmas.get(i)
        val tag = sent.tags.get(i)
        if (validToken(word, lemma)) {
          if (isContentTag(tag)) {
            outLemmas += lemma
          }
        }
      }
    }
    //    println (" *extractValidLemmasFromSegment: " + outLemmas.toList)
    outLemmas.toList
  }

  def extractValidNounVerbLemmas(doc: edu.arizona.sista.processors.Document): List[String] =
    extractValidNounVerbLemmas(doc, 0, doc.sentences.size)


  def extractValidNounVerbLemmas(doc: edu.arizona.sista.processors.Document, startSent: Int, endSent: Int): List[String] = {
    val lemmas = new ListBuffer[String]
    var i = startSent
    while (i < endSent) {
      val s = doc.sentences(i)
      //if (s.lemmas.isEmpty) throw new RuntimeException("ERROR: lemmas must exist for TermFilter.extractValidLemmas!")
      if (s.lemmas.isEmpty) {
        println("ERROR: lemmas must exist for TermFilter.extractValidContentLemmas!")
        return lemmas.toList
      }
      var j = 0
      while (j < s.size) {
        val w = s.words(j)
        val l = s.lemmas.get(j)
        val tag = s.tags.get(j)
        if (validToken(w, l)) {
          if (TermFilter.isNounVerbTag(tag)) {
            lemmas += l
          }
        }
        j += 1
      }
      i += 1
    }

    lemmas.toList
  }

}

object TermFilter {
  def isContentTag(tag: String) = {
    tag.startsWith("NN") || tag.startsWith("VB") || tag.startsWith("JJ") || tag.startsWith("RB")
  }

  def isNounVerbTag(tag: String) = {
    tag.startsWith("NN") || tag.startsWith("VB")
  }

}
