package edu.arizona.sista.qa.discparser

import java.io.File
import scala.io.Source
import java.util.regex.{Pattern,Matcher}
import scala.collection.mutable.ArrayBuffer
import org.slf4j.LoggerFactory
import edu.arizona.sista.processors.{Sentence, Document}
import edu.arizona.sista.processors.corenlp.CoreNLPProcessor
import edu.arizona.sista.struct.MutableNumber

/**
 * Parses the output of the discourse parser
 * User: mihais
 * Date: 8/6/13
 */
class DiscReader(val discourseDir:String) {
  val proc = new CoreNLPProcessor()

  def parse(docid:String, verbose:Boolean = false):Node = {
    var f = new File(discourseDir + File.separator + docid + ".txt.dis")
    if(f.exists()) return parseDis(f, verbose)
    f = new File(discourseDir + File.separator + docid + ".txt.edus")
    if(f.exists()) return parseEdus(f, verbose)
    throw new RuntimeException("ERROR: unknown docid " + docid + " in discourse repo!")
  }

  def parseEdus(file:File, verbose:Boolean = false):Node = {
    val edus = new ArrayBuffer[String]()
    val source = Source.fromFile(file)
    for(l <- source.getLines()) {
      if(! l.startsWith("<edu>") && ! l.endsWith("</edu>"))
        throw new RuntimeException("ERROR: incorrect <edu> format: " + l)
      edus += l.substring(5, l.length - 6)
    }
    source.close

    val tree = parseEdus(edus.toArray, 0)
    propagateText(tree)
    findIntraSentence(tree, verbose)
    tree
  }

  def parseEdus(edus:Array[String], offset:Int):Node = {
    if(offset == edus.length - 1) {
      new Node("Nucleus", "", annotate(edus(offset)), null)
    } else {
      val children = new Array[Node](2)
      children(0) = new Node("Nucleus", "", annotate(edus(offset)), null)
      children(1) = parseEdus(edus, offset + 1)
      new Node("Nucleus", "X", null, children)
    }
  }

  def parseDis(file:File, verbose:Boolean = false):Node = {
    val sb = new StringBuilder
    val source = Source.fromFile(file)
    for(l <- source.getLines()) {
      sb.append(l)
      sb.append(" ")
    }
    source.close()

    val tokens = tokenizeDis(sb.toString())
    if(verbose) {
      var i = 0
      while(i < tokens.length) {
        println((i + 1) + ": " + tokens(i))
        i += 1
      }
    }

    var offset = new MutableNumber[Int](0)
    var root = parseDisTokens(tokens, offset)
    if(verbose) {
      println("Raw tree:\n" + root)
    }

    propagateLabels(root)
    if(verbose) {
      println("After label propagation:\n" + root)
    }

    propagateText(root)
    if(verbose) {
      println("After text propagation:\n" + root)
    }

    findIntraSentence(root, verbose)
    if(verbose) {
      println("After finding intra-sentence nodes:\n" + root)
    }

    root
  }

  def findIntraSentence(node:Node, verbose:Boolean) {
    // intra-sentence nodes can have EOS punctuation only at the end of the text
    var foundEOS = false
    if(verbose) println("Inspecting text: " + DiscReader.textToString(node.text, Int.MaxValue))
    for(is <- 0 until node.text.sentences.length if ! foundEOS) {
      val s = node.text.sentences(is)
      for(it <- 0 until s.words.length if ! foundEOS) {
        if(is != node.text.sentences.length - 1 || it != s.words.length - 1) {
          val t = s.words(it)
          if(DiscReader.EOS.matcher(t).matches()) {
            foundEOS = true
            if(verbose) println("\tFOUND EOS: " + t)
          }
        }
      }
    }
    node.isIntraSentence = Some(! foundEOS)
    if(verbose) println("\tfoundEOS = " + foundEOS)

    if(! node.isTerminal)
      for(c <- node.children)
        findIntraSentence(c, verbose)
  }

  def propagateText(node:Node) {
    if(! node.isTerminal) {
      for(c <- node.children)
        propagateText(c)

      val sents = new ArrayBuffer[Sentence]()
      for(c <- node.children)
        for(s <- c.text.sentences)
          sents += s
      node.text = new Document(sents.toArray)
    }
  }

  def propagateLabels(node:Node) {
    if(node.isTerminal) {
      node.label = ""
    } else {
      var dir = ""
      var label = ""
      if(node.children(0).kind == "Nucleus" && node.children(1).kind == "Satellite") {
        dir = "LR"
        label = node.children(1).label
      } else if(node.children(1).kind == "Nucleus" && node.children(0).kind == "Satellite") {
        dir = "RL"
        label = node.children(0).label
      } else {
        assert(node.children(0).kind == node.children(1).kind)
        label = node.children(0).label
      }
      node.label = label + dir
      for(c <- node.children)
        propagateLabels(c)
    }
  }

  def parseDisTokens(tokens:Array[Token], offset:MutableNumber[Int]):Node = {
    consume("LP", tokens, offset)
    var t = consume("KIND", tokens, offset)
    val kind = t.value
    t = consume("SPAN|LEAF", tokens, offset)
    val isLeaf = t.kind == "LEAF"
    var label = ""
    if(kind != "Root") {
      t = consume("LABEL", tokens, offset)
      label = t.value
    }
    if(isLeaf) {
      t = consume("TEXT", tokens, offset)
      val text = t.value
      consume("RP", tokens, offset)
      new Node(kind, label, annotate(text), null)
    } else {
      var endOfNode = false
      val children = new ArrayBuffer[Node]()
      while(! endOfNode) {
        t = lookAhead(tokens, offset)
        // println(s"LOOKAHEAD: $t")
        if(t.kind == "RP") {
          endOfNode = true
        } else {
          val n = parseDisTokens(tokens, offset)
          // println("PARSED: " + n)
          children += n
        }
      }
      consume("RP", tokens, offset)
      val node = new Node(kind, label, null, children.toArray)
      if(children.size != 2) {
        throw new RuntimeException("ERROR: found tree with more than two children " + node)
      }
      node
    }
  }

  def consume(kind:String, tokens:Array[Token], offset:MutableNumber[Int]):Token = {
    if(offset.value >= tokens.length)
      throw new RuntimeException("ERROR: end of stream reached to soon!")

    if(Pattern.compile(kind, Pattern.CASE_INSENSITIVE).matcher(tokens(offset.value).kind).matches()) {
      val t = tokens(offset.value)
      offset.value += 1
      t
    } else {
      val v = tokens(offset.value).kind
      throw new RuntimeException(s"ERROR: expected $kind but seen $v at position $offset!")
    }
  }

  def lookAhead(tokens:Array[Token], offset:MutableNumber[Int]):Token = {
    if(offset.value >= tokens.length)
      throw new RuntimeException("ERROR: end of stream reached to soon!")
    tokens(offset.value)
  }

  def tokenizeDis(buffer:String):Array[Token] = {
    val tokens = new ArrayBuffer[Token]()
    var offset = 0
    while(offset < buffer.length) {
      offset = skipWhitespaces(buffer, offset)
      if(offset < buffer.length) {
        // println("TOKENIZING: " + buffer.substring(offset, Math.min(offset + 20, buffer.length)))
        var found = false
        for(pattern <- DiscReader.PATTERNS if ! found) {
          val m = pattern.pattern.matcher(buffer)
          if(matchesAt(m, offset)) {
            found = true
            var value = ""
            if(pattern.hasValue)
              value = buffer.substring(m.start(1), m.end(1))
            tokens += new Token(pattern.kind, value)
            offset = m.end()
          }
        }
        if(! found) {
          throw new RuntimeException("ERROR: cannot tokenize this text: " +
            buffer.substring(offset, Math.min(offset + 20, buffer.length)) +
            "...")
        }
      }
    }
    tokens.toArray
  }

  def matchesAt(m:Matcher, offset:Int):Boolean = {
    if(m.find(offset)) {
      if(m.start() == offset) return true
    }
    false
  }

  def skipWhitespaces(buffer:String, offset:Int):Int = {
    var of = offset
    while(of < buffer.length && Character.isWhitespace(buffer.charAt(of))) of += 1
    of
  }

  def annotate(text:String): Document = {
    val doc = proc.mkDocument(text)
    proc.tagPartsOfSpeech(doc)
    proc.lemmatize(doc)
    doc.clear()
    doc
  }
}

class Node (val kind:String,
            var label:String,
            var text:Document,
            val children:Array[Node],
            var isIntraSentence:Option[Boolean] = None) {
  def isTerminal:Boolean = children == null

  override def toString:String = {
    val os = new StringBuilder
    print(os, 0, true)
    os.toString()
  }
  def toString(printChildren:Boolean):String = {
    val os = new StringBuilder
    print(os, 0, printChildren)
    os.toString()
  }
  def print(os:StringBuilder, offset:Int, printChildren:Boolean) {
    var i = 0
    while(i < offset) {
      os.append(" ")
      i += 1
    }
    os.append(kind)
    if(label.length > 0) {
      os.append(":")
      os.append(label)
    }
    if(isIntraSentence.isDefined && isIntraSentence.get == true) {
      os.append("(is)")
    }

    if(text != null) {
      os.append(" TEXT:")
      if(isTerminal) {
        os.append(DiscReader.textToString(text, Int.MaxValue))
      } else {
        val words = getWords(text)
        var i:Int = 0
        val howMany = 3
        while(i < math.min(words.length, howMany)) {
          if(i > 0) os.append(" ")
          os.append(words(i))
          i += 1
        }
        os.append(" ...")
        i = math.max(words.length - howMany, 0)
        while(i < words.length) {
          os.append(" ")
          os.append(words(i))
          i += 1
        }
      }
    }

    if(printChildren) {
      os.append("\n")
      if(! isTerminal) {
        for(c <- children) c.print(os, offset + 2, printChildren)
      }
    }
  }

  def getWords(text:Document):Array[String] = {
    val wb = new ArrayBuffer[String]()
    for(s <- text.sentences) {
      for(w <- s.words) {
        wb += w
      }
    }
    wb.toArray
  }
}

class Token (val kind:String, val value:String) {
  override def toString:String = if(value.length > 0) kind + ":" + value else kind
}

class TokenPattern(val pattern:Pattern, val kind:String, val hasValue:Boolean)

object DiscReader {
  val logger = LoggerFactory.getLogger(classOf[DiscReader])

  val PATTERNS = Array[TokenPattern](
    new TokenPattern(Pattern.compile("\\(text\\s+_!(.+?)_!\\)", Pattern.CASE_INSENSITIVE), "TEXT", true),
    new TokenPattern(Pattern.compile("\\(rel2par\\s+([a-z\\-A-Z]*)\\)", Pattern.CASE_INSENSITIVE), "LABEL", true),
    new TokenPattern(Pattern.compile("\\(span\\s+[0-9]+\\s+[0-9]+\\)", Pattern.CASE_INSENSITIVE), "SPAN", false),
    new TokenPattern(Pattern.compile("\\(leaf\\s+[0-9]+\\)", Pattern.CASE_INSENSITIVE), "LEAF", false),
    new TokenPattern(Pattern.compile("\\(leaf\\s+[0-9]+\\)", Pattern.CASE_INSENSITIVE), "LEAF", false),
    new TokenPattern(Pattern.compile("(root)", Pattern.CASE_INSENSITIVE), "KIND", true),
    new TokenPattern(Pattern.compile("(nucleus)", Pattern.CASE_INSENSITIVE), "KIND", true),
    new TokenPattern(Pattern.compile("(satellite)", Pattern.CASE_INSENSITIVE), "KIND", true),
    new TokenPattern(Pattern.compile("\\(", Pattern.CASE_INSENSITIVE), "LP", false),
    new TokenPattern(Pattern.compile("\\)", Pattern.CASE_INSENSITIVE), "RP", false)
  )

  val EOS = Pattern.compile("[\\.|\\?|!]+", Pattern.CASE_INSENSITIVE)

  def textToString(text:Document, max:Int):String = {
    val os = new StringBuilder
    var first = true
    var count = 0
    for(s <- text.sentences if count < max) {
      for(w <- s.words if count < max) {
        if(! first) os.append(" ")
        os.append(w)
        first = false
        count += 1
      }
    }
    if(count == max) {
      os.append("...")
    }
    os.toString()
  }

  def main(args:Array[String]) {
    val reader = new DiscReader("")
    val top = new File(args(0))
    if(top.isDirectory) {
      for(f <- top.listFiles()){
        if(f.getName.endsWith(".dis")){
          println("Parsing file " + f)
          val p = reader.parseDis(f, verbose=true)
          println(p)
        }
      }
    } else {
      val p = reader.parseDis(top, verbose=true)
      println(p)
    }
  }
}
