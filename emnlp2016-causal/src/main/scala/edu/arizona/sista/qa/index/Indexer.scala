package edu.arizona.sista.qa.index

import edu.arizona.sista.qa.index.Indexer._
import edu.arizona.sista.utils.StringUtils._
import edu.arizona.sista.processors.corenlp.CoreNLPProcessor
import edu.arizona.sista.processors.DocumentSerializer
import org.apache.lucene.analysis.WhitespaceAnalyzer

import org.apache.lucene.util.Version
import collection.mutable.ArrayBuffer
import org.apache.lucene.index.IndexWriter
import org.apache.lucene.store.SimpleFSDirectory
import java.io.File
import org.apache.lucene.document.Field
import edu.arizona.sista.utils.StringUtils
import org.slf4j.LoggerFactory
import edu.arizona.sista.qa.scorer.Question


/**
 * Indexes a collection of documents in our XML format
 * User: mihais
 * Date: 2/11/13
 */
class Indexer (val collectionDir:String, val indexDir:String) {
  lazy val processor = new CoreNLPProcessor()
  lazy val serializer = new DocumentSerializer
  lazy val termFilter = new TermFilter
  lazy val depBuilder = new DependencyBuilder(termFilter)
  lazy val indexer = new IndexWriter(
    new SimpleFSDirectory(new File(indexDir)),
    // new IndexWriterConfig(Indexer.VERSION, new WhitespaceAnalyzer(Indexer.VERSION)))   // Lucene 4.x
    new WhitespaceAnalyzer, true,
    IndexWriter.MaxFieldLength.LIMITED)

  def index(dryrun:Boolean = false, partialAnnotation:Boolean = false) {
    val docIterator = new DocumentIterator(collectionDir)
    var docCount = 0
    while(docIterator.hasNext) {
      val doc = docIterator.next()
      addToIndex(doc, dryrun, partialAnnotation)

      docCount += 1
      //if (docCount % 100 == 0){
      if (docCount % 1 == 0){
        println("Indexed " + docCount + " documents...")
        //println("Internalizer stats:")
        //Processor.in.stats()
      }
    }
    println("Added " + docCount + " documents to the index.")
    if (! dryrun) {
      println("Closing index...")
      // Forces merging of segments (optimize search process at the cost of longer index)
      indexer.optimize()            // Lucene 3.x
      //indexer.forceMerge(1)       // Lucene 4.x
      indexer.close()
    }
    println("Done.")

  }

  def indexFromGoldAnswers(qaPairs:Array[Question], dryrun:Boolean = false, partialAnnotation:Boolean = false) {
    var docCount = 0
    logger.info("Will index " + qaPairs.length + " questions.")

    for (question <- qaPairs) {
      for (ga <- question.goldAnswers) {

        // Place GA into Indexer Document structure
        val answerAnnotation = mkPartialAnnotation(ga.text)
        val sentences = new ArrayBuffer[Sentence]()
        for (i <- 0 until answerAnnotation.sentences.size) {
          val sent = new Sentence(answerAnnotation.sentences(i).getSentenceText(), i+1, 1)
          sentences.append(sent)
        }
        val doc = new Document(ga.docid, sentences.toList)

        // Note: The original text is passed in because there are infrequent formatting issues that cause an extra sentence when re-assembling the text -- I think from processing URLs.
        addToIndex(doc, dryrun, partialAnnotation, ga.text)

        docCount += 1
        if (docCount % 10 == 0) {
          println("Indexed " + docCount + " documents...")
          //println("Internalizer stats:")
          //Processor.in.stats()
        }
      }
    }

    println("Added " + docCount + " documents to the index.")
    if (! dryrun) {
      println("Closing index...")
      // Forces merging of segments (optimize search process at the cost of longer index)
      indexer.optimize()           // Lucene 3.x
      //indexer.forceMerge(1)      // Lucene 4.x
      indexer.close()
    }
    println("Done.")

  }


  def addToIndex(doc:Document, dryrun:Boolean, partialAnnotation:Boolean = false, originalText:String = "") {
    // we store in the index: the docid; the actual text; the NLP annotations; the paragraph ids
    // if text is passed in "originalText", it will use that for annotation instead of the sentences in "doc".  this is
    //   centrally for CQA mode to deal with formatting issues.
    val sents = getSentences(doc)
    val sids = getSentenceIds(doc)
    val pids = getParagraphIds(doc)

    // make sure the sentence ids are position in the sents array + 1
    assert(sents.length == sids.length)
    assert(sents.length == pids.length)
    for (i <- 0 to sids.length - 1) {
      assert(i + 1 == sids(i))
    }

    val docid = doc.docid
    var annotation:edu.arizona.sista.processors.Document = null
    if (partialAnnotation) {
      if (originalText != "") {
        annotation = mkPartialAnnotation(originalText)
      } else {
        annotation = mkPartialAnnotationFromSentences(sents)
      }
    } else {
      if (originalText != "") {
        annotation = processor.annotate(originalText)
      } else {
        annotation = processor.annotateFromSentences(sents)
      }
    }
    assert(annotation.sentences.length == pids.length)

    val annotationAsString = serializer.save(annotation)
    val bow = getNonStopWords(annotation)
    val bol = getNonStopLemmas(annotation)
    val bod = getDependencies(annotation)
    val pidsAsString = pids.mkString(" ")

    if (dryrun) {
      println("DOCID: " + docid)
      println("TEXT: " + bow)
      println("LEMMAS: " + bol)
      println("PIDS: " + pidsAsString)
      println("ANNOTATION: " + annotationAsString)
    } else {
      val doc = new org.apache.lucene.document.Document
      doc.add(new Field(DOCID, docid, Field.Store.YES, Field.Index.NOT_ANALYZED))
      doc.add(new Field(PARAGRAPH_IDS, pidsAsString, Field.Store.YES, Field.Index.NOT_ANALYZED))
      doc.add(new Field(TEXT, bow, Field.Store.YES, Field.Index.ANALYZED))
      doc.add(new Field(LEMMAS, bol, Field.Store.YES, Field.Index.ANALYZED))
      doc.add(new Field(DEPENDENCIES, bod, Field.Store.YES, Field.Index.ANALYZED))
      doc.add(new Field(ANNOTATION, annotationAsString, Field.Store.YES, Field.Index.NOT_ANALYZED))
      indexer.addDocument(doc)

      //logger.debug("TEXT: " + bow)
      //logger.debug("DEPS: " + bod)
    }

  }

  private def getNonStopWords(doc:edu.arizona.sista.processors.Document): String = {
    termFilter.extractValidWords(doc, 0, doc.sentences.length).mkString(" ")
  }

  private def getNonStopLemmas(doc:edu.arizona.sista.processors.Document): String = {
    termFilter.extractValidLemmas(doc, 0, doc.sentences.length).mkString(" ")
  }

  private def getDependencies(doc:edu.arizona.sista.processors.Document): String = {
    depBuilder.buildDependencies(doc).mkString(" ")
  }

  private def getSentences(doc:Document): Array[String] = {
    val sents = new ArrayBuffer[String]
    for(s <- doc.sentences) sents += s.text
    sents.toArray
  }

  private def getSentenceIds(doc:Document): Array[Int] = {
    val sents = new ArrayBuffer[Int]
    for(s <- doc.sentences) sents += s.offset
    sents.toArray
  }

  private def getParagraphIds(doc:Document): Array[Int] = {
    val ids = new ArrayBuffer[Int]
    for(s <- doc.sentences) ids += s.paragraph
    ids.toArray
  }

  def mkPartialAnnotation(text:String):edu.arizona.sista.processors.Document = {
    val doc = processor.mkDocument(text)
    mkPartialAnnotationFromDoc(doc)
  }

  def mkPartialAnnotationFromSentences(sents:Array[String]):edu.arizona.sista.processors.Document = {
    val doc = processor.mkDocumentFromSentences(sents)
    mkPartialAnnotationFromDoc(doc)
  }

  def mkPartialAnnotationFromDoc(doc:edu.arizona.sista.processors.Document):edu.arizona.sista.processors.Document = {
    processor.tagPartsOfSpeech(doc)
    processor.lemmatize(doc)
    doc.clear
    doc
  }


}

object Indexer {
  val TEXT = "text"
  val LEMMAS = "lemmas"
  val DEPENDENCIES = "deps"
  val DOCID = "docid"
  val ANNOTATION = "annotation"
  val PARAGRAPH_IDS = "pids"

  // TODO: we keep this as Lucene 3.0, for backwards compatibility. At some point, change to 4.x
  val VERSION = Version.LUCENE_30
  //val VERSION = Version.LUCENE_42

  val logger = LoggerFactory.getLogger(classOf[Indexer])

  def main(args: Array[String]) {
    val props = argsToProperties(args)
    val collection = props.getProperty("collection")
    val index = props.getProperty("index")
    val dryrun = StringUtils.getBool(props, "index.dryrun", false)
    val partialAnnotation = StringUtils.getBool(props, "index.partialannotation", false)

    val indexer = new Indexer(collection, index)
    indexer.index(dryrun, partialAnnotation)
  }
}
