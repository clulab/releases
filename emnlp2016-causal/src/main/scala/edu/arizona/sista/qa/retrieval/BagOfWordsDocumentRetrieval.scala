package edu.arizona.sista.qa.retrieval

import edu.arizona.sista.processors.{DocumentSerializer, Document}
import org.apache.lucene.analysis.WhitespaceAnalyzer
import org.apache.lucene.queryParser.QueryParser
import org.apache.lucene.util.Version
import edu.arizona.sista.qa.index.{Indexer, TermFilter}
import org.apache.lucene.store.SimpleFSDirectory
import java.io.File
import org.apache.lucene.index.IndexReader
import org.apache.lucene.search.{TopScoreDocCollector, IndexSearcher}
import collection.mutable.ListBuffer
import edu.arizona.sista.utils.StringUtils

/**
 * 
 * User: mihais
 * Date: 3/15/13
 */
class BagOfWordsDocumentRetrieval(
  val indexDir:String,
  val maxDocCount:Int,
  val termFilter:TermFilter,
  val question:Document,
  val useWords:Boolean = false) extends DocumentRetrieval {
  // useWords: (false = use lemmas for query, true = use words for query)

  lazy val documentSerializer = new DocumentSerializer

  val luceneEscapeChars:String = "[\\\\\\+\\-\\!\\(\\)\\:\\^\\]\\{\\}\\~\\*\\?]"

  var query:String = ""

  /** We use the white-space analyzer because we will pre-process the query using our own tools */
  var queryParser:QueryParser = null

  // Determine retrieval mode (using words or lemmas), and setup query variables
  if (useWords) {
    throw new RuntimeException (" BagOfWordsDocumentRetrieval: Word-based retrieval not currently supported.  Use lemma mode instead. ")
    /*
    // Use Words
    queryParser = new QueryParser(
      Indexer.VERSION,
      Indexer.TEXT,
      new WhitespaceAnalyzer)
    query = termFilter.extractValidWords(question).mkString(" ")
    */
  } else {
    // Default: Use lemmas
    queryParser = new QueryParser(
      Indexer.VERSION,
      //Indexer.LEMMAS,
      Indexer.TEXT,               // temporary -- index was made with lemmas in text field
      new WhitespaceAnalyzer)
    query = termFilter.extractValidLemmas(question).mkString(" ")
  }




  def retrieve:List[DocumentCandidate] = {
    if (query.length == 0) {
      println("Found empty query. Bailing out...")
      return List()
    }

    if (useWords) {
      println("Using IR (word) query: " + query)
    } else {
      println("Using IR (lemma) query: " + query)
    }

    // open/close index for every query to avoid
    // leaving junk after ourselves in tomcat
    val dir = new SimpleFSDirectory(new File(indexDir));
    val indexReader = IndexReader.open(dir);
    val indexSearcher = new IndexSearcher(indexReader);
    println("Opened index with " + indexReader.numDocs() + " documents.")

    val collector = TopScoreDocCollector.create(maxDocCount, true)
    val escapedQuery = query.replaceAll(luceneEscapeChars, "\\\\$0")
    val luceneQuery = queryParser.parse(escapedQuery)
    indexSearcher.search(luceneQuery, collector)
    val hits = collector.topDocs().scoreDocs;
    println("Found " + hits.length + " hits.")

    var i = 0
    val docBuffer = new ListBuffer[DocumentCandidate]
    while(i < hits.length) {
      val doc = indexSearcher.doc(hits(i).doc);
      val score = hits(i).score
      val docid = doc.getField(Indexer.DOCID).stringValue()
      val annotation = documentSerializer.load(doc.getField(Indexer.ANNOTATION).stringValue())
      val pars = StringUtils.toIntArray(doc.getField(Indexer.PARAGRAPH_IDS).stringValue())

      //println("Reading document " + docid)
      //println("Paragraphs: " + pars.mkString(" "))

      docBuffer += new DocumentCandidate(docid, annotation, pars, score)
      i += 1
    }

    // now close the index
    indexSearcher.close() // TODO: Required for Lucene 3.x. This appears to not be needed in 4.x?
    indexReader.close()

    docBuffer.toList
  }
}
