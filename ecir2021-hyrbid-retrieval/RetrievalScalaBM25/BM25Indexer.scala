package ir


import java.io.File
import java.nio.file.Paths

import com.typesafe.config.{Config, ConfigFactory}
import org.apache.lucene.analysis.standard.StandardAnalyzer
import org.apache.lucene.document
import org.apache.lucene.document.Field
import org.apache.lucene.document.StringField
import org.apache.lucene.document.TextField
import org.apache.lucene.store.FSDirectory
import org.apache.lucene.store.NIOFSDirectory
import org.apache.lucene.store.SimpleFSDirectory
import org.apache.lucene.index.IndexReader
import org.apache.lucene.search.IndexSearcher
import org.apache.lucene.index.IndexWriter
import org.apache.lucene.index.IndexWriterConfig
import org.apache.lucene.index.DirectoryReader
import org.apache.lucene.queryparser.classic.QueryParser
import edu.stanford.nlp.simple.Sentence
import org.apache.lucene.analysis.core.WhitespaceAnalyzer

import scala.collection.JavaConverters._

class BM25Indexer {
  val stopWords = List("i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
  "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its",
  "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom",
  "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
  "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but",
  "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about",
  "against", "between", "into", "through", "during", "before", "after", "above", "below", "to",
  "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then",
  "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few",
  "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
  "too", "very", "s", "t", "can", "will", "just", "don", "should", "now")

  def rawTextToLemmas(rawText:String):List[String]={
    val cleanedText = rawText.replaceAll("[+\\-&|!(){} \\[ \\]^\"~*?:\\\\\\/\\']", " ")
    val cleanedTextObj = new Sentence(cleanedText.toLowerCase())
    val cleanedTextLemmas = cleanedTextObj.lemmas().asScala.toList

    cleanedTextLemmas
  }
}

// Index the wikipedia with Lucene
object BM25IndexBuilder extends App{
  val bm25Indexer = new BM25Indexer()

  val config =  ConfigFactory.load()
  val projectRootPath =Paths.get("").toAbsolutePath.toString

  val dataset = config.getString("dataset")

  val kbDir = new File(projectRootPath+"/data/"+ config.getString(s"${dataset}.kbDir"))
  val indexDir = new File(projectRootPath+"/data/"+ config.getString(s"${dataset}.indexDir"))

  val index = FSDirectory.open(indexDir.toPath)

  // Analyzer for reading and writting
  def analyzer = new StandardAnalyzer()
  //def analyzer = new WhitespaceAnalyzer  // use white space analyzer together with lemmatizer.

  // Create the index writer
  val indexWriter = {
    //val index = FSDirectory.open(indexDir.toPath)
    val index = FSDirectory.open(indexDir.toPath)
    val indexConfig = new IndexWriterConfig(analyzer)

    new IndexWriter(index, indexConfig)
  }

  val text = scala.io.Source.fromFile(kbDir).mkString
  //val articles = text.split("</?doc.*>").filter(_ != "")

  val articles = {
    if(dataset=="openbook") {text.split("\n").filter(_ != "")}
    else {text.split(" DOC_SEP\n ").filter(_ != "")} // I don't use brackets because brackets causes problems in regex match.
  } // Use DOC_SEP because \n is not a very good indicator of the actual separation of newline.

  var article_count = 0
  for(article <- articles){
    val doc = mkDocument(article, article_count)
    // Add to the index
    indexWriter.addDocument(doc)
    article_count+=1
  }
  println(s" Article count: ${article_count}")

  indexWriter.close

  def mkDocument(text:String, articleCount:Int):document.Document = {
    val doc = new document.Document()

    if (dataset=="openbook"){
      val text_normalized = bm25Indexer.rawTextToLemmas(text).mkString(" ")
      doc.add(new TextField("content", text_normalized, Field.Store.YES))
      doc.add(new TextField("articleCount", articleCount.toString, Field.Store.YES))

      doc
    }
    else{
      doc.add(new TextField("content", text, Field.Store.YES))
      doc.add(new TextField("articleCount", articleCount.toString, Field.Store.YES))

      doc
    }

  }


}

