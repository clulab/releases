
import java.io.{File, FileWriter, PrintWriter}
import java.nio.file.Paths

import com.typesafe.config.ConfigFactory

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.io.Source
import org.apache.lucene.analysis.Analyzer
import org.apache.lucene.analysis.core.WhitespaceAnalyzer
import org.apache.lucene.analysis.standard.StandardAnalyzer
import org.apache.lucene.document.{Document, Field}
import org.apache.lucene.index.{DirectoryReader, IndexWriter, IndexWriterConfig, Term}
import org.apache.lucene.queryparser.classic.QueryParser
import org.apache.lucene.search._
import org.apache.lucene.store.{FSDirectory, RAMDirectory}
import ir.BM25Indexer

object BM25Searcher extends App {
  // Create an in-memory directory
  // Create a searcher and get some results
  //val config = if (args.isEmpty) ConfigFactory.load()
  //else ConfigFactory.parseFile(new File(args(0))).resolve()
  val bm25Indexer = new BM25Indexer()

  val config = ConfigFactory.load()
  val projectRootPath =Paths.get("").toAbsolutePath.toString
  val dataset = config.getString("dataset")
  val indexDir = projectRootPath+"/data/"+ config.getString(s"${dataset}.indexDir")

  println(indexDir)
  val reader = DirectoryReader.open(FSDirectory.open(Paths.get(indexDir)))

  val searcher = new IndexSearcher(reader)
  //val resultDir = config.getString("wiki.resultDirTest")

  val TOTAL_HITS=config.getInt(s"${dataset}.totalHits")//create an array of length 1400 for each query to store the doc scores and names
  //val query = """ | Aristotle OR | Aristotle's OR | Orwell | """
  //val query = scala.io.Source.fromFile(config.getString("wiki.queryTest")).mkString
  val queryDevDir = new File(projectRootPath+"/data/"+ config.getString(s"${dataset}.devQueryDir"))
  val queryTestDir = new File(projectRootPath+"/data/"+ config.getString(s"${dataset}.testQueryDir"))

  val labelDevDir = new File(projectRootPath+"/data/"+ config.getString(s"${dataset}.devLabelDir"))
  val labelTestDir = new File(projectRootPath+"/data/"+ config.getString(s"${dataset}.testLabelDir"))

  val resultDevDir = projectRootPath+"/output_scores/"+ config.getString(s"${dataset}.devResult")
  val resultTestDir = projectRootPath+"/output_scores/"+ config.getString(s"${dataset}.testResult")

  val analyzer = new StandardAnalyzer()

  // TODO: load article for debugging purpose, delete this later.
//  val kbDir = new File(projectRootPath+"/data/"+ config.getString(s"${dataset}.kbDir"))
//  val text = scala.io.Source.fromFile(kbDir).mkString
//  val articles = {
//    if(dataset=="openbook") {text.split("\n").filter(_ != "")}
//    else {text.split(" DOC_SEP\n ").filter(_ != "")} // I don't use brackets because brackets causes problems in regex match.
//  }

  val searchTime = mutable.ArrayBuffer[Double]()
  val writeTime = mutable.ArrayBuffer[Double]()
  val totalTime = mutable.ArrayBuffer[Double]()

  def vanillaUseCase(queryDir:File, labelDir:File, resultDir:String, maxDocs:Int = TOTAL_HITS) {

    val allQueries = {
      if (dataset=="openbook"){
        scala.io.Source.fromFile(queryDir).mkString.split("\n").filter(_ != "")
      }else{scala.io.Source.fromFile(queryDir).mkString.split(" QUERY_SEP\n ").filter(_ != "")}
    }

    // run 20 dummy queries to make lucene
    for (queryIdx <- 0 until 20){
      val query = {
        if (dataset=="openbook"){
          bm25Indexer.rawTextToLemmas(allQueries(queryIdx)).mkString(" ")
        }else{
          allQueries(queryIdx).replaceAll("[+\\-&|!(){} \\[ \\]^\"~*?:\\\\\\/\\']", " ")
        }
      }
      val q = new QueryParser("content", analyzer).parse(query)
      val collector = TopScoreDocCollector.create(2000) // This is the original collector, which automatically sort the docs
      searcher.search(q, collector) // This is the default method. This is correct because search function returns void.
      val hits = collector.topDocs().scoreDocs //
    }

    val allLabels = scala.io.Source.fromFile(labelDir).mkString.split("\n").filter(_ != "")

    var query_count = 0

    val mrrList = mutable.ArrayBuffer[Float]()
    for(queryIdx <-allQueries.indices){
      if ((queryIdx+1)%100==0){
        println(s"==========================")
        println(s"\tprocessing query ${queryIdx+1} ... , MRR: ${mrrList.toList.sum/mrrList.length.toFloat}")
        println(s"\taverage search time: ${searchTime.toList.sum/searchTime.length.toFloat}")
        println(s"\taverage write time: ${writeTime.toList.sum/writeTime.length.toFloat}")
        println(s"\taverage total time: ${totalTime.toList.sum/totalTime.length.toFloat}")

      }

      val ( eventDocs, allDocList) = search(allQueries(queryIdx))
      val goldLabel = allLabels(queryIdx).toInt
      val mrr = if (allDocList.contains(goldLabel)){1F/(1F+allDocList.indexOf(goldLabel))} else {0F}
      //println(s"The result contains ${eventDocs.size} documents.")

      saveDocs(resultDir, eventDocs, queryIdx, mrr, maxDocs)  // TODO: this needs to be rewritten later.

      query_count+=1
      mrrList.append(mrr)

//      println("=============")
//      println(query)
//      println(goldLabel, mrr)
//      println(allDocList.slice(0,10))
//      println("\ngold article:", articles(goldLabel))
//      println("\ntop article:", articles(allDocList(0)))
//      scala.io.StdIn.readLine()
    }
    println("==================================================")
    println("MRR:", mrrList.toList.sum/mrrList.length.toFloat)
    println(s"\taverage search sort time: ${searchTime.toList.sum/searchTime.length.toFloat}")
    println(s"\taverage write time: ${writeTime.toList.sum/writeTime.length.toFloat}")
    println(s"\taverage total time: ${totalTime.toList.sum/totalTime.length.toFloat}")
    println(s"Done! Processed ${query_count} queries!")
    println("==================================================")

  }

  def search(query:String, totalHits:Int = TOTAL_HITS):(Seq[(Int, Float)], Seq[Int]) = {
    searchByField(query, "content", analyzer, totalHits)
  }

  def searchByField(queryRaw:String,
                    field:String,
                    analyzer:Analyzer,
                    totalHits:Int = TOTAL_HITS,
                    verbose:Boolean = false):(Seq[(Int, Float)], Seq[Int]) = {

    val t0 = System.nanoTime().toDouble

    // pre-process the query. Use lemmatization is needed.
    val query = {
      if (dataset=="openbook"){
        bm25Indexer.rawTextToLemmas(queryRaw).mkString(" ")
      }else{
        queryRaw.replaceAll("[+\\-&|!(){} \\[ \\]^\"~*?:\\\\\\/\\']", " ")
      }
    }
    val q = new QueryParser(field, analyzer).parse(query)
    val collector = TopScoreDocCollector.create(totalHits) // This is the original collector, which automatically sort the docs
    searcher.search(q, collector) // This is the default method. This is correct because search function returns void.
    val hits = collector.topDocs().scoreDocs //


    // get all matched docs without sorting.
//    val allDocs = searcher.search(q, totalHits, Sort.INDEXORDER, true, false)
//    val hits = allDocs.scoreDocs
    val t1 = System.nanoTime().toDouble
    searchTime.append((t1-t0)*1e-9)

    val results = new mutable.ArrayBuffer[(Int, Float)]()

    for(hit <- hits) {
      val docId = hit.doc
      val score = hit.score
      results.append((docId, score))

    }
    //val allDocList = results.sortBy(-_._2).map{case (x,y) => x}  //sort in descending order of scores.

    val allDocList = results.map{case (x,y) => x}

    if(verbose) println(s"""Found ${results.size} results for query "$query"""")

    (results, allDocList)
  }

  def saveDocs(resultDir:String, docIds:Seq[(Int, Float)], queryCount:Int, mrr:Float, maxDocs:Int): Unit = {
    val t2 = System.nanoTime().toDouble


    val sos = new PrintWriter(new FileWriter(resultDir + File.separator + "query_"+queryCount.toString +"_scores.tsv"))
    var count = 0


    for(docId <- docIds if count < maxDocs) {

      sos.println(s"${docId._1}\t${docId._2}")
      count += 1

    }

    val t3 = System.nanoTime().toDouble
    writeTime.append((t3-t2)*1e-9)
    totalTime.append(searchTime.last+writeTime.last)

    sos.println(s"=================================")
    sos.println(s"MRR\tSearchSortTime\tWriteTime\tTotalTime")
    sos.println(s"${mrr}\t${searchTime(queryCount)}\t${writeTime(queryCount)}\t${totalTime(queryCount)}")

    sos.close()
  }

  if (dataset=="nq"){
    vanillaUseCase(queryTestDir, labelTestDir, resultTestDir)
  }
  else{
    vanillaUseCase(queryDevDir, labelDevDir, resultDevDir)
    vanillaUseCase(queryTestDir, labelTestDir, resultTestDir)
  }

  reader.close()
}



