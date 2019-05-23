package org.clulab.clint

import com.typesafe.scalalogging.LazyLogging
import org.clulab.embeddings.word2vec.Word2Vec
import java.io._
import java.util.HashMap
import com.typesafe.config.ConfigFactory
import ai.lum.common.ConfigUtils._
import scala.io._

/**
 * Need sanitized words/phrases and their vectors
 */
//val lines = Source.fromFile("t1").getLines.toArray
//val bw = new FileWriter(new File("t1.sanitized"))
//for(l <- lines) {
//  bw.write(s"${Word2Vec.sanitizeWord(l.replace(" ", "_"))}\n")
//}
//bw.close

object kNNClassifierIndex extends App with LazyLogging {

  val config = ConfigFactory.load()
  //  val datadir = config[String]("clint.data-dir")

  val entitiesFile: String = config[String]("clint.goldLabelsFile")
  // datadir + "/conll.goldlabels"
  val w2vfile = config[String]("clint.w2vVectors")
  // datadir + "/vectors.txt"
  val knn_offline_list_file = config[String]("clint.knnFile")
  //datadir + "/knn_offline_list.txt"
  val indexDir: File = config[File]("clint.index-dir")
  val entityLexicon = LexiconBuilder.loadLexicon(new File(indexDir, "entity.lexicon"))


  val entitiesSet = Source.fromFile(entitiesFile).getLines.map {
    l =>
      val tmp = l.split("\t")
      val k = tmp(0)
      val v = tmp.tail
      k -> v
  }.toMap.unzip._2.flatten.toSet

  val sep = "-" * 70

  val w2v = new Word2Vec(w2vfile, None)

  val pb = new me.tongfei.progressbar.ProgressBar(s"entities", entitiesSet.size)
  pb.start

  val entitySimMap = new HashMap[(String, String), Double]()
  val dataToWrite = for (e1: String <- entitiesSet.toArray) yield {
    val e1Tokenised = e1.split(" +").map { e => Word2Vec.sanitizeWord(e) }

    val e2SimSorted = (for (e2: String <- entitiesSet - e1) yield {
      val e2Tokenised = e2.split(" +").map { e => Word2Vec.sanitizeWord(e) }
      //      val sim = w2v.sanitizedTextSimilarity(e1Tokenised, e2Tokenised)
      val sim = w2v.similarity(Word2Vec.sanitizeWord(e1.replace(" ", "_")), Word2Vec.sanitizeWord(e2.replace(" ", "_")))
      //        bw.write(s"$e1\t$lbl1\t$e2\t$lbl2\t$sim\n")
      entitySimMap.put((e1, e2), sim)
      (e2, sim)
    }).toArray.sortBy(-_._2)

    //    bw.write(s"${e1}\t${e2SimSorted.mkString(" ;; ")}\n")
    pb.step
    (e1, e2SimSorted)
  }

  logger.info(s"Writing data to file ${knn_offline_list_file}")
  val bw = new BufferedWriter(new FileWriter(new File(knn_offline_list_file)))
  for (d <- dataToWrite) {
    val (e1, e2SimSorted) = (d._1, d._2)
    bw.write(s"${e1} @@ ${entityLexicon.inverseLexicon(e1)}\t${e2SimSorted.map(x => s"${x._1} ## ${entityLexicon.inverseLexicon(x._1)} ::: ${x._2}").mkString(" ;; ")}\n")
  }

  bw.close
  pb.stop
}
//    val missingEntites = (for(e <- entities.par) yield {
//      pb.step
//      val closeByW2Vwords = w2v.mostSimilarWords(Word2Vec.sanitizeWord(e), 1000) // TODO: Make this a parameter
////      logger.info(sep)
////      logger.info(s"Entity $e")
////      logger.info(sep)
//      closeByW2Vwords.size
//      val closeByEntities = for((cbw,score) <- closeByW2Vwords) yield {
//        val cbes = entities.map (_.toLowerCase ).filter { x => x.contains(cbw.toLowerCase) }
//        (cbw, cbes, score)
//      }
//      
//      if(closeByEntities.size > 0) {
//        for((cbw, cbes, score) <- closeByEntities){
//  //        logger.info(s"$e\t${cbw}\t${score}\t${cbes.mkString(":: ")}")
//          bw.write((s"$e\t$cat\t${cbw}\t${score}\t${cbes.mkString(":: ")}\n"))
//        }
//      }
//      else {
//        bw.write(s"$e\tNIL\tNIL\tNIL\n")
//      }
//      
////      logger.info(sep)
//      
//      if(closeByW2Vwords.size == 0) 1 else 0
//      
//    }).sum
//    pb.stop
//    logger.info(s"Number of missing entities in $cat : $missingEntites")
  
  
//  val x = labelToEntitesMap("PER").head
//  println(s"Words most similar to ${x}: ")
//  for(t <- w2v.mostSimilarWords(x.split(" +").toSet, 40)) {
//    println(t._1 + " " + t._2)
//  }
