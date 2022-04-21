package controllers

import java.io.PrintWriter
import java.text.SimpleDateFormat
import java.util.Date
import scala.util.Try

import javax.inject._
import play.api.mvc._
import play.api.libs.json._
import scala.util.control.Breaks._

import ai.lum.common.ConfigFactory
import org.clulab.odinsynth.{CorpusReader, Searcher, Spec, Query}
import org.clulab.odinsynth.scorer.StaticWeightScorer
import ai.lum.odinson.{Document => OdinsonDocument}

import org.clulab.processors.clu.CluProcessor
import _root_.org.clulab.processors.{Document => CluDocument}
import org.clulab.processors.fastnlp.FastNLPProcessor
import org.clulab.dynet.DyNetSync
import scala.collection.mutable
import edu.cmu.dynet.Initialize
import org.clulab.odinsynth.evaluation.DocumentFromSentences
import ai.lum.odinson.ExtractorEngine
import org.clulab.odinsynth.Parser
import scala.concurrent.Future
import org.clulab.dynet.Utils.initializeDyNet

import java.io.File
import java.nio.file.Files
import java.nio.file.Paths
import scala.concurrent.ExecutionContext.Implicits.global
import scala.collection.SeqLike
import java.io.File

// sbt -J-Xmx16g "webapp/run 9001"
@Singleton
class SearcherController  @Inject() (cc: ControllerComponents) extends AbstractController(cc) {

  private lazy val odinsonDocumentCache = {
    val path = f"${System.getProperty("user.home")}/.cache/odinsynth_cache"
    val f = new File(path)
    if (!f.exists()) {
      f.mkdirs()
    }
    path
    // "/home/rvacareanu/projects/odinsynth_cache3/"
  }



  // FIXME This is used for solution checking: (1) Check in the searcher; (2) remove usage
  lazy val p = {
    initializeDyNet()
    new FastNLPProcessor
  }
  
  // curl -X POST -H "Content-Type: application/json" -d '{"query": "[word=this] [word=is]", "sentences": [["this", "is", "a", "test"], ["this", "is", "a", "new", "test"]], "specs": [{"sentId": 0, "start": 0, "end": 2}, {"sentId": 1, "start": 0, "end": 2}] }' 127.0.0.1:9000/isSolution
  def isSolution() = Action.async { request =>
    val jValue = request.body.asJson.get
    val query: String = (jValue \ "query").as[JsString].value
    val sentences = (jValue \ "sentences").as[JsArray].value.to[Seq].map { it => it.as[JsArray].value.to[Seq].map(_.as[String]) }
    
    val specs     = (jValue \ "specs").as[JsArray].value.to[Seq].flatMap { it => 
      val sentenceJs = (it \ "sentId")
      val startJs = (it \ "start")
      val endJs   = (it \ "end")
      if (sentenceJs.isDefined && startJs.isDefined && endJs.isDefined) {
        Some((sentenceJs.as[Int], startJs.as[Int], endJs.as[Int]))
      } else {
        None
      }
    }

    val sentencesAndSpecs = sentences.zip(specs.sortBy(_._1))

    // val sentencesAndSpecs = (jValue \ "sentences_specs").as[JsArray].value.to[Seq].map { it => 
    //   val sentence = (it \ "sentence").as[String].split(" ")
    //   val startJs = (it \ "start")
    //   val endJs   = (it \ "end")
    //   val startEnd = if (startJs.isDefined && endJs.isDefined) {
    //     Some((startJs.as[Int], endJs.as[Int]))
    //   } else {
    //     None
    //   }
    //   (sentence.toSeq, startEnd)
    // }
    // val sentences = sentencesAndSpecs.map(_._1)

    Future {
      val doc = DocumentFromSentences.documentFromSentencesAndCache(sentences, p, odinsonDocumentCache)
      // println(jValue.toString())
      val specs = sentencesAndSpecs.zipWithIndex.map { case ((_, startEnd), index) =>
          Spec(doc.id, startEnd._1, startEnd._2, startEnd._3)
      }.toSet
// 
      val ee  = ExtractorEngine.inMemory(doc)
// 
      // The searcher, this time, is not used for searching but only for the "isSolution" check
      // TODO refactor
      val searcher = new Searcher(docs = Seq(doc), specs = specs, fieldNames = Set.empty[String], maxSteps = None, writer = None, scorer = StaticWeightScorer.apply(), withReward = false)
      val q           = Parser.parseBasicQuery(f"""${query}""")
      val partialQ    = q.getValidQuery.map(_.pattern)
      val isSolution  = searcher.isSolution(q)
      val reward      = searcher.rewardAsF1(q) // Partial reward
      val compromised = searcher.heuristicsCheck(q).isEmpty // If the heuristics tell us that you cannot reach a solution, this search is compromised

      (searcher.isSolution(q), reward, partialQ, compromised)
    }.map { case (isSolution, partialReward, partialQ, compromised) => 
      // println("We're in the map now")
      val partialQuery = partialQ.getOrElse("")
      
      val result = Json.obj(
        "solution"       -> isSolution,
        "partial_reward" -> math.abs(partialReward).toFloat,
        "compromised"    -> compromised,
        "partial_query"  -> partialQuery,
      )
      Ok(result)
    }
  }

  def test() = Action.async { request => Future { Ok(Json.parse("""{"status": "ok"}""")) } }

}

object ThisIsATemporaryTest extends App {
  import org.clulab.odinsynth.holeGlyph
  // val q  = f"[word=this] ${holeGlyph}"
  // val q  = """((([lemma=south])+) (□))+"""
  val q  = s"""(□) (□)"""//.replace("\"", "")
  val pq = Parser.parseBasicQuery(q) 
  println(pq)
  val sentencesAndSpecs: Seq[(Seq[String], Option[(Int, Int)])] = Seq(
    (Seq("It", "explores", "the", "combination", "of", "distinct", "musical", "materials", ":", "pentatonic", ",", "diatonic", "and", "chromatic", ",", "which", "are", "juxtaposed", "in", "different", ",", "and", "sometimes", "disconnected", "time", "contexts", ",", "always", "in", "search", "of", "suggestive", "sonorities", "."), Some((5, 7))),
    (Seq("Of", "the", "11", "radio", "frequencies", "available", "for", "use", "only", "3", "of", "them", "are", "distinct", "enough", "to", "not", "cause", "interference", "when", "wireless", "devices", "are", "in", "close", "proximity", "."), Some((13, 15))),
    (Seq("The", "plants", "later", "assume", "a", "distinct", "yellow", "color", "."), Some((5, 7))),
  )

  private lazy val odinsonDocumentCache = "/home/rvacareanu/projects/odinsynth_cache3/"

  val sentences = sentencesAndSpecs.unzip._1
  val p = {
    initializeDyNet()
    new FastNLPProcessor
  }
  val doc = DocumentFromSentences.documentFromSentencesAndCache(sentences, p, odinsonDocumentCache)
  val specs = sentencesAndSpecs.zipWithIndex.flatMap { case ((_, startEndOpt), index) =>
    if(startEndOpt.isDefined) {
      Some(Spec(doc.id, index, startEndOpt.get._1, startEndOpt.get._2))
    } else {
      None
    }
  }.toSet


    val searcher = new Searcher(docs = Seq(doc), specs = specs, fieldNames = Set.empty[String], maxSteps = None, writer = None, scorer = StaticWeightScorer.apply(), withReward = false)
    println(searcher.isSolution(pq))
    println(searcher.reward(pq))
    println(pq.checkOverApproximation(searcher.subsetEe, searcher.subsetSpecs))
    println(pq.checkUnderApproximation(searcher.subsetEe, searcher.subsetSpecs))

}