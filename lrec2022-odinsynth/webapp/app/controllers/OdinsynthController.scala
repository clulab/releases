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
import org.clulab.odinsynth.{
  CorpusReader, 
  MaskedSearcher,
  Searcher, 
  SearcherUtils,
  SearchResult, 
  Spec,
  SynthesizedRule, 
  Query
}
import org.clulab.odinsynth.rest._
import org.clulab.odinsynth.scorer.StaticWeightScorer
import org.clulab.odinsynth.scorer.DynamicWeightScorer
import org.clulab.processors.Processor
import ai.lum.odinson.{ 
  Document => OdinsonDocument, 
  Sentence => OdinsonSentence 
}
import com.typesafe.config.{ Config, ConfigRenderOptions }
import org.clulab.odinsynth.scorer.Scorer

/** This controller creates an `Action` to handle HTTP requests to the
  * application's home page.
  */
@Singleton
class OdinsynthController @Inject() (cc: ControllerComponents)
    extends AbstractController(cc) {

  import org.clulab.odinsynth.rest.JsonUtils.SearchResultJsonOps
  // -------------------------------------------------
  lazy val reader = CorpusReader.fromConfig
  // -------------------------------------------------

  val FIELD_NAMES = Set("word", "lemma", "tag")
  // initialize dynet and create processor
  lazy val processor = SearcherUtils.mkProcessor()

  lazy val config = ConfigFactory.load()

  lazy val scorer: Scorer = {
    val scorerType = config.getString("odinsynth.scorer.scorerType")
    if (scorerType == "StaticWeightScorer") {
      StaticWeightScorer()
    } else {
      DynamicWeightScorer(config.getString("odinsynth.scorer.endpoint"))
    }
  }
  lazy val maxSteps: Option[Int] = {
    if (config.hasPath("odinsynth.scorer.maxSteps")) {
      Some(config.getInt("odinsynth.scorer.maxSteps"))
    } else {
      None
    }
  }

  def scorerVersion() = Action { implicit request: Request[AnyContent] =>
    Ok(f"${scorer.version}")
  }

  def configInfo = Action {
    val options = ConfigRenderOptions.concise.setJson(true)
    Ok(Json.parse(config.root.render(options)))
  }

  // curl -v 127.0.0.1:9000/search?query=the
  def search(query: String) =
    Action { implicit request: Request[AnyContent] =>
      println(f"search -> ${query}")
      if (Try { reader.extractMatchesFromSearchString(query) }.isSuccess) {
        // SearchResult
        val searchResults: Seq[SearchResult] = reader.extractMatchesFromSearchString(query)
        // generate custom json 
        // using SearchResultJsonOps.playJson
        val results = JsArray(searchResults.map(_.playJson))
        // pass to view
        Ok(results)
      } else {
        Ok("")
      }
    }

  /**
    * Loads the JSON for an [[ai.lum.odinson.Document]] for the provided ID.
    */ 
  def getOdinsonDocument(docId: Int) =
    Action { implicit request: Request[AnyContent] =>
      val doc = reader.getOdinsonDoc(docId).toJson
      // pass to view
      Ok(doc)
    }
  // the class to generate rules should be declared here to keep state
  var rules: Iterator[SynthesizedRule] = null
  // whenever we call next this can just be called again
  // later we can have a better solution to this
  //
  def generateRule =
    Action(parse.json) { implicit request: Request[JsValue] =>
      println(f"generateRule -> ${request.body}")
      // get data
      val data = (request.body \ "data").as[List[JsValue]]

      // get list of specs
      //
      // first generate the sentences
      val sentences = (for {
        v <- data
        //sentId = (v \ "sentId").as[Int]
        docId = (v \ "docId").as[String]
        sentId = (v \ "sentId").as[Int]
      } yield reader
        .loadParentDocByDocumentId(docId)
        .sentences(sentId)).distinct.toSeq

      // create the document
      val docName = "test-doc"
      val doc = OdinsonDocument.apply(
        id = docName, 
        metadata = Nil, 
        sentences = sentences
      )
      // filter counter-examples
      val positiveExamples =
        data.filter(v => Try((v \ "span" \ "start").as[Int]).isSuccess)
      // start search
      val specs = (for {
        v <- positiveExamples;
        sentId = (v \ "sentId").as[Int]
        docId = (v \ "docId").as[String]
        start = (v \ "span" \ "start").as[Int]
        end = (v \ "span" \ "end").as[Int]
        sentence = reader.loadParentDocByDocumentId(docId).sentences(sentId)
        fakeSentId = sentences.indexOf(sentence)
      } yield Spec.apply(docName, fakeSentId, start, end)).toSet

      val searcher = new Searcher(Seq(doc), specs, FIELD_NAMES, None, None, scorer, false)

      // pass to view
      rules = searcher.findAll.iterator
      if (rules.hasNext) {
        val nextRule = rules.next
        if (nextRule != null) {
          Ok(nextRule.rule.pattern+"\t"+nextRule.nSteps.toString+"\t"+nextRule.currentSteps)
        } else {
          Ok("No rule was found")
        }
      } else {
        Ok("No rule was found")
      }
    }

  /**
    * Converts a subclass/implementation of [[org.clulab.odinsynth.rest.Example]] to an [[ai.lum.odinson.Sentence]].
    */ 
  def toSentence(example: Example): OdinsonSentence = {
    reader.loadParentDocByDocumentId(example.docId).sentences(example.sentId)
  }

  /**
    * Converts an [[org.clulab.odinsynth.Spec]] and [[ai.lum.odinson.Sentence]] pair into a token-based mask.
    * <br>
    * Currently unused.
    */ 
  def toMask(spec: Spec, sentence: OdinsonSentence): Seq[Boolean] = {
    // initialize mask
    val mask: Array[Boolean] = (1 to sentence.numTokens).map{_ => true }.toArray
    val maskValue = false
    spec.captures.foreach { 
      case (_: String, capture: Spec) =>
        // FIXME: should this be until?
        (spec.start to spec.end).foreach { i =>
          mask(i) = maskValue
        }
    }
    mask
  }

  /**
    * Creates an instance of an [[ai.lum.odinson.Document]] from a sequence of [[ai.lum.odinson.Sentence]]s.
    */ 
  def mkTestDoc(
    id: String = "test-doc", 
    sentences: Seq[OdinsonSentence]
  ): OdinsonDocument = OdinsonDocument(
    id = id, 
    metadata = Nil, 
    sentences = sentences
  )

  /** Generates an [[org.clulab.odinsynth.MaskedSearcher]] using several defaults.
   * 
   */ 
  def mkMaskedSearcher(
    doc: OdinsonDocument,
    specs: Seq[Set[Spec]],
    masked: Seq[Boolean],
  ): MaskedSearcher = new MaskedSearcher(
    originalDoc = doc,
    specs = specs,
    masked = masked,
    fieldNames = FIELD_NAMES,
    // FIXME: do we want to specify a max num. of steps?
    maxSteps = maxSteps,
    // FIXME: do we need/want a writer?
    writer = None,
    scorer = scorer,
    withReward = false,
  )

  /** Generates an [[org.clulab.odinsynth.Searcher]] using several defaults. 
   * 
   */ 
  def mkSearcher(
    doc: OdinsonDocument,
    specs: Set[Spec]
  ): Searcher = new Searcher(
    docs = Seq(doc),
    specs = specs,
    fieldNames = FIELD_NAMES,
    maxSteps = maxSteps,
    // FIXME: is this right?
    writer = None,
    scorer = scorer,
    withReward = false
  )

  def remapExampleSentId(examples: Seq[Example]): (Seq[Example], Seq[OdinsonSentence], OdinsonDocument) = {

    val sentences = examples.map(toSentence).distinct.toSeq
    // FIXME: why not just use reader.loadParentDocByDocumentId(docId) ?
    val doc = mkTestDoc(sentences = sentences)
    

    val sentenceToId: Map[String, Int] = examples.map(it => f"${it.docId}-${it.sentId}").distinct.zipWithIndex.toMap
    // Adjust the example to correspond to the created document (@see doc)
    val examplesForDoc = examples.map { 
      case pe: PositiveExample => pe.copy(docId = doc.id, sentId = sentenceToId(f"${pe.docId}-${pe.sentId}"))
      case ne: NegativeExample => ne.copy(docId = doc.id, sentId = sentenceToId(f"${ne.docId}-${ne.sentId}"))
    }

    (examplesForDoc, sentences, doc)

  }

  /**
   * Generates a rule without masking masking the specification.
   */
  def generateUnMaskedRule: Action[AnyContent] = Action { request => 
    val json = request.body.asJson.get
    
    // get data
    val data = (json \ "data").as[List[JsValue]]
    val examples: Seq[Example] = data.map(ExampleUtils.fromPlayJson)

    val sentences = examples.map(toSentence).distinct.toSeq

    // FIXME: why not just use reader.loadParentDocByDocumentId(docId) ?
    val doc = mkTestDoc(sentences = sentences)
    // consider only positive examples
    val positiveExamples: Seq[PositiveExample] = examples.collect { case pe: PositiveExample => pe }
    // start search
    val specifications: Set[Spec] = positiveExamples.map(_.toSpec).toSet

    // initialize searcher
    // FIXME: MaskedSearcher seems robust enough to handle both use cases (i.e., generateUnMaskedRule vs generateMaskedRule)
    val searcher = mkSearcher(doc, specifications)

    rules = searcher.findAll.iterator
    // FIXME: rules uses some sort of global var, but it should probably instead relate to a session
    val nextRule = rules.next

    // FIXME: why not return a simple JSON response?
    Ok(nextRule.rule.pattern+"\t"+nextRule.nSteps.toString+"\t"+nextRule.currentSteps)
  }

  /**
   * Endpoint for generating a rule in the entity use-case (i.e. there is one masked part and a context part)
   */
  def generateEntityMaskedRule: Action[AnyContent] = Action { request => 
    val json = request.body.asJson.get
    
    // get data
    val data = (json \ "data").as[List[JsValue]]
    val rawExamples: Seq[Example] = data.map(ExampleUtils.fromPlayJson)
    val (examples, sentences, doc) = remapExampleSentId(rawExamples)//examples.map(toSentence).distinct.toSeq

    // filter counter-examples
    val positiveExamples: Seq[PositiveExample] = examples.collect { case pe: PositiveExample => pe }
    // start search
    val specifications: Seq[Set[Spec]] = positiveExamples.map(_.toEntitySpec).transpose.map(_.toSet)
    val masked        : Seq[Seq[Boolean]]   = positiveExamples.map(it => it.maskOrder).distinct

    // initialize searcher
    val searcher = mkMaskedSearcher(
      doc = doc, 
      specs = specifications,
      masked = masked.flatten
    )

    rules = searcher.findAll.iterator

    // FIXME: rules uses some sort of global var, but it should probably instead relate to a session
    // FIXME look into nulls
    val nextRule = if(rules.hasNext) {
      val next = rules.next
      if (next != null) Some(next) else None
    } else { 
      None
    }

    // FIXME: why not return a simple JSON response?
    Ok(nextRule.map(_.rule.pattern).getOrElse("")+"\t"+nextRule.map(_.nSteps.toString).getOrElse("")+"\t"+nextRule.map(_.currentSteps).getOrElse(Float.PositiveInfinity))
  }
  /**
   * Endpoint for generating a rule in the relation use-case (i.e. there are two masked parts and a context part)
   */
  def generateRelationMaskedRule: Action[AnyContent] = ???


  def nextRule =
    Action { implicit request: Request[AnyContent] =>
      if (rules.hasNext) {
        val nextRule = rules.next
        // returns a different rule
        Ok(nextRule.rule.pattern+"\t"+nextRule.nSteps.toString+"\t"+nextRule.currentSteps)
      } else {
        Ok("" + "\t" + "" + "\t" + Float.PositiveInfinity.toString())
      }
    }
}
