package org.clulab.odinsynth

import ai.lum.common.ConfigUtils._
import ai.lum.common.{ConfigFactory}
import ai.lum.odinson.{
  Document => OdinsonDocument,
  Sentence => OdinsonSentence,
  NamedCapture,
  _
}
import ai.lum.odinson.lucene.OdinResults

import com.typesafe.config.{ConfigValueFactory, Config}
import java.io.{ File, PrintWriter }
import org.apache.lucene.document.{Document => LuceneDocument}
import ujson.Value
import upickle.default._
import upickle.default.{macroRW, ReadWriter => RW}


object CorpusReader {

  def fromConfig: CorpusReader = {
    val config = ConfigFactory.load()
    val extractorEngine = ExtractorEngine.fromConfig()
    val numEvidenceDisplay =
      config.get[Int]("ui.numEvidenceDisplay").getOrElse(3)
    val consolidateByLemma =
      config.get[Boolean]("ui.lemmaConsolidation").getOrElse(true)
    new CorpusReader(
      extractorEngine,
      numEvidenceDisplay,
      consolidateByLemma,
      None,
      config
    )
  }

  def fromExtractorEngine(ee: ExtractorEngine, config: Config): CorpusReader = {
    val numEvidenceDisplay =
      config.get[Int]("ui.numEvidenceDisplay").getOrElse(3)
    val consolidateByLemma =
      config.get[Boolean]("ui.lemmaConsolidation").getOrElse(true)
    new CorpusReader(
      ee,
      numEvidenceDisplay,
      consolidateByLemma,
      None,
      config
    )
  }

  def inMemory(ee: ExtractorEngine, docs: Seq[OdinsonDocument]) = {
    val config = ConfigFactory.load()
    val numEvidenceDisplay =
      config.get[Int]("ui.numEvidenceDisplay").getOrElse(3)
    val consolidateByLemma =
      config.get[Boolean]("ui.lemmaConsolidation").getOrElse(true)
    //
    new CorpusReader(ee, numEvidenceDisplay, consolidateByLemma, Some(docs), config)
  }
  // TODO: in memmory
}


class CorpusReader(
    val extractorEngine: ExtractorEngine,
    val numEvidenceDisplay: Int,
    consolidateByLemma: Boolean,
    documents: Option[Seq[OdinsonDocument]],
    config: Config
) {

  val docsDir = config[File]("odinson.docsDir")
  val DOC_ID_FIELD = config[String]("odinson.index.documentIdField")
  val SENTENCE_ID_FIELD = config[String]("odinson.index.sentenceIdField")
  val WORD_TOKEN_FIELD = config[String]("odinson.displayField")
  val pageSize = config[Int]("odinson.pageSize")

  def extractMatchesFromSearchString(search: String): Seq[SearchResult] = {
    extractMatchesFromSearchString(search, 8)
  }

  def extractMatchesFromSearchString(
      search: String,
      size: Int
  ): Seq[SearchResult] = {
    // build query
    val query = extractorEngine.compiler.mkQuery(search)
    // get mentions
    val results = extractorEngine.query(query, size)

    val matches = for {
      scoreDoc <- results.scoreDocs
      tokens = extractorEngine.getTokens(scoreDoc)
      matchHead = scoreDoc.matches.head
      sentence: OdinsonSentence =
        getOdinsonDoc(scoreDoc.doc).sentences(getSentenceIndex(scoreDoc.doc))
    } yield {
      SearchResult(
        text = tokens.mkString(" "),
        // get the documentId
        documentId = getDocId(scoreDoc.doc),
        // get the sentence id
        sentenceId = getSentenceIndex(scoreDoc.doc),
        // add the match
        matchStart = matchHead.start,
        matchEnd = matchHead.end,
        namedCaptures = matchHead.namedCaptures.toList,
        sentence = sentence,
        totalHits = results.totalHits,
      )
    }
    matches
  }

  def loadParentDocByDocumentId(documentId: String): OdinsonDocument = {
    // lucene doc containing metadata
    val parentDoc: LuceneDocument = extractorEngine.getParentDoc(documentId)
    // check inMemory index
    if (documents.isDefined) {
      documents.get.filter(d => d.id == documentId).head
    } else {
      val odinsonDocFile =
        new File(docsDir, parentDoc.getField("fileName").stringValue)
      OdinsonDocument.fromJson(odinsonDocFile)
    }
  }

  def retrieveSentenceJson(
      documentId: String,
      sentenceIndex: Int
  ): OdinsonDocument = {
    val odinsonDoc: OdinsonDocument = loadParentDocByDocumentId(documentId)
    odinsonDoc
  }

  def getSentenceIndex(luceneDocId: Int): Int = {
    val doc = extractorEngine.indexReader.document(luceneDocId)
    // if the document is already a base document
    if (doc.getValues(SENTENCE_ID_FIELD).length == 0) {
      luceneDocId
    } else {
      doc.getValues(SENTENCE_ID_FIELD).head.toInt
    }
  }

  def getDocId(luceneDocId: Int): String = {
    val doc: LuceneDocument = extractorEngine.indexReader.document(luceneDocId)
    doc.getValues(DOC_ID_FIELD).head
  }

  def getOdinsonDoc(sentenceId: Int): OdinsonDocument = {

    val sentenceIndex = getSentenceIndex(sentenceId)
    val documentId = getDocId(sentenceId)
    val odinsonDoc = retrieveSentenceJson(documentId, sentenceIndex)
    odinsonDoc
  }
}
