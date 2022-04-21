package org.clulab.odinsynth.rulegen

import ai.lum.odinson.{ExtractorEngine}
import ai.lum.odinson.{Document => OdinsonDoc, StringField, TokensField, Sentence}
import org.apache.lucene.document.{Document => LuceneDoc}
import scala.util.Random
import java.io.File

/** Deal with json files and the lucene index
 *
 * @constructor create a new file handler
 * @param extractorEngine [[ai.lum.odinson.ExtractorEngine]]
 */
class DocumentHandler(extractorEngine: ExtractorEngine) {
  var docId: Option[Int] = None
  var luceneDoc: Option[LuceneDoc] = None
  var odinsonDoc: Option[OdinsonDoc] = None
  /* Returns a random sentence read with [[ai.lum.odinson.ExtractorEngine]] */ 
  def getRandomSentence: Sentence = {
    // get a random document
    this.getRandomDocumentId
    // come up with a random doc id
    this.getLuceneDocument
    // 
    this.getOdinsonDocument
    //
    // get a random sentence
    // description of the annotations field
    // numbef of fields 7, fields:
    // raw, word, tag, lemma, entity, chunk, dependencies
    // have an array with annotations
    Random.shuffle(this.odinsonDoc.get.sentences).head
  }
  
  /** Returns a field from from a lucene document */
  def getLuceneDocumentField(field: String): String = this.luceneDoc.get.getField(field).stringValue
  
  /** Returns a random document id from lucene index */
  def getRandomDocumentId  = this.docId = Some(Random.nextInt(extractorEngine.numDocs))
  
  /** Returns a lucene document [[org.apache.lucene.document.Document]] */
  def getLuceneDocument = this.luceneDoc = Some( extractorEngine.doc(this.docId.get) )
  
  /** Returns return the complete path to the json document */
  def getFileName = new File(
    "/data/nlp/corpora/umbc/odinson_docs_basic_dependencies/" +
    getLuceneDocumentField("docId") +
    ".json")
  
  /** Returns the odinson document [[ai.lum.odinson.Document]] */
  def getOdinsonDocument = this.odinsonDoc = Some(OdinsonDoc.fromJson(this.getFileName))
  
  /** Returns turn a fild from a sealed trait
   *
   * @param field name of the field (raw, tag, word...)
   * @param sentence the selected sentence [[ai.lum.odinson.Sentence]]
   * @return the sequence of annotations
   */
  def getField (field: String, sentence: Sentence): Seq[String] = {
    sentence.fields.filter(_.name == field).head match {
      case t: TokensField =>  t.tokens
      case _ => Seq()
    }
  }
}

