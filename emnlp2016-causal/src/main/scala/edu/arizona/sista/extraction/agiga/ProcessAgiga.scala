package preprocessing.agiga

import java.io.PrintWriter

import edu.arizona.sista.odin._
import edu.arizona.sista.processors.fastnlp.FastNLPProcessor
//import sentenceanalysis.AnalysisSentenceAPI._
//import extractionUtils.WorldTreeActions
//import utils._
import java.io._
import edu.jhu.agiga.{AgigaSentence, StreamingDocumentReader, AgigaPrefs}
import scala.collection.JavaConverters._
import edu.arizona.sista.processors.{DependencyMap, Document, Sentence}
import edu.arizona.sista.struct.DirectedGraph
import scala.collection.parallel.ForkJoinTaskSupport
import edu.arizona.sista.utils.Serializer

/** https://github.com/myedibleenso/processors-agiga
  * Created by bsharp on 2/29/16.
  */


// For working with the agiga deps
abstract class DependencyRepresentation
case class Root(i:Int) extends DependencyRepresentation
case class Edge(headIndex:Int, depIndex:Int, relation: String) extends DependencyRepresentation

object ProcessAgiga {
  val proc = new FastNLPProcessor(withDiscourse = false)

  def mkDependencies(s: AgigaSentence):DirectedGraph[String] = {
    // collapsed dependencies...
    val collapsedDeps = s.getColCcprocDeps.asScala

    val graphComponents =
      for {
        c <- collapsedDeps
        // component indices for edge construction
        depIndex = c.getDepIdx
        headIndex = c.getGovIdx
        // relation
        rel = c.getType
      } yield {
        headIndex match {
          case -1 => Root(depIndex)
          case _ => Edge(headIndex, depIndex, rel)
        }
      }

    val edges:List[(Int, Int, String)] = graphComponents
      .collect { case e: Edge => e }
      .map( e => (e.headIndex, e.depIndex, e.relation))
      .toList

    val roots:Set[Int] = graphComponents.collect { case r: Root => r }.map( r => r.i).toSet

    new DirectedGraph[String](edges, roots)
  }

  /** Converts agiga annotations to a Processors Document
    * and then generates a text representation of that Document using a specified "view"
    * view: words, lemmas, tags, entities, etc
    */
  def agigaDocToDocument(filename: String): Document = {
    // Setup Gigaword API Preferences
    val prefs = new AgigaPrefs()
    // label for agiga dependency type
    prefs.setAll(true)
    prefs.setWord(true)
    // Retrieve all gigaword documents contained within a given file
    val reader = new StreamingDocumentReader(filename, prefs)
    val sentences = for {
      agigaDoc <- reader.iterator().asScala
      s <- agigaDoc.getSents.asScala
      tokens = s.getTokens.asScala
      // words
      words = tokens.map(_.getWord)
      // lemmas
      lemmas = tokens.map(_.getLemma)
      // pos tags
      posTags = tokens.map(_.getPosTag)
      // ner labels
      nerLabels = tokens.map(_.getNerTag)
      // offsets
      startOffsets = tokens.map(_.getCharOffBegin)
      endOffsets = tokens.map(_.getCharOffEnd)
      deps = mkDependencies(s)

    } yield {

      val s = new Sentence(
        /** Actual tokens in this sentence */
        words.toArray,
        /** Start character offsets for the words; start at 0 */
        startOffsets.toArray,
        /** End character offsets for the words; start at 0 */
        endOffsets.toArray,
        /** POS tags for words (OPTION) */
        tags = Some(posTags.toArray),
        /** Lemmas (OPTION) */
        lemmas = Some(lemmas.toArray),
        /** NE labels (OPTION) */
        entities = Some(nerLabels.toArray),
        /** Normalized values of named/numeric entities, such as dates (OPTION) */
        norms = None,
        /** Shallow parsing labels (OPTION) */
        chunks = None,
        /** Constituent tree of this sentence; includes head words Option[Tree[String]] */
        syntacticTree = None,
        /** *Dependencies */
        dependenciesByType = new DependencyMap)
      // 1 for collapsed Stanford dependencies
      s.setDependencies(1, deps)
      s
    }

    new Document(sentences.toArray)
  }

  // Get all files in a directory ending with a given extension
  def findFiles(collectionDir: String, fileExtension:String): Array[File] = {
    val dir = new File(collectionDir)
    dir.listFiles(new FilenameFilter {
      def accept(dir: File, name: String): Boolean = name.endsWith(fileExtension)
    })
  }

  def main(args:Array[String]): Unit = {

    val dir = "/data/nlp/corpora/agiga/data/xml/"
    //val files = findFiles(dir, "gz").map(f => f.getAbsolutePath)
    //val agigaFile = "afp_eng_199405.xml.gz"
    val fileList = scala.io.Source.fromFile(args(0)).getLines().toArray.map(_.trim)

    for (agigaFile <- fileList) {
      val filePath = dir + agigaFile
      // Extract a Document from the file
      val doc = agigaDocToDocument(filePath)
      println ("Document Extracted from " + filePath)

      // Do chunking (not included with the agiga release?)
      proc.chunking(doc)
      println("Done chunking.")

      // Serialize and save
      val serFile = filePath + ".chunked.ser"
      Serializer.save(doc, serFile)
      println ("Serialized Document saved to " + serFile)
    }

  }

}
