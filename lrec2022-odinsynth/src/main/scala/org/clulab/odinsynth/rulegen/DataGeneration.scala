package org.clulab.odinsynth.rulegen

import ai.lum.odinson.{ExtractorEngine}
import ai.lum.odinson.{Document, StringField, TokensField, Sentence}
import java.io.File
import scala.util.Random
import scala.collection.mutable.ArrayBuffer
import util.control.Breaks._

/** Generates rules
  *
  * 1. randomly select a document from an index
  * 2. randomly select a sentence from the document
  * 3. select a list of candidate spans based on chunks
  * 4. select the longest candidate
  * 5. generate a rule based on the annotations
  * 6. compile and query the rule
  * 7. save the list of mentions as specs
  * 8. save the rule
  */
object DataGeneration extends App {
  // create a little interface with the user.
  // TODO: add a BATCH option.
  // get the extractor engine
  val ee = ExtractorEngine.fromConfig()
  //
  // RANDOM SENTENCE SELECTION
  val docHandler = new DocumentHandler(ee)
  var total: Int = 1
  val result: ArrayBuffer[String] = new ArrayBuffer
  // get a random sentence
  while (true) {
    breakable {
      val sentence: Sentence = docHandler.getRandomSentence
      //  get annotations from sentence
      val annotations: Map[String, Seq[String]] = sentence
      // create a list with the annotationsk
      .fields
        .map(f => f.name -> docHandler.getField(f.name, sentence))
        // renove empty fields (nonTokenFields)
        .filter(f => f._2.nonEmpty)
        // keep only raw, word, and tag bc we are only interested in them for now
        .filter(f =>
          Seq("raw", "word", "tag", "lemma", "entity", "chunk").contains(f._1)
        )
        // convert it to a map
        .toMap
      println(s"sentence: <${annotations("raw").mkString(" ")}>")
      //
      // candidate selection
      val candidateSelector = new CandidateSelector(annotations)
      //
      // use the requested tokenIds
      val candidates = candidateSelector.getCandidates(annotations("chunk"))
      // break if shit happens
      if (candidates.isEmpty) break
      //
      //
      val tokenIds = candidateSelector.getTokenIds(candidates)
      println(s"selected span x:<${tokenIds.head}> and y:<${tokenIds.tail}>")
      //
      if (tokenIds.length < 4) break

      val selectedAnn = candidateSelector.getSelectedAnnotation(tokenIds)
      selectedAnn.map(f =>
        println(s"annType: <${f._1}>  annotations: <${f._2.mkString(" ")}>")
      )
      //
      // RULE GENERATION
      val ruleGen = new RuleGenerator(selectedAnn, tokenIds.size)
      // generate rule
      val rule = ruleGen.run.trim
      if (result.contains(rule)) {
        println("Rule was already generated. Skipping...")
        break
      } else {
        println(s"adding rule to results: <${rule}>")
        result += rule
      }
      //
      //
      // match the rule
      val ruleMatcher = new RuleMatcher(ee)
      //
      val fullRule = """
        |rules:
        |  - name: testrule
        |    type: basic
        |    pattern: |
        |      """.stripMargin + rule
      // SPECS GENERATION
      //
      //println(s"rule generated: <${rule}>")
      val mentions = ruleMatcher.getMentions(fullRule)
      //
      val nDocs = mentions.map(_.docId).distinct.size
      println(s"# hits: ${mentions.size}, # matching docs: $nDocs")
      //
      println(s"processed rule n: <${total}>")
      total = total + 1

      // redundancy check
      //val specs = Spec.fromOdinsonMentions(mentions)
      // odinson document
      //val document: Document = docHandler.odinsonDoc.get
      // field names
      //val fieldNames = Set("tag", "word", "entity")
      //
      //val searcher = new Searcher(document, specs, fieldNames)
      // run check
      // parse rule
      //val state = Parser.parseBasicQuery(rule)
      //println(s"Running redundancy check <${searcher.redundancyCheck(state)}>")
      //
      //println("\n...\n...\n...\npress whawtever to keep playing...")
      //scala.io.StdIn.readLine()
    }
  }
}
