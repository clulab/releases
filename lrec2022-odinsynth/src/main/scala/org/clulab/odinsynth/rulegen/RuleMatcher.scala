package org.clulab.odinsynth.rulegen

import ai.lum.odinson.{ExtractorEngine, Mention}

/* Maches a rule with the index and returns the lisst of mentions
 * 
 * @constructor create a new rulematcher
 * @param extractorEngine the extractor engine [[ai.lum.odinson.ExtractorEngine]]
 */
class RuleMatcher(extractorEngine: ExtractorEngine) {
  /** Return a list of mentions that matches rule
   *
   *  @param rule a valid odinson rule
   *  @return a list of mentions 
   */
  def getMentions(rule: String): Seq[Mention] = {
    // compile the rule 
    val extractor = extractorEngine.ruleReader.compileRuleFile(rule)
    // match a rule agains the index
    val mentions = extractorEngine.extractMentions(extractor, 10, false, false).toSeq
    // get the nuber of hits
    val nHits = mentions.size
    // get the number of docs hit by the rule
    val nDocs = mentions.map(_.docId).distinct.size
    // just returning the mentions
    mentions
    /* this part of the code generates the TSV file
     * TODO: move this to the file handler class?
    val extractions: Seq[String] = mentions.map(m => {
      // columns: document \t sentence_id \t start_offset \t end_offset
      // this should be a list of things Marco asked me to do.
      List(
        m.odinsonMatch.start, 
        m.odinsonMatch.end,
        // marco said the luceneDocId is for internal use only
        m.docId,
        // and the sentence id
        m.sentenceId
        // 
        // we do not need the string anymore
        //ee.doc(m.luceneDocId).getField("raw").stringValue() // get the sentence
      ).mkString("\t")
    })
    extractions
    */
  }
} 
