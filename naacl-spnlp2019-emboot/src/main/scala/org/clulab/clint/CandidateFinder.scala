package org.clulab.clint

import org.clulab.odin._
import org.clulab.processors.Document
import scala.collection.Seq
import scala.collection.mutable.ListBuffer

trait CandidateFinder {
  def findCandidates(doc: Document): Seq[Mention]
}

class NounPhraseCandidateFinder extends CandidateFinder {

  val rules = """
    |- name: NounPhrases
    |  label: Candidate
    |  type: token
    |  priority: 1
    |  action: filterNounPhrases
    |  pattern: |
    |    [chunk='B-NP'][chunk='I-NP']*
    |""".stripMargin

  object actions extends Actions {
    def filterNounPhrases(mentions: Seq[Mention], state: State): Seq[Mention] = {
      // only keep mentions if at least one of the tokens is a noun
      for {
        m <- mentions
        tags <- m.tags
        if tags.exists(_.startsWith("N"))
      } yield m
    }
  }

  val extractor = ExtractorEngine(rules, actions)

  def findCandidates(doc: Document): Seq[Mention] = {
    extractor.extractFrom(doc)
  }

}

/*
  // Version 1 

class OracleCandidateFinder extends CandidateFinder {
  
    val rule = """
  				|- name: OracleEntityCandidates
  				|  label: Candidate
  				|  type: token
  				|  priority: 1
  				|  pattern: |
  				|    [entity=/^B-/]? [entity=/^I-/]+
  				|""".stripMargin
  
  	val extractor = ExtractorEngine(rule)
  
    def findCandidates(doc: Document): Seq[Mention] = {
      extractor.extractFrom(doc)
    }
*/

/** reads annotations in IOB2 format used by ScienceIE Dataset (converted to CoNLL format using the brat tool utility) */
class OracleCandidateFinderScienceIE extends CandidateFinder {
  
  val rules = """
    |    
    |- name: Material_Candidates
    |  label: [Material, Candidate]
    |  type: token
    |  priority: 1
    |  pattern: |
    |    [entity='I-Material']+
    |      |
    |    [entity='B-Material'] [entity='I-Material']*
    |
    |- name: Task_Candidates
    |  label: [Task, Candidate]
    |  type: token
    |  priotity: 1
    |  pattern: |
    |    [entity='I-Task']+
    |      |
    |    [entity='B-Task'] [entity='I-Task']*
    |
    |- name: Process_Candidates
    |  label: [Process, Candidate]
    |  type: token
    |  priority: 1
    |  pattern: |
    |    [entity='I-Process']+
    |      |
    |    [entity='B-Process'] [entity='I-Process']*
    |""".stripMargin

  val extractor = ExtractorEngine(rules)

  def findCandidates(doc: Document): Seq[Mention] = {
    extractor.extractFrom(doc)
  }

}

/** reads annotations in IOB2 format used by Ontonotes Dataset (converted to CoNLL format using the brat tool utility ; the conversion to ann format is done by ReadOntoNotesData.scala) */
class OracleCandidateFinderOntonotes extends CandidateFinder {
  
  val rules = """
    |- name: PERSON_Candidates
    |  label: [PERSON, Candidate]
    |  type: token
    |  priority: 1
    |  pattern: |
    |    [entity='B-PERSON'] [entity='I-PERSON']*
    |
    |- name: NORP_Candidates
    |  label: [NORP, Candidate]
    |  type: token
    |  priority: 1
    |  pattern: |
    |    [entity='B-NORP'] [entity='I-NORP']*
    |
    |- name: FAC_Candidates
    |  label: [FAC, Candidate]
    |  type: token
    |  priority: 1
    |  pattern: |
    |    [entity='B-FAC'] [entity='I-FAC']*
    |
    |- name: ORG_Candidates
    |  label: [ORG, Candidate]
    |  type: token
    |  priority: 1
    |  pattern: |
    |    [entity='B-ORG'] [entity='I-ORG']*
    |
    |- name: GPE_Candidates
    |  label: [GPE, Candidate]
    |  type: token
    |  priority: 1
    |  pattern: |
    |    [entity='B-GPE'] [entity='I-GPE']*
    |
    |- name: LOC_Candidates
    |  label: [LOC, Candidate]
    |  type: token
    |  priority: 1
    |  pattern: |
    |    [entity='B-LOC'] [entity='I-LOC']*
    |
    |- name: PRODUCT_Candidates
    |  label: [PRODUCT, Candidate]
    |  type: token
    |  priority: 1
    |  pattern: |
    |    [entity='B-PRODUCT'] [entity='I-PRODUCT']*
    |
    |- name: EVENT_Candidates
    |  label: [EVENT, Candidate]
    |  type: token
    |  priority: 1
    |  pattern: |
    |    [entity='B-EVENT'] [entity='I-EVENT']*
    |
    |- name: WORK_OF_ART_Candidates
    |  label: [WORK_OF_ART, Candidate]
    |  type: token
    |  priority: 1
    |  pattern: |
    |    [entity='B-WORK_OF_ART'] [entity='I-WORK_OF_ART']*
    |
    |- name: LAW_Candidates
    |  label: [LAW, Candidate]
    |  type: token
    |  priority: 1
    |  pattern: |
    |    [entity='B-LAW'] [entity='I-LAW']*
    |
    |- name: LANGUAGE_Candidates
    |  label: [LANGUAGE, Candidate]
    |  type: token
    |  priority: 1
    |  pattern: |
    |    [entity='B-LANGUAGE'] [entity='I-LANGUAGE']*
    |
    |- name: DATE_Candidates
    |  label: [DATE, Candidate]
    |  type: token
    |  priority: 1
    |  pattern: |
    |    [entity='B-DATE'] [entity='I-DATE']*
    |
    |- name: TIME_Candidates
    |  label: [TIME, Candidate]
    |  type: token
    |  priority: 1
    |  pattern: |
    |    [entity='B-TIME'] [entity='I-TIME']*
    |
    |- name: PERCENT_Candidates
    |  label: [PERCENT, Candidate]
    |  type: token
    |  priority: 1
    |  pattern: |
    |    [entity='B-PERCENT'] [entity='I-PERCENT']*
    |
    |- name: MONEY_Candidates
    |  label: [MONEY, Candidate]
    |  type: token
    |  priority: 1
    |  pattern: |
    |    [entity='B-MONEY'] [entity='I-MONEY']*
    |
    |- name: QUANTITY_Candidates
    |  label: [QUANTITY, Candidate]
    |  type: token
    |  priority: 1
    |  pattern: |
    |    [entity='B-QUANTITY'] [entity='I-QUANTITY']*
    |
    |- name: ORDINAL_Candidates
    |  label: [ORDINAL, Candidate]
    |  type: token
    |  priority: 1
    |  pattern: |
    |    [entity='B-ORDINAL'] [entity='I-ORDINAL']*
    |
    |- name: CARDINAL_Candidates
    |  label: [CARDINAL, Candidate]
    |  type: token
    |  priority: 1
    |  pattern: |
    |    [entity='B-CARDINAL'] [entity='I-CARDINAL']*
    |""".stripMargin

  val extractor = ExtractorEngine(rules)

  def findCandidates(doc: Document): Seq[Mention] = {
    extractor.extractFrom(doc)
  }

}

/** reads annotations in IOB2 format, used by CoNLL 2003 */
class OracleCandidateFinder extends CandidateFinder {

  val rules = """
    |- name: LOC_Candidates
    |  label: [LOC, Candidate]
    |  type: token
    |  priority: 1
    |  pattern: |
    |    [entity='I-LOC']+
    |      |
    |    [entity='B-LOC'] [entity='I-LOC']*
    |
    |- name: PER_Candidates
    |  label: [PER, Candidate]
    |  type: token
    |  priotity: 1
    |  pattern: |
    |    [entity='I-PER']+
    |      |
    |    [entity='B-PER'] [entity='I-PER']*
    |
    |- name: ORG_Candidates
    |  label: [ORG, Candidate]
    |  type: token
    |  priority: 1
    |  pattern: |
    |    [entity='I-ORG']+
    |      |
    |    [entity='B-ORG'] [entity='I-ORG']*
    |
    |- name: MISC_Candidates
    |  label: [MISC, Candidate]
    |  type: token
    |  priority: 1
    |  pattern: |
    |    [entity='I-MISC']+
    |      |
    |    [entity='B-MISC'] [entity='I-MISC']*
    |""".stripMargin

  val extractor = ExtractorEngine(rules)

  def findCandidates(doc: Document): Seq[Mention] = {
    extractor.extractFrom(doc)
  }

}
