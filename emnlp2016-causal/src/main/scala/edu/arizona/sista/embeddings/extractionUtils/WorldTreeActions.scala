package extractionUtils

import edu.arizona.sista.odin._
import edu.arizona.sista.utils.DependencyUtilsException


/**
  * Created by bsharp on 2/18/16.
  */
class WorldTreeActions extends Actions {

  // these patterns are meant to be applied on the lemmas
  val posRegTriggerPattern = "acceler|accept|accumul|action|activat|aid|allow|associ|augment|cataly|caus|cleav|confer|contribut|convert|direct|driv|elev|elicit|enabl|enhanc|escort|export|gener|high|increas|induc|induct|initi|interact|interconvert|involv|lead|led|major|mediat|modul|necess|overexpress|potent|proce|produc|prolong|promot|rais|reactivat|re-express|releas|render|requir|rescu|respons|restor|result|retent|signal|stimul|support|synerg|synthes|target|trigger|underli|up-regul|upregul".r

  val negRegTriggerPattern = "abolish|abrog|absenc|antagon|arrest|attenu|block|blunt|deactiv|decreas|defect|defici|degrad|delay|deplet|deregul|diminish|disengag|disrupt|down|down-reg|downreg|drop|dysregul|elimin|impair|imped|inactiv|inhibit|interf|knockdown|lack|limit|loss|lost|lower|negat|neutral|nullifi|oppos|overc|perturb|prevent|reduc|reliev|remov|repress|resist|restrict|revers|shutdown|slow|starv|suppress|supress|uncoupl".r

  def matchesRegulationTrigger(s: String): Boolean = {
    posRegTriggerPattern.findFirstIn(s).nonEmpty || negRegTriggerPattern.findFirstIn(s).nonEmpty
  }

  def getHeadLemma(m: Mention): Option[String] = {
    val result = try {
      m.synHeadLemma
    } catch {
      case e: StackOverflowError => {
        println(s"1 DependencyUtils error at sentence ${m.sentence}, ignoring")
        println("\t" + m.sentenceObj.getSentenceText())
        None
      }
      case e: DependencyUtilsException => {
        println(s"DependencyUtilsException (can't find root): error at sentence ${m.sentence}, ignoring")
        println("\t" + m.sentenceObj.words.zipWithIndex.toSeq)
        println("\t" + m.tokenInterval)
        println("\t" + m.sentenceObj.dependencies.get)
        None
      }

    }
    result
  }

  def getHeadTag(m: Mention): Option[String] = {
    val result = try {
      m.synHeadTag
    } catch {
      case e: StackOverflowError => {
        println(s"1 DependencyUtils error at sentence ${m.sentence}, ignoring")
        println("\t" + m.sentenceObj.getSentenceText())
        None
      }
      case e: DependencyUtilsException => {
        println(s"DependencyUtilsException (can't find root): error at sentence ${m.sentence}, ignoring")
        println("\t" + m.sentenceObj.words.zipWithIndex.toSeq)
        println("\t" + m.tokenInterval)
        println("\t" + m.sentenceObj.dependencies.get)
        None
      }

    }

    result
  }

  def handleTransparent(mentions: Seq[Mention], state: State): Seq[Mention] = {
    val results = for {
      m <- mentions
      headLemma <- getHeadLemma(m)
      if Transparent contains headLemma
    } yield m
    results
  }

  def filterNounPhrases(mentions: Seq[Mention], state: State): Seq[Mention] = for {
    m <- mentions
    headTag <- getHeadTag(m)
    if headTag startsWith "NN"
    headLemma <- getHeadLemma(m)
    // head shouldn't be a trigger
    if !matchesRegulationTrigger(headLemma)
  } yield m


}


