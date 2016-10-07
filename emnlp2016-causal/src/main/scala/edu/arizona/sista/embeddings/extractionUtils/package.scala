import java.io.PrintWriter

import edu.arizona.sista.odin._
import edu.arizona.sista.processors.{Sentence, Document}
import java.util.concurrent.{Callable, FutureTask, TimeUnit}

import scala.collection.mutable.ArrayBuffer

package object extractionUtils {

  val Transparent = Set(
    "ability", "achievement", "activity", "addition", "adjustment", "administration", "agent", "alteration", "amount", "auc",
    "case", "capacity", "cohort", "combination", "concentration", "concept", "conclusion", "condition", "consequence", "country", "cow", "curve",
    "datum", "development", "diagnosis", "difference",
    "effect", "estimate", "evidence","exacerbation", "exposure", "expression",
    "factor", "finding", "form", "formation", "frequency", "group",
    "hour", "hypothesis", "idea", "importance", "improvement", "incidence", "indication","intervention",
    "kind", "level", "notion", "measurement", "mg", "mouse", "number", "observation", "occurrence", "origin",
    "part", "performance", "persistence", "pool", "possibility", "presence", "prevalence", "priority", "progression", "project", "proliferation", "proposal",
    "range", "rate", "referral", "risk", "role",
    "safety", "size", "state", "status", "step", "strategy", "study", "substance", "susceptibility",
    "technique", "time", "treatment", "turn", "type",
    "understanding", "use", "view", "work", "year", "dozen", "hundred", "thousand", "score", "multitude", "lot", "bunch", "set", "host", "world"
  )

  // NOTE: these should use the lemma form
  val StopWords = Set("background", "common", "introduction", "invitation", "invitee", "test", "first", "wide", "other", "paper", "procedure", "revision", "fact", "question", "word", "hour")


    // add RB to grab "not"?
  val VALIDTAG = """^(NN|JJ|VB).*"""

  val VALID_NE_TAGS = Array("PERSON", "LOCATION", "ORGANIZATION", "DATE")

  // a subset of punctuation that we want to avoid
  // should we use \p{Punct} instead?
  val PUNCT =
    """.*[%\.\]\[\(\)].*"""

  @throws(classOf[java.util.concurrent.TimeoutException])
  def timedRun[F](timeout: Long)(f: => F): F = {

    val task = new FutureTask(new Callable[F]() {
      def call() = f
    })

    new Thread(task).start()

    task.get(timeout, TimeUnit.MILLISECONDS)
  }


  def displayMentions(mentions: Seq[Mention], doc: Document): Unit = {
    val mentionsBySentence = mentions groupBy (_.sentence) mapValues (_.sortBy(_.start)) withDefaultValue Nil
    for ((s, i) <- doc.sentences.zipWithIndex) {
      println(s"sentence #$i")
      println(s.getSentenceText)
      println("Tokens: " + (s.words.indices, s.words, s.tags.get).zipped.mkString(", "))
      printSyntacticDependencies(s)
      println

      val sortedMentions = mentionsBySentence(i).sortBy(_.label)
      val (events, entities) = sortedMentions.partition(_ matches "Event")
      val (tbs, rels) = entities.partition(_.isInstanceOf[TextBoundMention])
      val sortedEntities = tbs ++ rels.sortBy(_.label)
      println("entities:")
      sortedEntities foreach displayMention

      println
      println("events:")
      events foreach displayMention
      println("=" * 50)
    }
  }

  def printSyntacticDependencies(s:Sentence): Unit = {
    if(s.dependencies.isDefined) {
      println(s.dependencies.get.toString)
    }
  }

  def displayMention(mention: Mention) {
    val boundary = s"\t${"-" * 30}"
    println(s"${mention.labels} => ${mention.text}")
    println(boundary)
    println(s"\tRule => ${mention.foundBy}")
    val mentionType = mention.getClass.toString.split("""\.""").last
    println(s"\tType => $mentionType")
    println(boundary)
    mention match {
      case tb: TextBoundMention =>
        println(s"\t${tb.labels.mkString(", ")} => ${tb.text}")
//        println(s"\theadLemma:${tb.synHeadLemma.get}, ${tb.tokenInterval.toString()}")
      case em: EventMention =>
        println(s"\ttrigger => ${em.trigger.text}")
//        println(s"\theadLemma:${em.synHeadLemma.get}, ${em.tokenInterval.toString()}")
        displayArguments(em)
      case rel: RelationMention =>
        displayArguments(rel)
      case _ => ()
    }
    println(s"$boundary\n")
  }

  def displayArguments(b: Mention): Unit = {
    b.arguments foreach {
      case (argName, ms) =>
        ms foreach { v =>
          println(s"\t$argName ${v.labels.mkString("(", ", ", ")")} => ${v.text}")
        }
    }
  }

  def printMention(mention: Mention, pw:PrintWriter) {
    val boundary = s"\t${"-" * 30}"
    pw.println(s"${mention.labels} => ${mention.text}")
    pw.println(boundary)
    pw.println(s"\tRule => ${mention.foundBy}")
    val mentionType = mention.getClass.toString.split("""\.""").last
    pw.println(s"\tType => $mentionType")
    pw.println(boundary)
    mention match {
      case tb: TextBoundMention =>
        pw.println(s"\t${tb.labels.mkString(", ")} => ${tb.text}")
      case em: EventMention =>
        pw.println(s"\ttrigger => ${em.trigger.text}")
        printArguments(em, pw)
      case rel: RelationMention =>
        printArguments(rel, pw)
      case _ => ()
    }
    pw.println(s"$boundary\n")
  }

  def printArguments(b: Mention, pw: PrintWriter): Unit = {
    b.arguments foreach {
      case (argName, ms) =>
        ms foreach { v =>
          pw.println(s"\t$argName ${v.labels.mkString("(", ", ", ")")} => ${v.text}")
        }
    }
  }


  def mentionToString(mention: Mention):String = {
    val os = new StringBuilder

    val boundary = s"\t${"-" * 30}\n"
    os.append(s"${mention.labels} => ${mention.text}\n")
    os.append(boundary)
    os.append(s"\tRule => ${mention.foundBy}\n")
    val mentionType = mention.getClass.toString.split("""\.""").last
    os.append(s"\tType => $mentionType\n")
    os.append(boundary)
    mention match {
      case tb: TextBoundMention =>
        os.append(s"\t${tb.labels.mkString(", ")} => ${tb.text}\n")
      case em: EventMention =>
        os.append(s"\ttrigger => ${em.trigger.text}\n")
        os.append( argumentsToString(em) )
      case rel: RelationMention =>
        os.append( argumentsToString(rel) )
      case _ => ()
    }
    os.append(s"$boundary\n")

    os.toString()
  }

  def argumentsToString(b: Mention):String = {
    val os = new StringBuilder

    b.arguments foreach {
      case (argName, ms) =>
        ms foreach { v =>
          os.append(s"\t$argName ${v.labels.mkString("(", ", ", ")")} => ${v.text}\n")
        }
    }

    os.toString()
  }

  def getCausalArgsHack (b: Mention, filterContent:Boolean = false):(String, String) = {
    val controlled = b.arguments.get("controlled")
    val controller = b.arguments.get("controller")

    // todo: implement or remove
    ???

  }


  def causalArgumentsToTuple(b: Mention,
                             filterContent:Boolean = true,
                             collapseNE:Boolean = false,
                             view:String):(Array[String], Array[String], Array[String]) = {
    val controllerText = new ArrayBuffer[String]
    val controlledText = new ArrayBuffer[String]
    val exampleText = new ArrayBuffer[String]

    val args = b.arguments
    val controllerMentions = args.get("controller")
    val controlledMentions = args.get("controlled")
    val triggerMentions = args.get("trigger")
    val exampleMentions = args.get("example")

    if (controllerMentions.isDefined) {
      val mentions = collapseMentions(controllerMentions.get)
      controllerText.insertAll(0, mentions.map(m => getContentText(m, filterContent, collapseNE, view)))
    }

    if (controlledMentions.isDefined) {
      val mentions = collapseMentions(controlledMentions.get)
      controlledText.insertAll(0, mentions.map(m => getContentText(m, filterContent, collapseNE, view)))
    }

    if (exampleMentions.isDefined) {
      val mentions = collapseMentions(exampleMentions.get)
      exampleText.insertAll(0, mentions.map(m => getContentText(m, filterContent, collapseNE, view)))
    }

    (controllerText.toArray, controlledText.toArray, exampleText.toArray)
  }

  def getContentText(m: Mention, filterPOS:Boolean = true, collapseNE:Boolean = false, view:String): String = {
    val validWords = new ArrayBuffer[String]
    val words = m.words
    val lemmas = m.lemmas
    val tags = m.tags
    val tokens = view match {
      case "lemmas" => if (lemmas.isDefined) lemmas.get else throw new RuntimeException ("Error: lemmas not populated.")
      case "words" => words
      case "wordsWithTags" => {
        if (tags.isDefined) words.zip(tags.get).map(tup => s"${tup._1}_${tup._2}")
        else throw new RuntimeException ("Error: tags not populated.")
      }
      case "lemmasWithTags" => {
        if (lemmas.isDefined && tags.isDefined) lemmas.get.zip(tags.get).map(tup => s"${tup._1}_${tup._2}")
        else throw new RuntimeException ("Error: lemmas not populated.")
      }
      case _ => throw new RuntimeException("ERROR: unsupported view in getContentText --> " + view)
    }

    val entitiesOpt = m.entities

    if (tags.isEmpty) return m.text

    for (i <- words.indices) {
      if (isContentTag(tags.get(i)) || !filterPOS) {
        // If there are no NE tags, use word
        if (entitiesOpt.isEmpty || !collapseNE) validWords.append(tokens(i))
        else {
          // Get the entities
          val entities = entitiesOpt.get
          // If the word is a named entity (and we're collapsing), replace with the NE tag
          if (!Array("O", "MISC").contains(entities(i))) validWords.append(entities(i) + "_NER")
          else validWords.append(tokens(i))
        }
      }
    }

    validWords.mkString(" ")
  }

  def isContentTag(tag: String) = {
    // TODO: consider adding IN
    tag.startsWith("NN") || tag.startsWith("VB") || tag.startsWith("JJ") || tag.startsWith("RB")
  }



  // "the prevalence of lung cancer" => "lung cancer"
  // "cancer risk." => "cancer"
  def generateCaptionFromMention(m: Mention):String = {
    val s = m.sentenceObj
    val interval = m.tokenInterval
    val tags = s.tags.get
    val filteredTerms = for {
      i <- interval.start until interval.end
      lemma = s.lemmas.get(i).toLowerCase()
      tag = tags(i)
      // ensure that lemma is not a transparent noun
      if ! (Transparent contains lemma)
      // avoid a set of stopwords
      if ! (StopWords contains lemma)
      // avoid punctuation
      if ! lemma.matches(PUNCT)
      // ensure that tag is valid
      if tag.matches(VALIDTAG)
    } yield lemma
    // prepare the filtered terms for further filtering
    filteredTerms.mkString(" ")
  }


  // Filter out mentions which are completely contained within other mentions from the same sentence
  def collapseMentions (mentions:Seq[Mention]):Seq[Mention] = {
    val mentionsOut = new ArrayBuffer[Mention]
    val groupedBySentence = mentions.groupBy(m => m.sentence)
    for (sentence <- groupedBySentence.keySet) {
      val sentenceOut = new ArrayBuffer[Mention]
      val sentenceMentions = groupedBySentence.get(sentence).get
      val sorted = sentenceMentions.sortBy(- _.tokenInterval.length)
      for (m <- sorted) {
        var keep:Boolean = true
        for (alreadyFound <- sentenceOut) {
          if (alreadyFound.tokenInterval.contains(m.tokenInterval)) keep = false
        }
        if (keep) sentenceOut.append(m)
      }

      mentionsOut.insertAll(mentionsOut.length, sentenceOut)
    }

    mentionsOut
  }

  def sentenceToString(s:Sentence):String = {
    val sb = new StringBuilder

    sb.append("TEXT:\t" + s.getSentenceText() + "\n")
    sb.append("WORDS:\t" + s.words.mkString("\t") + "\n")
    sb.append("LEMMAS:\t" + s.lemmas.get.mkString("\t") + "\n")
    sb.append("TAGS:\t" + s.tags.get.mkString("\t") + "\n")
    //sb.append("CHUNKS:\t" + s.chunks.get.mkString("\t") + "\n")
    sb.append("NER:\t" + s.entities.get.mkString("\t") + "\n")
    sb.append("DEPS:\n" + s.dependencies.get.toString)

    sb.toString()
  }

}

class ArgumentSimple (val label: String, val text:String) {
  override def toString() = s"[$label]:$text"
}
