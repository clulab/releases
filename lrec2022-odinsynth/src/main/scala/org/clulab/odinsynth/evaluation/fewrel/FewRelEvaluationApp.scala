package org.clulab.odinsynth.evaluation.fewrel

import org.clulab.processors.fastnlp.FastNLPProcessor
import org.clulab.dynet.Utils.initializeDyNet
import org.clulab.odinsynth.evaluation.DocumentFromSentences
import ai.lum.odinson.TokensField
import org.clulab.odinsynth.evaluation.fstacred.SupportSentence
import org.clulab.odinsynth.evaluation.fstacred.FewShotSentence
import scala.io.Source
import org.clulab.odinsynth.{using, EnhancedType, EnhancedColl, Spec, TacredSearcher, SynthesizedRule, Parser}
import ujson.Value
import org.clulab.odinsynth.evaluation.fstacred.MultiQueryEpisode
import org.clulab.odinsynth.evaluation.fstacred.Episode
import scala.collection.mutable
import ai.lum.odinson.Document
import upickle.default.{ReadWriter => RW, macroRW, write, read}
import java.io.PrintWriter
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.ConcurrentHashMap
import org.clulab.odinsynth.evaluation.fstacred.FewShotTacredApp
import scala.collection.parallel.ForkJoinTaskSupport
import java.util.concurrent.ForkJoinPool
import scala.util.Random
import org.clulab.odinsynth.evaluation.tacred.PatternDirection
import org.clulab.odinsynth.evaluation.tacred.TacredPattern
import org.clulab.odinsynth.scorer.DynamicWeightScorer
import org.clulab.odinsynth.evaluation.fstacred.wordsOnSyntacticPathBetween
import ai.lum.odinson.ExtractorEngine
import org.clulab.odinsynth.evaluation.tacred.TacredEvaluation
import org.clulab.odinsynth.ConcatQuery
import org.clulab.dynet.ConstEmbeddingsGlove
import org.clulab.odinsynth.scorer.Scorer
import java.io.File


object FewRelEvaluationApp extends App {

  implicit val rwPD: RW[PatternDirection] = macroRW
  implicit val rwTP: RW[TacredPattern]    = macroRW

  val r = new Random(1)
    
  // val fakeSyntaxRules = loadExpandedSupportSentences("/data/nlp/corpora/few-rel/rules/211230/def_expanded_lessthan5_c=1_8_512_fakesyntax_250steps_1sols.pkl").toMap//.mapValues(_.take(args(1).toInt))
                                        // .mapValues(_.map { it => val query = Parser.parseBasicQuery(it.pattern).asInstanceOf[ConcatQuery]; it.copy(pattern=new ConcatQuery(query.queries.tail.init).pattern) }.filterNot(_.pattern.isEmpty()) )

  // val surfaceRules    = loadExpandedRules("/data/nlp/corpora/few-rel/rules/211230/def_expanded_lessthan5_c=1_8_512_surface_250steps_1sols.pkl").toMap.withDefaultValue(Seq.empty) //.mapValues(_.take(args(1).toInt))
                                        // .mapValues(_.map { it => val query = Parser.parseBasicQuery(it.pattern).asInstanceOf[ConcatQuery]; it.copy(pattern=new ConcatQuery(query.queries.tail.init).pattern) }.filterNot(_.pattern.isEmpty()) )


  // val episodes = loadSequenceOfEpisodes("/data/nlp/corpora/few-rel/data_processed/5_way_1_shots_10K_episodes_1q_seed_1_na_05_processed.json")


  // println(surfaceRules.size)
  simpleRun()

  def simpleRun(): Unit = {

    // val path   = args.head
    // val path   = "/data/nlp/corpora/fs-tacred/few-shot-dev/dev_episodes/5_way_1_shots_10K_episodes_3q_seed_160290.json"
    // val path   = "/data/nlp/corpora/fs-tacred/few-shot-dev/dev_episodes/5_way_5_shots_10K_episodes_3q_seed_160290.json"
    val cache  = "/data/nlp/corpora/fs-tacred/odinsynth_cache"
    val scorer = DynamicWeightScorer("http://localhost:8001")

    val p = {
      initializeDyNet()
      new FastNLPProcessor
    }

    val episodes = loadSequenceOfEpisodes("/data/nlp/corpora/few-rel/data_processed/5_way_1_shots_10K_episodes_1q_seed_1_na_05_processed.json")
    
    val fakeSyntaxRules = loadMultiRulesPerSentenceFromFile("/data/nlp/corpora/few-rel/rules/train_c=1_8_512_fakesyntax_1000steps_5sols.pkl").toMap//.mapValues(_.take(args(1).toInt))
                                          // .mapValues(_.map { it => val query = Parser.parseBasicQuery(it.pattern).asInstanceOf[ConcatQuery]; it.copy(pattern=new ConcatQuery(query.queries.tail.init).pattern) }.filterNot(_.pattern.isEmpty()) )

    val surfaceRules    = loadMultiRulesPerSentenceFromFile("/data/nlp/corpora/few-rel/rules/train_c=1_8_512_surface_1000steps_5sols.pkl").toMap//.withDefaultValue(Seq.empty) //.mapValues(_.take(args(1).toInt))
                                          // .mapValues(_.map { it => val query = Parser.parseBasicQuery(it.pattern).asInstanceOf[ConcatQuery]; it.copy(pattern=new ConcatQuery(query.queries.tail.init).pattern) }.filterNot(_.pattern.isEmpty()) )

    val selectedSs = r.shuffle(episodes.flatMap(_.supportSentences)).take(50).toSet
    val q1 = surfaceRules.toSeq.flatMap { case (ss, tp) => tp.map { it => (ss, it) } }.filter(it => selectedSs.contains(it._1))
    val q2 = fakeSyntaxRules.toSeq.flatMap { case (ss, tp) => tp.map { it => (ss, it) } }.filter(it => selectedSs.contains(it._1))
    val pw1 = new PrintWriter(new File("z_surface"))
    pw1.println("sentence\thighlighted\tpattern")
    q1.foreach { case (ss, tp) =>
      pw1.println(f"${ss.sentence.tokens.mkString(" ")}\t${ss.sentence.getHighlightedPartWithTypes().mkString(" ")}\t${tp.pattern}")
    }
    pw1.close()

    val pw2 = new PrintWriter(new File("z_fakesyntax"))
    pw2.println("sentence\thighlighted\twords_on_syntax\tpattern")
    q2.foreach { case (ss, tp) =>
      val doc = DocumentFromSentences.documentFromSentencesAndCache(Seq(ss.sentence.getSentenceWithTypes), p, cache)  
      val newWords = wordsOnSyntacticPathBetween(doc, ss.sentence.getFirstType, ss.sentence.getSecondType)

      pw2.println(f"${ss.sentence.tokens.mkString(" ")}\t${ss.sentence.getHighlightedPartWithTypes().mkString(" ")}\t${newWords.mkString(" ")}\t${tp.pattern}")
    }
    pw2.close()
    System.exit(1)
                                          
    println(
      f"""
      |Fake syntax rules: ${fakeSyntaxRules.toSeq.flatMap(_._2).size}
      |Surface rules    : ${surfaceRules.toSeq.flatMap(_._2).size}
      """.stripMargin
    )
    val nr = new AtomicInteger()
    val nr2 = new AtomicInteger()
    val nr3 = new AtomicInteger()
    val episodeResults = episodes
                            // .par.let{it => it.tasksupport = new ForkJoinTaskSupport(new ForkJoinPool(32)); it }
                            .map { case Episode(supportSentences, querySentence, queryRelation) =>

      val episodeRelations = supportSentences.map(_.relation).toSet
 // val rules = supportSentences.flatMap { x => surfaceRules   (x) } ++ surfaceTrainRules

      val (rules1, ee1, spec1) = {
        val rules = supportSentences.flatMap { x => 
          val rules = fakeSyntaxRules(x)
          if (x.relation == queryRelation) {
            rules.map { it =>
              val query = new ConcatQuery(Parser.parseBasicQuery(it.pattern).asInstanceOf[ConcatQuery].queries.tail.init)
              it.copy(pattern=f"[word=${querySentence.getFirstType}] $query [word=${querySentence.getSecondType}]")
            }.filterNot(_.pattern.isEmpty())
          } else {
            rules
          }
          rules
        } //++ fakeSyntaxTrainRules.getOrElse((querySentence.getFirstType, querySentence.getSecondType), Seq.empty)
        val doc = DocumentFromSentences.documentFromSentencesAndCache(Seq(querySentence.getSentenceWithTypes), p, cache)
        
        val newWords = wordsOnSyntacticPathBetween(doc, querySentence.getFirstType, querySentence.getSecondType)
        val newDoc = DocumentFromSentences.documentFromSentencesAndCache(Seq(newWords), p, cache)

        val ee = ExtractorEngine.inMemory(newDoc)
        (rules, ee, Spec(newDoc.id, 0, 0, newWords.length))
      }

      val (rules2, ee2, spec2) = {
        val rules = supportSentences.flatMap { x => 
          val rules = surfaceRules(x)
          rules
        } //++ surfaceTrainRules.getOrElse((querySentence.getFirstType, querySentence.getSecondType), Seq.empty)
        val (tokens, spec) = querySentence.getSentenceWithTypesAndSpecWithTypes(0)
        val doc = DocumentFromSentences.documentFromSentencesAndCache(Seq(querySentence.getSentenceWithTypes), p, cache)
        val ee = ExtractorEngine.inMemory(doc)
        (rules, ee, spec.copy(docId = doc.id))
      }
      val q = supportSentences.groupBy(_.relation)
      if (q.contains(queryRelation) && q(queryRelation).flatMap { x => fakeSyntaxRules(x) }.size == 0 && q(queryRelation).flatMap { x => surfaceRules(x) }.size == 0) {
        nr2.getAndIncrement()
      }
      val q2 = supportSentences.filter(_.relation == queryRelation).map(it => (it.sentence.subjType, it.sentence.objType))
      if (q2.contains((querySentence.subjType, querySentence.objType))) {
        nr3.getAndIncrement()
      }

      // println(rules1.size)
      // println(rules2.size)
      // System.exit(1)


      val result = {
        val tempResult = TacredEvaluation.predictRelations(rules2, ee2, spec2, querySentence.getDirectionality)
        tempResult.groupBy(_._1).mapValues(_.map(_._2).sum)
      }.toSeq

      val prediction = {
        if(result.nonEmpty) {
          // println(f"${result} - $queryRelation ${q.getOrElse(queryRelation, Seq.empty)} ${q.getOrElse(queryRelation, Seq.empty).flatMap { x => fakeSyntaxRules(x) }} ${q.getOrElse(queryRelation, Seq.empty).flatMap { x => surfaceRules(x) }}")
          val maxValue = result.maxBy(_._2)
          val maxEntries = result.groupBy(_._2)
          r.shuffle(maxEntries.maxBy(_._1)._2).head._1
        } else {
          nr.getAndIncrement()
          "no_relation"
        }
      }
      val goldRelation = if(episodeRelations.contains(queryRelation)) queryRelation else "no_relation"

      // if (prediction == goldRelation) {
      //   println("-"*100)
      //   println(f"\t${queryRelation}; (${querySentence.subjStart}, ${querySentence.subjEnd}, ${querySentence.getSubjTokens.mkString(" ")}, ${querySentence.subjType}); (${querySentence.objStart}, ${querySentence.objEnd}, ${querySentence.getObjTokens.mkString(" ")}, ${querySentence.objType}); ${querySentence.tokens.mkString(" ")}".replace("'", "").replace("\"", ""))
      //   supportSentences.foreach { it => println(f"\t\t${it.relation}; (${it.sentence.subjStart}, ${it.sentence.subjEnd}, ${it.sentence.getSubjTokens.mkString(" ")}, ${it.sentence.subjType}); (${it.sentence.objStart}, ${it.sentence.objEnd}, ${it.sentence.getObjTokens.mkString(" ")}, ${it.sentence.objType}); ${it.sentence.tokens.mkString(" ")}".replace("'", "").replace("\"", "")) }
      //   println(f"\t${rules1.toString().replace("'", "_q1_").replace("\"", "_q2_")}")
      //   println(f"\t${rules2.toString().replace("'", "_q1_").replace("\"", "_q2_")}")
      //   println(
      //     f"""
      //     | Our prediction: ${prediction}
      //     | Rules for gold (${goldRelation} - ${queryRelation}):
      //     |     Fake syntax ${rules1.filter(_.relation == queryRelation).map(_.pattern)}
      //     |     Surface     ${rules2.filter(_.relation == queryRelation).map(_.pattern)}
      //     | We have the following relation-rules: 
      //     |     Fake syntax ${rules1.groupBy(_.relation).map(it => (it._1, it._2.size))}
      //     |     Surface     ${rules2.groupBy(_.relation).map(it => (it._1, it._2.size))}
      //     """.stripMargin.replace("'", "_q1_").replace("\"", "_q2_")
      //     )
      //   println("-"*100)
      //   println(" ")
      // }
      (goldRelation, prediction)

    }.toIndexedSeq
    println(nr)
    println(nr2)
    println(nr3)
    println("$"*100)
    println(episodeResults.count(it => it._1 == "no_relation"))
    // println(microScores(episodeResults, ignoreLabels = Set("no_relation")))
    println(episodeResults.count(it => it._1 == it._2))
    println(episodeResults.count(it => it._1 != it._2))
    println("-"*100)
    println(episodeResults.size)
    println(episodeResults.count(it => it._1 == it._2) / episodeResults.size.toDouble)
    println(episodeResults.count(it => it._1 == it._2) / (episodeResults.count(it => it._1 == it._2) + episodeResults.count(it => it._1 != it._2)).toDouble)
    println(FewShotTacredApp.microScores(episodeResults, ignoreLabels = Set.empty))
    println("#"*100)
    println(FewShotTacredApp.microScores(episodeResults, ignoreLabels = Set("no_relation")))
  }



  def simpleRunWithExpansions(): Unit = {

    // val path   = args.head
    // val path   = "/data/nlp/corpora/fs-tacred/few-shot-dev/dev_episodes/5_way_1_shots_10K_episodes_3q_seed_160290.json"
    // val path   = "/data/nlp/corpora/fs-tacred/few-shot-dev/dev_episodes/5_way_5_shots_10K_episodes_3q_seed_160290.json"
    val cache  = "/data/nlp/corpora/fs-tacred/odinsynth_cache"
    val scorer = DynamicWeightScorer("")

    val p = {
      initializeDyNet()
      new FastNLPProcessor
    }

    val episodes   = loadSequenceOfEpisodes("/data/nlp/corpora/few-rel/data_processed/5_way_1_shots_10K_episodes_1q_seed_1_na_05_processed.json")

    val expandedSs = loadExpandedSupportSentences("/home/rvacareanu/projects/temp/odinsynth_specexpander/specexpander/data/data.jsonl").toMap

    val fakeSyntaxRules = loadExpandedRules("/data/nlp/corpora/few-rel/rules/211230/def_expanded_lessthan5_c=1_8_512_fakesyntax_250steps_1sols.pkl").toMap.withDefaultValue(Seq.empty) //.mapValues(_.take(args(1).toInt))
    //                                       // .mapValues(_.map { it => val query = Parser.parseBasicQuery(it.pattern).asInstanceOf[ConcatQuery]; it.copy(pattern=new ConcatQuery(query.queries.tail.init).pattern) }.filterNot(_.pattern.isEmpty()) )

    val surfaceRules    = loadExpandedRules("/data/nlp/corpora/few-rel/rules/211230/def_expanded_lessthan5_c=1_8_512_surface_250steps_1sols.pkl").toMap.withDefaultValue(Seq.empty) //.mapValues(_.take(args(1).toInt))
                                          // .mapValues(_.map { it => val query = Parser.parseBasicQuery(it.pattern).asInstanceOf[ConcatQuery]; it.copy(pattern=new ConcatQuery(query.queries.tail.init).pattern) }.filterNot(_.pattern.isEmpty()) )
    val episodeResults = episodes
                            // .filterNot(_.querySentence.getHighlightedPart().size > 5).par
                            .par
                            // .par.let{it => it.tasksupport = new ForkJoinTaskSupport(new ForkJoinPool(32)); it }
                            .map { case Episode(supportSentences, querySentence, queryRelation) =>

      val episodeRelations = supportSentences.map(_.relation).toSet

      val expandedSupportSentences = supportSentences.flatMap { it => expandedSs(it.sentence).map { fss => SupportSentence(it.relation, fss) } }

      

      val (rules1, ee1, spec1) = {
        val rules = expandedSupportSentences.flatMap { x => 
          val rules = fakeSyntaxRules(x)
          if (x.relation == queryRelation) {
            rules.map { it =>
              val query = new ConcatQuery(Parser.parseBasicQuery(it.pattern).asInstanceOf[ConcatQuery].queries.tail.init)
              it.copy(pattern=f"[word=${querySentence.getFirstType}] $query [word=${querySentence.getSecondType}]")
            }.filterNot(_.pattern.isEmpty())
          } else {
            rules
          }
          rules
        } //++ fakeSyntaxTrainRules.getOrElse((querySentence.getFirstType, querySentence.getSecondType), Seq.empty)
        val doc = DocumentFromSentences.documentFromSentencesAndCache(Seq(querySentence.getSentenceWithTypes), p, cache)
        
        val newWords = wordsOnSyntacticPathBetween(doc, querySentence.getFirstType, querySentence.getSecondType)
        val newDoc = DocumentFromSentences.documentFromSentencesAndCache(Seq(newWords), p, cache)

        val ee = ExtractorEngine.inMemory(newDoc)
        (rules, ee, Spec(newDoc.id, 0, 0, newWords.length))
      }


      val (rules2, ee2, spec2) = {
        val rules = expandedSupportSentences.flatMap { x => 
          val rules = surfaceRules(x)
          rules
        } //++ surfaceTrainRules.getOrElse((querySentence.getFirstType, querySentence.getSecondType), Seq.empty)
        val (tokens, spec) = querySentence.getSentenceWithTypesAndSpecWithTypes(0)
        val doc = DocumentFromSentences.documentFromSentencesAndCache(Seq(querySentence.getSentenceWithTypes), p, cache)
        val ee  = ExtractorEngine.inMemory(doc)
        (rules, ee, spec.copy(docId = doc.id))
      }

      val result = {
        val tempResult = TacredEvaluation.predictRelations(rules2, ee2, spec2, querySentence.getDirectionality) ++ TacredEvaluation.predictRelations(rules1, ee1, spec1, querySentence.getDirectionality)
        tempResult.groupBy(_._1).mapValues(_.map(_._2).sum)
      }.toSeq

      val prediction = {
        if(result.nonEmpty) {
          // println(f"${result} - $queryRelation ${q.getOrElse(queryRelation, Seq.empty)} ${q.getOrElse(queryRelation, Seq.empty).flatMap { x => fakeSyntaxRules(x) }} ${q.getOrElse(queryRelation, Seq.empty).flatMap { x => surfaceRules(x) }}")
          val maxValue = result.maxBy(_._2)
          val maxEntries = result.groupBy(_._2)
          r.shuffle(maxEntries.maxBy(_._1)._2).head._1
        } else {
          "no_relation"
        }
      }
      val goldRelation = if(episodeRelations.contains(queryRelation)) queryRelation else "no_relation"


      (goldRelation, prediction)

    }.toIndexedSeq

    println(episodeResults.count(_._1 == "no_relation"))
    println(episodeResults.size)
    println(episodeResults.count(it => it._1 == it._2) / episodeResults.size.toDouble)
    // println(episodeResults.count(it => it._1 == it._2) / (episodeResults.count(it => it._1 == it._2) + episodeResults.count(it => it._1 != it._2)).toDouble)
    println(FewShotTacredApp.microScores(episodeResults, ignoreLabels = Set.empty))
    println("#"*100)
    println(FewShotTacredApp.microScores(episodeResults, ignoreLabels = Set("no_relation")))
    println("$"*100)
    println(episodeResults.count(_._1 == "no_relation"))
    println(episodeResults.size)
    println(episodeResults.count(it => it._1 == it._2) / episodeResults.size.toDouble)

  }


  def loadMultiRulesPerSentenceFromFile(loadpath: String): Seq[(SupportSentence, Seq[TacredPattern])] = {
    using(Source.fromFile(loadpath)) { it => 
      it.getLines().toList.map { it =>
        val string = it.mkString
        read[(SupportSentence, Seq[TacredPattern])](string)
      }
    }
  }

  def loadExpandedRules(path: String): Seq[(SupportSentence, Seq[TacredPattern])] = {
    using(Source.fromFile(path)) { it =>
      it.getLines().toList.map { line =>
        read[(SupportSentence, Seq[TacredPattern])](line)
      }.toSeq
    }
  }

  def loadSequenceOfEpisodes(path: String): Seq[Episode] = {
    implicit val rw: RW[Episode] = macroRW
    using(Source.fromFile(path)) { it =>
      it.getLines().toList.map { line =>
        read[Episode](line)
      }.toSeq
    }
  }

  def loadExpandedSupportSentences(path: String): Seq[(FewShotSentence, Seq[FewShotSentence])] = {
    using(Source.fromFile(path)) { it =>
      it.getLines().toList.map { line =>
        read[(FewShotSentence, Seq[FewShotSentence])](line)
      }.toSeq
    }
  }


}
