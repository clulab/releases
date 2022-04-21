package org.clulab.odinsynth.evaluation.fstacred

import java.io.{File, PrintWriter}
import java.util.concurrent.ForkJoinPool
import java.util.concurrent.atomic.AtomicInteger

import scala.io.Source
import scala.collection.mutable
import scala.collection.parallel.ForkJoinTaskSupport
import scala.util.Try

import ujson.Js
import ujson.Value
import upickle.default.{ReadWriter => RW, macroRW, write, read}

import ai.lum.odinson.{ExtractorEngine, Document, TokensField, GraphField}

import org.clulab.processors.fastnlp.FastNLPProcessor
import org.clulab.dynet.Utils.initializeDyNet
import org.clulab.odinsynth.evaluation.DocumentFromSentences
import org.clulab.odinsynth.evaluation.PandasLikeDataset
import org.clulab.odinsynth.{using, EnhancedType, EnhancedColl, Spec, TacredSearcher, SynthesizedRule, Parser}
import org.clulab.odinsynth.scorer.{Scorer, DynamicWeightScorer}
import org.clulab.utils.MathUtils.EnhancedNumericCollection
import org.clulab.odinsynth.evaluation.tacred.{TacredEvaluation, PatternDirection, LogSpecSizeWeight, TacredPattern, SubjObjDirection, ObjSubjDirection}
import org.clulab.odinsynth.{ConcatQuery, RepeatQuery, generalizedCross}
import scala.util.Random

/**
  * An application to work with the json files from the Few-Shot TACRED dataset
  */
object FewShotTacredApp extends App {

  implicit val rwPD: RW[PatternDirection] = macroRW
  implicit val rwTP: RW[TacredPattern]    = macroRW

  run()

  def run(): Unit = {
    val path   = "few_shot_data.json"
    val cache  = "/data/nlp/corpora/fs-tacred/odinsynth_cache"
    val scorer = DynamicWeightScorer("http://localhost:8001")

    val p = {
      initializeDyNet()
      new FastNLPProcessor
    }

    val episodes = readTrainTestEpisodes(path).flatMap(_.unroll)
    // val qwe = episodes.filter(_.querySentence.tokens.mkString(" ").contains("Kit Yarrow , 50 , a professor of psychology"))
    // qwe.map(_.querySentence).foreach(println)
    // System.exit(1)
    // println(episodes.filterNot(it => it.supportSentences.map(_.relation).contains(it.queryRelation) || it.queryRelation == "no_relation").size)
    // println(episodes.size)

    // System.exit(1)
    // val cacheRules = mutable.Map.empty[String, Pattern]
    val supportSentenceToRules = loadRulesFromFile("rules.pkl").toMap
    // val supportSentenceToRules = loadMultiRulesPerSentenceFromFile("/data/nlp/corpora/fs-tacred/odinsynth_rule_cache/dev_metatrain_c=25_8_512_dev.pkl").toMap
    
    // val trainRules = loadTrainRules().map(_.copy(relation="no_relation"))
    // (0.844,0.057,0.107) with 
    val trainRules = loadSequenceOfRules("train_rules").flatMap(_._2).flatMap { it => 
      val rule = Parser.parseBasicQuery(it.pattern)
      val cq = rule.asInstanceOf[ConcatQuery]
      val q1 = ConcatQuery(cq.queries.reverse)
      // val q2 = ConcatQuery(Vector(cq.queries.last) ++ cq.queries.tail.init ++ Vector(cq.queries.head))

      val q1d = if (it.direction == SubjObjDirection) ObjSubjDirection else SubjObjDirection

      Seq(it.copy(relation="no_relation", direction=it.direction), it.copy(pattern = q1.pattern,relation="no_relation", direction=q1d)) 
    }
    val episodeResults = episodes
                            .par.let{it => it.tasksupport = new ForkJoinTaskSupport(new ForkJoinPool(32)); it }
                            .map { case Episode(supportSentences, querySentence, queryRelation) =>
      val episodeRelations = supportSentences.map(_.relation).toSet
      // val simpleRulesChange = supportSentences.flatMap { supportSentence =>
      //   if (supportSentence.sentence.getHighlightedPart().size == 1) {
      //     val (tokens, spec) = supportSentence.sentence.getSentenceWithTypesAndSpecWithoutTypes(0)
      //     val doc        = DocumentFromSentences.documentFromSentencesAndCache(Seq(tokens), p, cache)
      //     val firstType  = Parser.parseBasicQuery(f"[word=${supportSentence.sentence.getFirstType.toLowerCase()}]")
      //     val secondType = Parser.parseBasicQuery(f"[word=${supportSentence.sentence.getSecondType.toLowerCase()}]")
      //     val searcher   = new TacredSearcher(Seq(doc), Set(spec.copy(docId = doc.id)), Set("tag"), Some(10000), None, scorer, false, firstType, secondType)
      //     val rule       = searcher.findFirst()

      //     Some(TacredPattern(rule.get.rule.pattern, supportSentence.relation, supportSentence.sentence.getDirectionality, 1.0))
      //   } else {
      //     supportSentenceToRules(supportSentence).map { it => TacredPattern(it.rule, it.relation, supportSentence.sentence.getDirectionality, 1.0) }
      //   }
      // }

      // NOTE start
      // Such an explicit call instead of supportSentences.flatMap(supportSentencesToRules) is needed
      // See https://stackoverflow.com/a/37726788 and https://stackoverflow.com/a/37726987
      // In short, calling it explicitly makes use of an implicit function (option2Iterable)
      // If you don't call it explicitly you need:
      //    SupportSentence => GenTraversableOnce[Pattern]
      // but you have the map (something like SupportSentence => Option[Pattern])
      // So you would need an implicit conversion from SupportSentence => Option[Pattern] to SupportSentence => GenTraversableOnce[Pattern]
      // which is not defined
      // You can define it:
      //    implicit def customConversion(f: SupportSentence => Option[Pattern]): SupportSentence => GenTraversableOnce[Pattern] = ss => Option.option2Iterable(f(ss))
      // And then it will work
      // val rules = supportSentences.flatMap(supportSentenceToRules)
      // NOTE end
      val rules = supportSentences.flatMap { x => 
        supportSentenceToRules(x).map { it => 
          // if (x.sentence.getFirstType == x.sentence.getSecondType) {
            // Seq(
              // TacredPattern(it.rule, it.relation, SubjObjDirection, 1.0),
              // TacredPattern(it.rule, it.relation, ObjSubjDirection, 1.0),
            // )
          // } else {
            // Seq(TacredPattern(it.rule, it.relation, x.sentence.getDirectionality, 1.0))
          // }

          TacredPattern(it.rule, it.relation, x.sentence.getDirectionality, 1.0)
        } 
      }.flatMap { it => 
        val rule = Parser.parseBasicQuery(it.pattern)
        val cq = rule.asInstanceOf[ConcatQuery]
        val q1 = ConcatQuery(cq.queries.reverse)
        // val q2 = ConcatQuery(Vector(cq.queries.last) ++ cq.queries.tail.init ++ Vector(cq.queries.head))

        val q1d = if (it.direction == SubjObjDirection) ObjSubjDirection else SubjObjDirection

        Seq(it, it.copy(pattern=q1.pattern, direction=q1d))
       } ++ trainRules
      

      val (queryTokens, querySpec) = querySentence.getSentenceWithTypesAndSpecWithTypes(0)
      val ee = ExtractorEngine.inMemory(DocumentFromSentences.documentFromSentencesAndCache(Seq(queryTokens), p, cache))

      val result = TacredEvaluation.predictRelations(rules, ee, querySpec, querySentence.getDirectionality)
      val prediction = if(result.nonEmpty) result.maxBy(_._2)._1 else "no_relation"
      val goldRelation = if(episodeRelations.contains(queryRelation)) queryRelation else "no_relation"
      // val goldRelation = queryRelation // if(episodeRelations.contains(queryRelation)) queryRelation else "no_relation"
      val debugString = f"""
      |{
      |\t"
      |\tQuery
      |\tTokens:       ${querySentence.tokens.mkString(" ")}
      |\tS with types: ${querySentence.getSentenceWithTypesAndSpecWithoutTypes(0)._1.mkString(" ")}
      |\tFirst Type:   ${querySentence.getFirstType}
      |\tSecond Type:  ${querySentence.getSecondType}
      |\tDirection:    ${querySentence.getDirectionality}
      |\tPrediction:   ${result.groupBy(_._1).mapValues(_.size).toSeq}
      |\tRelation:     ${queryRelation} - ${goldRelation}\n
      |\tS Relations:  ${episodeRelations.toSeq.sorted}
      |\tSupport sentences and rules
      |${supportSentences.map(it => f"\tTokens:       ${it.sentence.tokens.mkString(" ")}\n\tS with types: ${it.sentence.getSentenceWithTypesAndSpecWithTypes(0)._1.mkString(" ")}\n\tFirstType:    ${it.sentence.getFirstType}\n\tSecondType:   ${it.sentence.getSecondType}\n\tDirection:    ${it.sentence.getDirectionality}\n\tRule:         ${supportSentenceToRules(it).map { pattern => f"${pattern.rule}" }}\n\tRelation:     ${it.relation}").mkString(sep="\n\n")}
      |\t"
      |}\n\n\n\n
      """.stripMargin

      (goldRelation, prediction)
      
      // if(result.nonEmpty) {
      //   if (queryRelation != "no_relation") {
      //     println(f"${queryRelation} ('${result.maxBy(_._2)._1}') - ${querySentence.getFirstType};${querySentence.getSecondType} (${supportSentences.map(it => f"${it.sentence.getFirstType}; ${it.sentence.getSecondType}").mkString(sep="  ")})")
      //   }
      //   (queryRelation, result.maxBy(_._2)._1)
      // } else {
      //   if (queryRelation != "no_relation") {
      //     println(f"${queryRelation} ('no_relation') - ${querySentence.getFirstType};${querySentence.getSecondType} (${supportSentences.map(it => f"${it.sentence.getFirstType};${it.sentence.getSecondType}").mkString(sep="  ")})")
      //   }
      //   (queryRelation, "no_relation")
      // }
    }.toIndexedSeq

    println(microScores(episodeResults, ignoreLabels = Set("no_relation")))

  }

  /**
    * Performance (5-way 1-shot): (P, R, F1) = (0.0577,0.9,0.1084)
    * 
    * The idea of this baseline is to check the supporting sentences for sentences where the entity types
    * are identical with the entity types of the query sentence
    * 
    * Then, randomly sample from the resulting list of relations.
    * 
    */
  def baselineRun(): Unit = {
    import scala.util.Random
    val random = new Random(1)
    val path   = "/data/nlp/corpora/fs-tacred/few-shot-dev/dev_episodes/5_way_1_shots_10K_episodes_3q_seed_160290.json"
    val episodes = readTrainTestEpisodes(path).flatMap(_.unroll)

    val episodeResults = episodes
                            .par.let{it => it.tasksupport = new ForkJoinTaskSupport(new ForkJoinPool(32)); it }
                            .map { case Episode(supportSentences, querySentence, queryRelation) =>
      val episodeRelations = supportSentences.map(_.relation).toSet
                            
      val entitiesToRelation = supportSentences
                                          .flatMap { it => Seq(
                                            ((it.sentence.getFirstType, it.sentence.getSecondType), it.relation),
                                            ((it.sentence.getSecondType, it.sentence.getFirstType), it.relation),
                                          ) }
                                          .groupBy(_._1).mapValues(_.map(_._2))
                                          .withDefaultValue(Seq("no_relation")) //.toMap.withDefaultValue("no_relation")
                                          

      val e = (querySentence.getFirstType, querySentence.getSecondType)
      val prediction = random.shuffle(entitiesToRelation(e)).head

      
      val goldRelation = if(episodeRelations.contains(queryRelation)) queryRelation else "no_relation"

      (goldRelation, prediction)
    }.toIndexedSeq

    println(microScores(episodeResults, ignoreLabels = Set("no_relation")))

  }

  /**
    * Performance (5-way 1-shot): (P, R, F1) = (0.0616,0.63,0.1122)
    * 
    * The difference bewteen this and baselineRun is that we use
    * the train data as well
    * We use a map between the entity types encountered in train and their relation
    * 
    * When predicting:
    *   (1) we take the relations associated with (Entity1, Entity2) in the support sentence
    *   (2) we take the relations associated with (Entity1, Entity2) in the support sentence in the training sentences
    *       and relabel them as "no_relation". 
    * 
    * Finally, for prediction, we sample from a list constructed as follows:
    *   (a) all the relations resulting from (1)
    *   (b) only one "no_relation" resulting from (2), if it is not empty
    * 
    * Note: In (b) we do not take all because they might overwhelm the rest of the relations. However, we noticed that
    * using a number of log10 of the size of the list resulting in (2) as "no_relation" pushes the performance to: (P, R, F1) = (0.0926,0.4657,0.1545)
    * And using a number of log results in: (P, R, F1) = (0.1294,0.3828,0.1934)
    * 
    * 
    */
  def betterBaselineRun(): Unit = {
    import scala.util.Random
    val random = new Random(1)
    val path   = "/data/nlp/corpora/fs-tacred/few-shot-dev/dev_episodes/5_way_1_shots_10K_episodes_3q_seed_160290.json"
    val episodes = readTrainTestEpisodes(path).flatMap(_.unroll)

    val allTrainEpisodes = readUnrolledTestEpisodes("/data/nlp/corpora/fs-tacred/few-shot-dev/train_episodes/5_way_1_shots_10K_episodes_3q_seed_160290.json") ++
                 readUnrolledTestEpisodes("/data/nlp/corpora/fs-tacred/few-shot-dev/train_episodes/5_way_1_shots_10K_episodes_3q_seed_160291.json") ++ 
                 readUnrolledTestEpisodes("/data/nlp/corpora/fs-tacred/few-shot-dev/train_episodes/5_way_1_shots_10K_episodes_3q_seed_160292.json") ++ 
                 readUnrolledTestEpisodes("/data/nlp/corpora/fs-tacred/few-shot-dev/train_episodes/5_way_1_shots_10K_episodes_3q_seed_160293.json") ++ 
                 readUnrolledTestEpisodes("/data/nlp/corpora/fs-tacred/few-shot-dev/train_episodes/5_way_1_shots_10K_episodes_3q_seed_160294.json") ++ 
                 readUnrolledTestEpisodes("/data/nlp/corpora/fs-tacred/few-shot-dev/train_episodes/5_way_5_shots_10K_episodes_3q_seed_160290.json") ++ 
                 readUnrolledTestEpisodes("/data/nlp/corpora/fs-tacred/few-shot-dev/train_episodes/5_way_5_shots_10K_episodes_3q_seed_160291.json") ++ 
                 readUnrolledTestEpisodes("/data/nlp/corpora/fs-tacred/few-shot-dev/train_episodes/5_way_5_shots_10K_episodes_3q_seed_160292.json") ++ 
                 readUnrolledTestEpisodes("/data/nlp/corpora/fs-tacred/few-shot-dev/train_episodes/5_way_5_shots_10K_episodes_3q_seed_160293.json") ++ 
                 readUnrolledTestEpisodes("/data/nlp/corpora/fs-tacred/few-shot-dev/train_episodes/5_way_5_shots_10K_episodes_3q_seed_160294.json")

    val trainEpisodes = allTrainEpisodes.flatMap(_.supportSentences).map(it => (it.sentence.getFirstType, it.sentence.getSecondType, it.relation)).groupBy(it => (it._1, it._2)).mapValues(_.map(_._3)).withDefaultValue(Seq.empty[String])


    val episodeResults = episodes
                            .par.let{it => it.tasksupport = new ForkJoinTaskSupport(new ForkJoinPool(32)); it }
                            .map { case Episode(supportSentences, querySentence, queryRelation) =>
      val episodeRelations = supportSentences.map(_.relation).toSet
                            
      val entitiesToRelation = supportSentences
                                          .flatMap { it => Seq(
                                            ((it.sentence.getFirstType, it.sentence.getSecondType), it.relation),
                                            ((it.sentence.getSecondType, it.sentence.getFirstType), it.relation),
                                          ) }
                                          .groupBy(_._1).mapValues(_.map(_._2))
                                          .withDefaultValue(Seq("no_relation")) //.toMap.withDefaultValue("no_relation")
                                          

      val e = (querySentence.getFirstType, querySentence.getSecondType)
      val prediction = random.shuffle(entitiesToRelation(e) ++ trainEpisodes(e).map(it => "no_relation").take(1)).head
      // val prediction = random.shuffle(entitiesToRelation(e) ++ trainEpisodes(e).map(it => "no_relation").take(math.log10(trainEpisodes(e).size).toInt)).head

      
      val goldRelation = if(episodeRelations.contains(queryRelation)) queryRelation else "no_relation"

      (goldRelation, prediction)
    }.toIndexedSeq

    println(microScores(episodeResults, ignoreLabels = Set("no_relation")))
    
    
  }

  /**
    * Simple baseline where we predict one random relation out of the relations
    * associated with the support sentences + "no_relation"
    */
  def randomPredictionBaselineRun(): Unit = {
    import scala.util.Random
    val random = new Random(1)
    val path   = "/data/nlp/corpora/fs-tacred/few-shot-dev/dev_episodes/5_way_1_shots_10K_episodes_3q_seed_160290.json"
    val episodes = readTrainTestEpisodes(path).flatMap(_.unroll)

    val episodeResults = episodes
                            .par.let{it => it.tasksupport = new ForkJoinTaskSupport(new ForkJoinPool(32)); it }
                            .map { case Episode(supportSentences, querySentence, queryRelation) =>
      val episodeRelations = (supportSentences.map(_.relation) ++ Seq("no_relation")).distinct
                                                                      
      val prediction = random.shuffle(episodeRelations).head

      
      val goldRelation = if(episodeRelations.contains(queryRelation)) queryRelation else "no_relation"

      (goldRelation, prediction)
    }.toIndexedSeq

    println(microScores(episodeResults, ignoreLabels = Set("no_relation")))

  }

  /**
    * This run is different than "run" because we load the rules that were generated after we 
    * clustered the 5 support sentences per relation for each episode
    * 
    * To speed things up, we only consider "unique" clusters. As a consequence, we need to also 
    * load a "mapping" file, which tells us which unique clusters that this initial cluster
    * corrsesponds to.
    * 
    * Additionally, we need to resolve the episode for which that cluster was created. This can 
    * be done from the paths (e.g. "<..>/clusters_0/df_ep0/1.org_founded/cluster_r0_1")
    * 
    */
  def runWithClusters(): Unit = {    
    val weightingScheme = LogSpecSizeWeight
    val rules = PandasLikeDataset("clusters_0_unique_all_solutions_copy.tsv", index=false).let { pld =>
      pld.filter { it => Try(Parser.parseBasicQuery(TacredEvaluation.handleQuotesInPatterns(it("pattern")))).isSuccess }//.let { it => println(it.length()); it.map(_("pattern")).take(20).foreach(println); System.exit(1); it }
        .map { it => 
          val cluster = PandasLikeDataset(it("cluster_path"))
          val subjType = cluster.lines.map(it => it("subj_type").toLowerCase()).distinct.map(it => f"word=$it").mkString(start="[", sep=" | ", end = "]")
          val objType = cluster.lines.map(it => it("obj_type").toLowerCase()).distinct.map(it => f"word=$it").mkString(start="[", sep=" | ", end = "]")                              
          val direction = PatternDirection.fromIntValue(it("direction").toInt)
          
          val pattern = it("pattern")
          (it("cluster_path"), TacredPattern(pattern, it("relation"), PatternDirection.fromIntValue(it("direction").toInt), weightingScheme.weight(it)))
      }
    }.toMap
    
    val individualRules = loadRulesFromFile("rules_dev.pkl").toMap

    val path   = "few_shot_data.json"
    val cache  = "/data/nlp/corpora/fs-tacred/odinsynth_cache"

    val p = {
      initializeDyNet()
      new FastNLPProcessor
    }

    val episodes = readTrainTestEpisodes(path).zipWithIndex.flatMap { case (mqe, idx) => mqe.unroll.map { e => (e, idx) } }
    val originalClusterSeqMap = using(Source.fromFile(f"unique_identical_clusters_paths")) { it => 
      val string = it.mkString
      val json = ujson.read(string)//.asInstanceOf[Map[String, String]]
      json.obj.mapValues(_.str)
    }.toSeq.map { case (key, value) => 
      val split = key.split("/").toList
      (split(8), split(9).split("\\.").last, value) 
    }.groupBy(_._1).mapValues(_.map(_._3)) //.map { case (key, value) => (key.split("/")) }
    
    val goldPred = episodes.map { case (episode, episodeNumber) =>
      val rulesForEpisode = episode.supportSentences.flatMap { x => individualRules(x).map { it => TacredPattern(it.rule, it.relation, x.sentence.getDirectionality, 1.0) } }

      val (queryTokens, querySpec) = episode.querySentence.getSentenceWithTypesAndSpecWithTypes(0)
      val ee = ExtractorEngine.inMemory(DocumentFromSentences.documentFromSentencesAndCache(Seq(queryTokens), p, cache))

      val result = TacredEvaluation.predictRelations(rulesForEpisode, ee, querySpec, episode.querySentence.getDirectionality)

      if(result.nonEmpty) {
        (episode.queryRelation, result.maxBy(_._2)._1)
      } else {
        (episode.queryRelation, "no_relation")
      }

    }

    println(microScores(goldPred, Set("no_relation")))

    // println(rules)
    // val patterns = readPatterns(PandasLikeDataset(rules.lines.sortBy(-_("spec_size").toInt)), tacredConfig)
  }

  /**
    * 
    *
    * @param supportSentence  : 
    * @param p                : 
    * @param cache            : 
    * @param scorer           : 
    * @param steps            : 
    * @param numberOfSolutions: 
    * @param fieldNames       : 
    * @return
    */
  def solveForSupportingSentence(
    supportSentence  : SupportSentence,
    p                : FastNLPProcessor,
    cache            : String,
    scorer           : Scorer,
    steps            : Option[Int],
    numberOfSolutions: Option[Int],
    fieldNames       : Set[String] = Set("word", "tag", "lemma")
  ): Seq[TacredPattern] = {
    
    val (tokens, spec) = supportSentence.sentence.getSentenceWithTypesAndSpecWithoutTypes(0)
    val doc            = DocumentFromSentences.documentFromSentencesAndCache(Seq(tokens), p, cache)

    val ft         = f"[word=${supportSentence.sentence.getFirstType.toLowerCase()}]"
    val st         = f"[word=${supportSentence.sentence.getSecondType.toLowerCase()}]"
    val firstType  = Parser.parseBasicQuery(ft)
    val secondType = Parser.parseBasicQuery(st)
    
    val searcher   = new TacredSearcher(Seq(doc), Set(spec.copy(docId = doc.id)), fieldNames, steps, None, scorer, false, firstType, secondType)

    if(numberOfSolutions.isDefined) {
      searcher.getAll(numberOfSolutions.get).map(it => TacredPattern(it.rule.pattern, supportSentence.relation, supportSentence.sentence.getDirectionality, it.nSteps))
    } else {
      searcher.findFirst().map(it => TacredPattern(it.rule.pattern, supportSentence.relation, supportSentence.sentence.getDirectionality, it.nSteps)).map { it => Seq(it) }.getOrElse(Seq.empty)
    }

  }

  def fakeSyntaxSolveForSupportingSentence(supportSentence: SupportSentence, 
    p: FastNLPProcessor, 
    cache: String, 
    scorer: Scorer, 
    steps: Option[Int],
    numberOfSolutions: Option[Int],
    fieldNames       : Set[String] = Set("word", "tag", "lemma")
  ): Seq[TacredPattern] = {

    val doc = DocumentFromSentences.documentFromSentencesAndCache(Seq(supportSentence.sentence.getSentenceWithTypes), p, cache)
    val newWords = wordsOnSyntacticPathBetween(doc, supportSentence.sentence.getFirstType, supportSentence.sentence.getSecondType)

    val ft = f"[word=${supportSentence.sentence.getFirstType.toLowerCase()}]" //.distinct.map(it => f"word=$it").mkString(start="[", sep=" | ", end = "]")
    val st = f"[word=${supportSentence.sentence.getSecondType.toLowerCase()}]" //.distinct.map(it => f"word=$it").mkString(start="[", sep=" | ", end = "]")
    val firstType  = Parser.parseBasicQuery(ft)
    val secondType = Parser.parseBasicQuery(st)

    val newDoc = DocumentFromSentences.documentFromSentencesAndCache(Seq(newWords), p, cache)

    val searcher   = new TacredSearcher(Seq(newDoc), Set(Spec(newDoc.id, 0, 1, newWords.size-1)), Set("word", "tag", "lemma"), steps, None, scorer, false, firstType, secondType)

    if(numberOfSolutions.isDefined) {
      searcher.getAll(numberOfSolutions.get).map(it => TacredPattern(it.rule.pattern, supportSentence.relation, supportSentence.sentence.getDirectionality, it.nSteps))
    } else {
      searcher.findFirst().map(it => TacredPattern(it.rule.pattern, supportSentence.relation, supportSentence.sentence.getDirectionality, it.nSteps)).map { it => Seq(it) }.getOrElse(Seq.empty)
    }
  }


  /**
    * If you wish to load what was written with this method, use loadMultiRulesPerSentenceFromFile
    * Note that we save every support sentence, even if we do not find a rule for it. In that case, the
    * sequence will be empty
    *
    * @param savepath        : where to save the result (String)
    * @param supportSentences: the sentences to attempt to generate a rule for (generate a rule for each sentence)
    * @param p               : processor object; used to create a document if one does not exist
    * @param cache           : caching processor's documents here
    * @param scorer          : the scorer to be used
    * @param fieldNames      : the field names to be used inside the searcher
    * @param numberOfSolution: how many solutions to attempt to find
    */
  def saveRulesToFile(
    savepath             : String,
    supportSentences     : Seq[SupportSentence],
    p                    : FastNLPProcessor,
    cache                : String,
    scorers              : Seq[DynamicWeightScorer],
    fieldNames           : Set[String] = Set("word", "tag", "lemma"),
    numberOfSolution     : Option[Int] = None,
    numberOfSteps        : Option[Int] = Some(1000),
    fakeSyntax           : Boolean = false,
    existingSolutionsPath: Option[String] = None, // Some("/data/nlp/corpora/few-rel/rules/train_c=1_8_512_surface_1000steps_5sols_part1_temp.pkl"),
    )                    : Unit = {
    implicit val rwPD: RW[PatternDirection] = macroRW
    implicit val rwTP: RW[TacredPattern]    = macroRW
    implicit val rwSS: RW[SupportSentence]  = macroRW
        
    val existingSolutions = if (existingSolutionsPath.isDefined && (new File(existingSolutionsPath.get)).exists()) {
      loadMultiRulesPerSentenceFromFile(existingSolutionsPath.get).toMap
    } else {
      Map.empty[SupportSentence, Seq[TacredPattern]]
    }

    val pw = new PrintWriter(new File(savepath))


    val random      = new Random(1)
    val taskSupport = new ForkJoinTaskSupport(new ForkJoinPool(scorers.size))
    val size        = (supportSentences.size / scorers.size).toInt + 1
    val ai          = new AtomicInteger()

    random.shuffle(supportSentences).zipWithIndex.sliding(size, size).toSeq.zip(scorers).par.let { it => it.tasksupport = taskSupport; it }
              .flatMap { case (ss, scorer) => ss.map { it => (it._1, it._2, scorer) } }.foreach { case (supportSentence, idx, scorer) => 
      
      val solution = if(Seq("\n", "\t", " ", "\u00a0").exists(supportSentence.sentence.tokens.contains)) { // "\n" is not correctly tokenized by huggingface tokenizers; "\u00a0" is a no-break space
        Seq.empty[TacredPattern]
      } else if (existingSolutions.contains(supportSentence)) {
          existingSolutions(supportSentence)
      } else {
        if(fakeSyntax) {
          fakeSyntaxSolveForSupportingSentence(supportSentence, p, cache, scorer, steps = numberOfSteps, numberOfSolutions = numberOfSolution, fieldNames = fieldNames)
        } else {
          solveForSupportingSentence(supportSentence, p, cache, scorer, steps = numberOfSteps, numberOfSolutions = numberOfSolution, fieldNames = fieldNames)
        }
      }

      


      val result = (supportSentence, solution)
      val string = write(result)
      println(f"${ai.getAndIncrement()}/${supportSentences.size} - ${solution.map(_.pattern)} (${scorer.apiCaller.scoreEndpoint})")
      // pw.synchronized {
      pw.println(string)
      // pw.flush()
      // }
      // pw.println(string)
      result
    }
    pw.close()
    
  }

  /**
    * Load the rules that are stored in the file, one json per line
    *
    * @param loadpath
    * @return
    */
  def loadSequenceOfRules(loadpath: String): Seq[(SupportSentence, Option[TacredPattern])] = {
    using(Source.fromFile(loadpath)) { it => 
      val data = it.getLines().map { it =>
        read[(SupportSentence, Option[TacredPattern])](it)        
      }
      data.toList.toSeq
    } 
  }

  def saveRulesToFileFromSequence(savepath: String, supportSentences: Seq[Seq[SupportSentence]], 
                                  p: FastNLPProcessor, 
                                  cache: String, 
                                  scorer: Scorer, 
                                  fieldNames: Set[String] = Set("word", "tag", "lemma")
                                  ): Unit = {
    // val writer = new PrintWriter(new File(savepath))
    val supportSentenceTorules = supportSentences.zipWithIndex.map { case (supportSentences, idx) => 
      assert(supportSentences.map(_.relation).toSet.size == 1)

      val tokensAndSpec = supportSentences.zipWithIndex.map { case (ss, sentId) => ss.sentence.getSentenceWithTypesAndSpecWithoutTypes(sentId) }
      val doc = DocumentFromSentences.documentFromSentencesAndCache(tokensAndSpec.map(_._1), p, cache)
      val spec = tokensAndSpec.map(_._2).map(_.copy(docId = doc.id)).toSet
      
      val firstTypeString  = supportSentences.map(_.sentence.getFirstType.toLowerCase()).map(it => f"word=$it").mkString(start="[", sep=" | ", end = "]")
      val secondTypeString = supportSentences.map(_.sentence.getSecondType.toLowerCase()).map(it => f"word=$it").mkString(start="[", sep=" | ", end = "]")
      val firstType  = Parser.parseBasicQuery(firstTypeString)
      val secondType = Parser.parseBasicQuery(secondTypeString)
      val searcher   = new TacredSearcher(Seq(doc), spec, fieldNames, Some(1000), None, scorer, false, firstType, secondType)
  
      val solution = searcher.findFirst()
      // writer.println(f"${supportSentence}\t$solution")
      println(f"${idx+1}/${supportSentences.size} - ${solution.map(_.rule.pattern)}")
      // solution.map { rule => (supportSentence, Pattern(rule.rule.pattern, supportSentence.relation)) }
      (supportSentences, solution.map { rule => Pattern(rule.rule.pattern, supportSentences.head.relation) })
    }.toMap
  
    // writer.close()
    
    val string = write(supportSentenceTorules.toList)
    using(new PrintWriter(new File(savepath))) { pw =>
      pw.println(string)
    }
    
  }

  def loadRulesFromFile(loadpath: String): Seq[(SupportSentence, Option[Pattern])] = {
    using(Source.fromFile(loadpath)) { it => 
      val string = it.mkString
      read[Seq[(SupportSentence, Option[Pattern])]](string)
    }
  }  
  
  def loadMultiRulesPerSentenceFromFile(loadpath: String): Seq[(SupportSentence, Seq[TacredPattern])] = {
    using(Source.fromFile(loadpath)) { it => 
      it.getLines().toList.map { it =>
        val string = it.mkString
        read[(SupportSentence, Seq[TacredPattern])](string)
      }
    }
  }

  
  /**
    * Reads the train and test episodes stored in @param filename
    *
    * @param filename
    * @return
    */
  def readTrainTestEpisodes(filename: String): Seq[MultiQueryEpisode] = {
    val json  = using(Source.fromFile(filename)) { it => ujson.read(it.mkString) }
    json.arr.let { it => 
      val itArr = it.arr

      val episodes = itArr(0).arr.zip(itArr(2).arr).map { case (trainTestDict, relations) =>

        val (trainRel, testRel) = (relations.arr(0).arr.map(_.str), relations.arr(1).arr.map(_.str)) 

        // Each internal array contains sentences corresponding to the same relation
        val train = trainTestDict("meta_train").arr.map { sentences => 
          sentences.arr.map(FewShotSentence.fromJson)
        }.zip(trainRel).flatMap { case (sentences, relation) => sentences.map(sentence => SupportSentence(relation, sentence)) }
        val test  = trainTestDict("meta_test").arr.map(FewShotSentence.fromJson)
        
        assert(test.length == testRel.length)
        MultiQueryEpisode(train, test, testRel)
        // test.zip(testRel).map { case (t, r) => Episode(train, t, r) }

      }
      
      episodes.toSeq
    }

  }

  /**
    * Read the episodes, where each episode contains exactly one query
    *
    * @param filename: String, where to find the file containing the data
    * @return a sequence of episodes
    */
  def readUnrolledTestEpisodes(filename: String): Seq[Episode] = {
    readTrainTestEpisodes(filename).flatMap(_.unroll)
  }

  def calculateConfusionMatrixForLabel(goldAndPrediction: Seq[(String, String)], label: String): ConfusionMatrix = {

    ConfusionMatrix(
      tp = goldAndPrediction.count { case (gold, predicted) => gold == predicted && gold == label},
      tn = goldAndPrediction.count { case (gold, predicted) => gold != label && predicted != label},
      fp = goldAndPrediction.count { case (gold, predicted) => gold != label && predicted == label },
      fn = goldAndPrediction.count { case (gold, predicted) => gold == label && predicted != label},
    )

  }

  /**
  * 
  *
  * @param goldAndPrediction a sequence of 2-tuple, where _1 is the gold label and _2 is the predicted label
  * @return 3-tuple, where _1 is precision, _2 is recall and _3 is f1
  */
  def microScores(goldAndPrediction: Seq[(String, String)], ignoreLabels: Set[String] = Set.empty): (Double, Double, Double) = {

    val labels = goldAndPrediction.map(_._1).distinct

    val result = labels.map(it => (it, calculateConfusionMatrixForLabel(goldAndPrediction, it)))
                     .filterNot(it => ignoreLabels.contains(it._1))
                     
    result.map(it => f"${it._1} - ${it._2} (p=${it._2.precision}; r=${it._2.recall}; f1=${it._2.f1})").foreach(println)                 
    println("-"*100)
    val no_rel = calculateConfusionMatrixForLabel(goldAndPrediction, "no_relation")
    println(f"$no_rel (p=${no_rel.precision}; r=${no_rel.recall}; f1=${no_rel.f1})")

    val tp = result.map(_._2.tp).sum.toDouble
    val fp = result.map(_._2.fp).sum.toDouble
    val fn = result.map(_._2.fn).sum.toDouble
    
    val p  = tp / (tp + fp)
    val r  = tp / (tp + fn)
    val f1 = if (p+r == 0) 0 else 2 * (p*r)/(p+r)
    (p, r, f1)
    
  }

  def macroAverageScores(goldAndPrediction: Seq[(String, String)]): (Double, Double, Double) = {
    val scores = goldAndPrediction.map(_._1).distinct.map { label =>
      val scoreForLabel = calculateConfusionMatrixForLabel(goldAndPrediction, label)
      println(f"$label - (p=${scoreForLabel.precision}), (r=${scoreForLabel.recall}), (f1=${scoreForLabel.f1})")
      (scoreForLabel.precision, scoreForLabel.recall, scoreForLabel.f1)
    }

    // scores.map { it => (it._1, goldAndPrediction.filter(_._1 == it._1).length, goldAndPrediction.filter(_._2 == it._1).length, it._2, it._3, it._4)}.foreach(println)

    (scores.map(_._1).sum / scores.length, scores.map(_._2).sum / scores.length, scores.map(_._3).sum / scores.length)
  }

  def allRules(sentence: SupportSentence, p: FastNLPProcessor, cache: String, fieldNames: Set[String] = Set("word", "lemma", "tag")): Seq[TacredPattern] = {

    val (tokens, spec) = sentence.sentence.getSentenceWithTypesAndSpecWithTypes(0)
    val doc = DocumentFromSentences.documentFromSentencesAndCache(Seq(tokens), p, cache)
    val range = spec.start until spec.end

    val vocabularyTokens = doc.sentences(0).fields
                              .collect { case TokensField(name, tokens) if (fieldNames.contains(name)) => (name, tokens) } // take only the data associated with a field name of interest
                              .flatMap { case (name, tokens) => tokens.zipWithIndex.map { case (token, idx) => (idx, name, token) } } // append the index of the token
                              .groupBy(_._1).mapValues(_.map { case (_, name, tokens) => (name, tokens) }) // switch to a map representation
                              .filterKeys(range.contains) // filter out the positions outside the highlighted part
                              .toSeq // go back to the seq representation
                              .sortBy(_._1) // put everything in order 
                              .toSeq

    generalizedCross(vocabularyTokens.map(_._2))
                                .map { rule => rule.map { case (name, value) => f"[$name=$value]" }.mkString(" ") }
                                .map { rule => TacredPattern(rule, sentence.relation, sentence.sentence.getDirectionality, 1.0) }
    

  }

  def appendReversedTacredRule(rules: Seq[TacredPattern]): Seq[TacredPattern] = {
    rules.flatMap { it =>
      val queries = Parser.parseBasicQuery(it.pattern).asInstanceOf[ConcatQuery].queries
      val pattern = (new ConcatQuery(Vector(queries.last) ++ queries.tail.init ++ Vector(queries.head))).pattern
      
      val direction = if(it.direction == SubjObjDirection) ObjSubjDirection else SubjObjDirection
      
      Seq(it, it.copy(pattern = pattern, direction = direction))
    }
  }
  

}
