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


case class FewRelSentence(
  tokens: Seq[String],
  subjStart: Int,
  subjEnd: Int,
  objStart: Int,
  objEnd: Int,
)
object FewRelSentence {
  def fromJson(jsonValue: Value): FewRelSentence = {
    FewRelSentence(
      tokens    = jsonValue("tokens").arr.map(_.str),
      subjStart = jsonValue("subj_start").num.toInt,
      subjEnd   = jsonValue("subj_end").num.toInt,
      objStart  = jsonValue("obj_start").num.toInt,
      objEnd    = jsonValue("obj_end").num.toInt,
    )

  }
}

/**
  * The idea of this app is to read the data, which is a similar format as that of the Few-Shot TACRED (without entity types, for examples, because there are not in the original data)
  * Then, transform it in the same objects as for Few-Shot TACRED (including entity types)
  * Then, save them to file
  */
object DataPreprocessingApp extends App {
  implicit val rw: RW[Episode] = macroRW
  val r       = new Random(1)
  val random      = new Random(1)
  val p = {
    initializeDyNet()
    new FastNLPProcessor
  }
  
  // println(FewShotTacredApp.loadMultiRulesPerSentenceFromFile("/data/nlp/corpora/few-rel/rules/train_c=1_8_512_fakesyntax_1000steps_5sols.pkl").toMap.head._1)
  // println(FewShotTacredApp.loadMultiRulesPerSentenceFromFile("/data/nlp/corpora/few-rel/rules/train_c=1_8_512_fakesyntax_1000steps_5sols.pkl").toMap.head._2)
  // System.exit(1)


  val cache  = "/data/nlp/corpora/fs-tacred/odinsynth_cache"
  val scorers = Seq(8002, 8003, 8004, 8005, 8006, 8007, 8008, 8009, 8010, 8011, 8012, 8013, 8014, 8015, 8016).map(it => DynamicWeightScorer(f"http://localhost:$it"))
  val taskSupport = new ForkJoinTaskSupport(new ForkJoinPool(40))

  // val taskSupport = new ForkJoinTaskSupport(new ForkJoinPool(32))
  
  def loadSequenceOfEpisodes(path: String): Seq[Episode] = {
    using(Source.fromFile(path)) { it =>
      it.getLines().toList.map { line =>
        read[Episode](line)
      }.toSeq
    }
  }


  implicit val rwPD: RW[PatternDirection] = macroRW
  implicit val rwTP: RW[TacredPattern]    = macroRW

  val episodes = loadSequenceOfEpisodes("/data/nlp/corpora/few-rel/data_processed/5_way_1_shots_10K_episodes_1q_seed_1_na_05_processed.json")
  val loadedExpanded = loadExpandedSupportSentences("/home/rvacareanu/projects/temp/odinsynth_specexpander/specexpander/data/data.jsonl").toMap
  val expanded = {
    episodes.flatMap(_.supportSentences).flatMap { it => it::loadedExpanded(it.sentence).toList.map { fss => SupportSentence(it.relation, fss) } }.filter(_.sentence.getHighlightedPart().size < 5)
  }

  FewShotTacredApp.saveRulesToFile(
    savepath         = f"/data/nlp/corpora/few-rel/rules/211230/def_expanded_lessthan5_c=1_8_512_surface_250steps_1sols.pkl",
    supportSentences = expanded, 
    p                = p,
    cache            = cache,
    scorers          = scorers,
    fieldNames       = Set("word", "tag", "lemma"),
    numberOfSolution = None,
    numberOfSteps    = Some(250),
    fakeSyntax       = false,
  )
  System.exit(1)

  println(episodes.flatMap(_.supportSentences).distinct.size)
  println(episodes.head)
  // val qwe123 = solvePair(
  //   episodes.head.querySentence,
  //   episodes.head.supportSentences(3),
  //   cache,
  //   DynamicWeightScorer(f"http://localhost:8002"),
  // )
  val rels = episodes.map { it => 
    val validRelations = it.supportSentences.map(_.relation)
    val goldRelation   = if (validRelations.contains(it.queryRelation)) {
      it.queryRelation
    } else {
      "no_relation"
    }
    (it.querySentence, goldRelation) 
  }.toMap
  val data = episodes.flatMap { e => e.supportSentences.map { ss => (e.querySentence, ss) } }.distinct//.take(10)
  val size = (data.size / scorers.size).toInt + 1
  val ai1 = new AtomicInteger()
  val pwSurface = new PrintWriter(new File("/data/nlp/corpora/few-rel/rules/211230/train_c=1_8_512_surface_1000steps_5sols.pkl"))
  r.shuffle(data).sliding(size, size).toSeq.zip(scorers).par.let { it => it.tasksupport = taskSupport; it }
  .flatMap { case (seq, scorer) =>
    seq.map { case (querySentence, ss) =>
      // if (ai1.incrementAndGet() % 100 == 0) {
        // println(ai1.get())
      // }
      val result = solvePair(
        querySentence,
        ss,
        cache,
        scorer,
        fakeSyntax=false,
        numberOfSolutions=None,
        numberOfSteps=Some(500),
        extra=f"${rels(querySentence)}"
      )
      ((querySentence, ss), result)    
    }
  }.foreach { case (key, value) =>
    val string = write((key, value))
    pwSurface.println(string)
  }
  pwSurface.close()

  val pwFs = new PrintWriter(new File("/data/nlp/corpora/few-rel/rules/211230/train_c=1_8_512_fakesyntax_1000steps_5sols.pkl"))
  r.shuffle(data).sliding(size, size).toSeq.zip(scorers).par.let { it => it.tasksupport = taskSupport; it }
  .flatMap { case (seq, scorer) =>
    seq.map { case (querySentence, ss) =>
      // if (ai1.incrementAndGet() % 100 == 0) {
        // println(ai1.get())
      // }
      val result = solvePair(
        querySentence,
        ss,
        cache,
        scorer,
        fakeSyntax=true,
        numberOfSolutions=None,
        numberOfSteps=Some(500),
        extra=f"${rels(querySentence)}",
      )
      ((querySentence, ss), result)
    }
  }.seq.foreach { case (key, value) =>
    val string = write((key, value))
    pwFs.println(string)
  }
  pwFs.close()
  
  // println(qwe123)
  System.exit(1)

  // val cache  = "/data/nlp/corpora/fs-tacred/odinsynth_cache"
  // val scorers = Seq(8002, 8003, 8004, 8005, 8006, 8007, 8008, 8009, 8010, 8011, 8012, 8013, 8014, 8015, 8016).map(it => DynamicWeightScorer(f"http://localhost:$it"))

  // val trainSentences = {
  //                     val x = loadSequenceOfEpisodes("/data/nlp/corpora/few-rel/data_processed/5_way_1_shots_10K_episodes_1q_seed_1_na_015_processed.json") ++
  //                             loadSequenceOfEpisodes("/data/nlp/corpora/few-rel/data_processed/5_way_1_shots_10K_episodes_1q_seed_1_na_05_processed.json") ++
  //                             loadSequenceOfEpisodes("/data/nlp/corpora/few-rel/data_processed/5_way_5_shots_10K_episodes_1q_seed_1_na_015_processed.json") ++
  //                             loadSequenceOfEpisodes("/data/nlp/corpora/few-rel/data_processed/5_way_5_shots_10K_episodes_1q_seed_1_na_05_processed.json")
  //                     x.flatMap(_.supportSentences).distinct
  //                   }
  // println(trainSentences.size)
  // FewShotTacredApp.saveRulesToFile(
  //   savepath         = f"/data/nlp/corpora/few-rel/rules/train_c=1_8_512_fakesyntax_1000steps_5sols.pkl",
  //   supportSentences = trainSentences, 
  //   p                = p,
  //   cache            = cache,
  //   scorers          = scorers,
  //   fieldNames       = Set("word", "tag", "lemma"),
  //   numberOfSolution = Some(5),
  //   numberOfSteps    = Some(500),
  //   fakeSyntax       = true,
  // )
  // println("#"*100)
  // FewShotTacredApp.saveRulesToFile(
  //   savepath         = f"/data/nlp/corpora/few-rel/rules/train_c=1_8_512_surface_1000steps_5sols_part1.pkl",
  //   supportSentences = trainSentences.filter(_.sentence.getHighlightedPart().size <= 6), 
  //   p                = p,
  //   cache            = cache,
  //   scorers          = scorers,
  //   fieldNames       = Set("word", "tag", "lemma"),
  //   numberOfSolution = Some(5),
  //   numberOfSteps    = Some(500),
  //   fakeSyntax       = false,
  // )
  // println("#"*100)
  // FewShotTacredApp.saveRulesToFile(
  //   savepath         = f"/data/nlp/corpora/few-rel/rules/train_c=1_8_512_surface_1000steps_5sols_part2.pkl",
  //   supportSentences = trainSentences.filterNot(_.sentence.getHighlightedPart().size <= 6), 
  //   p                = p,
  //   cache            = cache,
  //   scorers          = scorers,
  //   fieldNames       = Set("word", "tag", "lemma"),
  //   numberOfSolution = Some(5),
  //   numberOfSteps    = Some(500),
  //   fakeSyntax       = false,
  // )
  // System.exit(1)
  
  // val cache = new ConcurrentHashMap[Seq[String], Document]()
  val names = Seq(
    "/data/nlp/corpora/few-rel/data_processed/5_way_1_shots_10K_episodes_1q_seed_1_na_015_processed.json",
    "/data/nlp/corpora/few-rel/data_processed/5_way_1_shots_10K_episodes_1q_seed_1_na_05_processed.json", 
    "/data/nlp/corpora/few-rel/data_processed/5_way_5_shots_10K_episodes_1q_seed_1_na_015_processed.json",
    "/data/nlp/corpora/few-rel/data_processed/5_way_5_shots_10K_episodes_1q_seed_1_na_05_processed.json", 
  )
  Seq(
    "/data/nlp/corpora/few-rel/data/5_way_1_shots_10K_episodes_1q_seed_1_na_015.json",
    "/data/nlp/corpora/few-rel/data/5_way_1_shots_10K_episodes_1q_seed_1_na_05.json", 
    "/data/nlp/corpora/few-rel/data/5_way_5_shots_10K_episodes_1q_seed_1_na_015.json",
    "/data/nlp/corpora/few-rel/data/5_way_5_shots_10K_episodes_1q_seed_1_na_05.json", 
  ).zip(names).foreach { case (path, destination) => 
    println(f"Doing $path (cacheSize=${cache.size}). Saving it to $destination")
    val episodes = readUnrolledTestEpisodes(path, p)
    println("Everythins was read")
    using(new PrintWriter(destination)) { pw =>
      episodes.foreach { e =>
        val string = write(e)
        pw.println(string)
      }
    }
  }
  
  // Transform the FewRelSentence to a FewShotSentence
  // This involves getting the entity types between subjStart, subjEnd and objStart, objEnd
  // Because they are not provided by FewRel
  def transformToFewShotSentence(frs: FewRelSentence, p: FastNLPProcessor): FewShotSentence = {
    val (st, ot) = getEntityTypes(frs, p)
    FewShotSentence("id", "id", frs.tokens, frs.subjStart, frs.subjEnd, st, frs.objStart, frs.objEnd, ot) 
  }

  // Get the entity types (they are not gold; they are the prediction of the system)
  def getEntityTypes(frs: FewRelSentence, p: FastNLPProcessor): (String, String) = {
    val doc = DocumentFromSentences.documentFromSentencesKeepCase(Seq(frs.tokens), p)
    // if (cache.contains(frs.tokens)) {
      // cache.get(frs.tokens)
    // } else {
      // val d = DocumentFromSentences.documentFromSentencesKeepCase(Seq(frs.tokens), p)
      // cache.put(frs.tokens, d)
      // d
    // }
    val entity = doc.sentences.head.fields.collect { case TokensField("entity", tokens) => tokens }.head
    // if (entity.slice(frs.subjStart, frs.subjEnd + 1).toSet.filter(_!="O").size != 1 || entity.slice(frs.objStart, frs.objEnd + 1).toSet.filter(_!="O").size != 1) {
    //   println(frs)
    //   println(doc)
    //   println("-"*100)
    //   println(entity.slice(frs.subjStart, frs.subjEnd + 1))
    //   println(entity.slice(frs.objStart,  frs.objEnd  + 1))
    //   System.exit(1)
    // }
    val st = entity.slice(frs.subjStart, frs.subjEnd + 1).toSet.filter(_!="O").headOption.getOrElse("Entity")
    val ot = entity.slice(frs.objStart, frs.objEnd + 1).toSet.filter(_!="O").headOption.getOrElse("Entity")
    return (st, ot)

  }

  /**
  * Reads the train and test episodes stored in @param filename (FewRel format)
  *
  * @param filename
  * @return
  */
  def readTrainTestEpisodes(filename: String, p: FastNLPProcessor): Seq[MultiQueryEpisode] = {
    val json  = using(Source.fromFile(filename)) { it => ujson.read(it.mkString) }
    json.arr.let { it => 
      println(it.getClass().getName())
      val itArr = it.arr
      val ai = new AtomicInteger()
      val episodes = itArr(0).arr.zip(itArr(2).arr).map { case (trainTestDict, relations) =>
        if (ai.getAndIncrement() % 100 == 0) {
          println(ai.get())
        }

        val (trainRel, testRel) = (relations.arr(0).arr.map(_.str), relations.arr(1).arr.map(_.str)) 

        // Each internal array contains sentences corresponding to the same relation
        val train = trainTestDict("meta_train").arr.map { sentences => 
          sentences.arr.map(FewRelSentence.fromJson)
        }.zip(trainRel).flatMap { case (sentences, relation) => sentences.map(sentence => SupportSentence(relation, transformToFewShotSentence(sentence, p))) }
        val test  = trainTestDict("meta_test").arr.map(FewRelSentence.fromJson).map { it => transformToFewShotSentence(it, p) }
        
        MultiQueryEpisode(train, test, testRel)
        // test.zip(testRel).map { case (t, r) => Episode(train, t, r) }

      }
      
      episodes.toIndexedSeq.toSeq
    }
  }

  /**
    * Read the episodes, where each episode contains exactly one query
    *
    * @param filename: String, where to find the file containing the data
    * @return a sequence of episodes
    */
  def readUnrolledTestEpisodes(filename: String, p: FastNLPProcessor): Seq[Episode] = {
    readTrainTestEpisodes(filename, p).flatMap(_.unroll)
  }

  def cosineSimilarity[T](v1: Array[T], v2: Array[T])(implicit n: Numeric[T]): Double = {
    assert(v1.size == v2.size)

    var norm1 = 0.0
    var norm2 = 0.0

    var i = 0
    while (i < v1.size) {
      norm1 += n.toDouble(n.times(v1(i), v1(i)))
      norm2 += n.toDouble(n.times(v2(i), v2(i)))
      i += 1
    }

    return n.toDouble(dotProduct(v1, v2)) / (math.sqrt(norm1) * math.sqrt(norm2))
  }

  def dotProduct[T](v1: Array[T], v2: Array[T])(implicit n: Numeric[T]): T = {
    var result = n.zero

    assert(v1.size == v2.size)

    for (i <- 0 until v1.size) {
      result = n.plus(result, n.times(v1(i), v2(i)))
    }

    result
  }

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
    // println(episodes.flatMap(_.supportSentences).groupBy(_.relation).keySet.toList)
    // println(episodes.head)
    // System.exit(1)
    val (embeddings, size) = {
      ConstEmbeddingsGlove().let { cev =>
        val map = cev.w2i.keySet.toSeq.map { word => (word, cev.get(word).value.toSeq.toArray) }.toMap
        (map, map.head._2.size)
      }
    }

    def getOrEmpty(word: String): Array[Float] = if(embeddings.contains(word)) embeddings(word) else Array.fill(size)(0f)

    def cos(w1: String, w2: String): Double = {
      cosineSimilarity(getOrEmpty(w1), getOrEmpty(w2))
    }

    // val er = episodes.map { case Episode(supportSentences, querySentence, queryRelation) =>
    //   val episodeRelations = supportSentences.map(_.relation)
    //   val queryEntities    = querySentence.tokens.slice(querySentence.subjStart, querySentence.subjEnd + 1) ++ querySentence.tokens.slice(querySentence.objStart, querySentence.objEnd + 1) 
    //   val result = supportSentences.map(_.sentence.tokens).map { tokens =>
    //     val crossed = tokens.cross(querySentence.tokens)
    //     crossed.count(it => it._1 == it._2)
    //   }
    //   val goldRelation = if(episodeRelations.contains(queryRelation)) queryRelation else "no_relation"

    //   (goldRelation, supportSentences(result.zipWithIndex.maxBy(_._1)._2).relation)
    // }

    // println(er.count(it => it._1 == "no_relation"))
    // // println(microScores(er, ignoreLabels = Set("no_relation")))
    // println(er.count(it => it._1 == it._2))
    // println(er.count(it => it._1 != it._2))


    // System.exit(1)
    
    val fakeSyntaxRules = FewShotTacredApp.loadMultiRulesPerSentenceFromFile("/data/nlp/corpora/few-rel/rules/train_c=1_8_512_fakesyntax_1000steps_5sols.pkl").toMap//.mapValues(_.take(args(1).toInt))
                                          // .mapValues(_.map { it => val query = Parser.parseBasicQuery(it.pattern).asInstanceOf[ConcatQuery]; it.copy(pattern=new ConcatQuery(query.queries.tail.init).pattern) }.filterNot(_.pattern.isEmpty()) )

    val surfaceRules    = FewShotTacredApp.loadMultiRulesPerSentenceFromFile("/data/nlp/corpora/few-rel/rules/train_c=1_8_512_surface_1000steps_5sols.pkl").toMap//.withDefaultValue(Seq.empty) //.mapValues(_.take(args(1).toInt))
                                          // .mapValues(_.map { it => val query = Parser.parseBasicQuery(it.pattern).asInstanceOf[ConcatQuery]; it.copy(pattern=new ConcatQuery(query.queries.tail.init).pattern) }.filterNot(_.pattern.isEmpty()) )
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

      if (prediction == goldRelation) {
        println("-"*100)
        println(f"\t${queryRelation}; (${querySentence.subjStart}, ${querySentence.subjEnd}, ${querySentence.getSubjTokens.mkString(" ")}, ${querySentence.subjType}); (${querySentence.objStart}, ${querySentence.objEnd}, ${querySentence.getObjTokens.mkString(" ")}, ${querySentence.objType}); ${querySentence.tokens.mkString(" ")}".replace("'", "").replace("\"", ""))
        supportSentences.foreach { it => println(f"\t\t${it.relation}; (${it.sentence.subjStart}, ${it.sentence.subjEnd}, ${it.sentence.getSubjTokens.mkString(" ")}, ${it.sentence.subjType}); (${it.sentence.objStart}, ${it.sentence.objEnd}, ${it.sentence.getObjTokens.mkString(" ")}, ${it.sentence.objType}); ${it.sentence.tokens.mkString(" ")}".replace("'", "").replace("\"", "")) }
        println(f"\t${rules1.toString().replace("'", "_q1_").replace("\"", "_q2_")}")
        println(f"\t${rules2.toString().replace("'", "_q1_").replace("\"", "_q2_")}")
        println(
          f"""
          | Our prediction: ${prediction}
          | Rules for gold (${goldRelation} - ${queryRelation}):
          |     Fake syntax ${rules1.filter(_.relation == queryRelation).map(_.pattern)}
          |     Surface     ${rules2.filter(_.relation == queryRelation).map(_.pattern)}
          | We have the following relation-rules: 
          |     Fake syntax ${rules1.groupBy(_.relation).map(it => (it._1, it._2.size))}
          |     Surface     ${rules2.groupBy(_.relation).map(it => (it._1, it._2.size))}
          """.stripMargin.replace("'", "_q1_").replace("\"", "_q2_")
          )
        println("-"*100)
        println(" ")
      }
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

  def generateRules(): Unit = {

    val cache  = "/data/nlp/corpora/fs-tacred/odinsynth_cache"
    val scorers = Seq(8002, 8003, 8004, 8005, 8006, 8007, 8008, 8009, 8010, 8011, 8012, 8013, 8014, 8015, 8016).map(it => DynamicWeightScorer(f"http://localhost:$it"))

    val trainSentences = {
                        val x = loadSequenceOfEpisodes("/data/nlp/corpora/few-rel/data_processed/5_way_1_shots_10K_episodes_1q_seed_1_na_015_processed.json") ++
                                loadSequenceOfEpisodes("/data/nlp/corpora/few-rel/data_processed/5_way_1_shots_10K_episodes_1q_seed_1_na_05_processed.json") ++
                                loadSequenceOfEpisodes("/data/nlp/corpora/few-rel/data_processed/5_way_5_shots_10K_episodes_1q_seed_1_na_015_processed.json") ++
                                loadSequenceOfEpisodes("/data/nlp/corpora/few-rel/data_processed/5_way_5_shots_10K_episodes_1q_seed_1_na_05_processed.json")
                        x.flatMap(_.supportSentences).distinct
                      }
    println(trainSentences.size)
    FewShotTacredApp.saveRulesToFile(
      savepath         = f"/data/nlp/corpora/few-rel/rules/train_c=1_8_512_fakesyntax_1000steps_5sols.pkl",
      supportSentences = trainSentences, 
      p                = p,
      cache            = cache,
      scorers          = scorers,
      fieldNames       = Set("word", "tag", "lemma"),
      numberOfSolution = Some(5),
      numberOfSteps    = Some(1000),
      fakeSyntax       = true,
    )
    println("#"*100)
    FewShotTacredApp.saveRulesToFile(
      savepath         = f"/data/nlp/corpora/few-rel/rules/train_c=1_8_512_surface_1000steps_5sols_part1.pkl",
      supportSentences = trainSentences.filter(_.sentence.getHighlightedPart().size <= 6), 
      p                = p,
      cache            = cache,
      scorers          = scorers,
      fieldNames       = Set("word", "tag", "lemma"),
      numberOfSolution = Some(5),
      numberOfSteps    = Some(1000),
      fakeSyntax       = false,
    )
    println("#"*100)
    FewShotTacredApp.saveRulesToFile(
      savepath         = f"/data/nlp/corpora/few-rel/rules/train_c=1_8_512_surface_1000steps_5sols_part2.pkl",
      supportSentences = trainSentences.filterNot(_.sentence.getHighlightedPart().size <= 6), 
      p                = p,
      cache            = cache,
      scorers          = scorers,
      fieldNames       = Set("word", "tag", "lemma"),
      numberOfSolution = Some(5),
      numberOfSteps    = Some(1000),
      fakeSyntax       = false,
    )
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
    supportSentences : Seq[SupportSentence],
    p                : FastNLPProcessor,
    cache            : String,
    scorer           : Scorer,
    steps            : Option[Int],
    numberOfSolutions: Option[Int],
    fieldNames       : Set[String] = Set("word", "tag", "lemma"),
    relation         : String,
  ): Seq[TacredPattern] = {
    
    val (tokens, specs) = supportSentences.zipWithIndex.map { case (ss, idx) => ss.sentence.getSentenceWithTypesAndSpecWithoutTypes(idx) }.unzip
    val doc            = DocumentFromSentences.documentFromSentencesAndCache(tokens, p, cache)
    

    val ft         = supportSentences.map(_.sentence.getFirstType.toLowerCase()).mkString("[word=", " | ", "]")
    val st         = supportSentences.map(_.sentence.getSecondType.toLowerCase()).mkString("[word=", " | ", "]")
    val firstType  = Parser.parseBasicQuery(ft)
    val secondType = Parser.parseBasicQuery(st)
    
    val searcher   = new TacredSearcher(Seq(doc), specs.map(_.copy(docId = doc.id)).toSet, fieldNames, steps, None, scorer, false, firstType, secondType)

    if(numberOfSolutions.isDefined) {
      searcher.getAll(numberOfSolutions.get).map(it => TacredPattern(it.rule.pattern, relation, supportSentences.head.sentence.getDirectionality, it.currentSteps.toDouble))
    } else {
      searcher.findFirst().map(it => TacredPattern(it.rule.pattern, relation, supportSentences.head.sentence.getDirectionality, it.currentSteps.toDouble)).map { it => Seq(it) }.getOrElse(Seq.empty)
    }

  }

  def fakeSyntaxSolveForSupportingSentence(
    supportSentences : Seq[SupportSentence], 
    p                : FastNLPProcessor, 
    cache            : String, 
    scorer           : Scorer, 
    steps            : Option[Int],
    numberOfSolutions: Option[Int],
    fieldNames       : Set[String] = Set("word", "tag", "lemma"),
    relation         : String,
  ): Seq[TacredPattern] = {

    val doc = DocumentFromSentences.documentFromSentencesAndCache(supportSentences.map(_.sentence.getSentenceWithTypes), p, cache)
    val newWords = supportSentences.map { supportSentence =>
       wordsOnSyntacticPathBetween(doc, supportSentence.sentence.getFirstType, supportSentence.sentence.getSecondType)
    }

    val ft         = supportSentences.map(_.sentence.getFirstType.toLowerCase()).mkString("[word=", " | ", "]")
    val st         = supportSentences.map(_.sentence.getSecondType.toLowerCase()).mkString("[word=", " | ", "]")
    val firstType  = Parser.parseBasicQuery(ft)
    val secondType = Parser.parseBasicQuery(st)

    val newDoc = DocumentFromSentences.documentFromSentencesAndCache(newWords, p, cache)

    val searcher   = new TacredSearcher(Seq(newDoc), Set(Spec(newDoc.id, 0, 1, newWords.size-1)), Set("word", "tag", "lemma"), steps, None, scorer, false, firstType, secondType)

    if(numberOfSolutions.isDefined) {
      searcher.getAll(numberOfSolutions.get).map(it => TacredPattern(it.rule.pattern, relation, supportSentences.last.sentence.getDirectionality, it.currentSteps.toDouble))
    } else {
      searcher.findFirst().map(it => TacredPattern(it.rule.pattern, relation, supportSentences.last.sentence.getDirectionality, it.currentSteps.toDouble)).map { it => Seq(it) }.getOrElse(Seq.empty)
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
    supportSentences     : Seq[Seq[SupportSentence]],
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
      Map.empty[Seq[SupportSentence], Seq[TacredPattern]]
    }

    val pw = new PrintWriter(new File(savepath))


    val random      = new Random(1)
    val taskSupport = new ForkJoinTaskSupport(new ForkJoinPool(scorers.size))
    val size        = (supportSentences.size / scorers.size).toInt + 1
    val ai          = new AtomicInteger()

    random.shuffle(supportSentences).zipWithIndex.sliding(size, size).toSeq.zip(scorers).par.let { it => it.tasksupport = taskSupport; it }
              .flatMap { case (ss, scorer) => ss.map { it => (it._1, it._2, scorer) } }.foreach { case (supportSentence, idx, scorer) => 
      
      val solution = if(supportSentence.exists { supportSentence => Seq("\n", "\t", " ", "\u00a0").exists(supportSentence.sentence.tokens.contains) }) { // "\n" is not correctly tokenized by huggingface tokenizers; "\u00a0" is a no-break space
        Seq.empty[TacredPattern]
      } else if (existingSolutions.contains(supportSentence)) {
          existingSolutions(supportSentence)
      } else {
        if(fakeSyntax) {
          fakeSyntaxSolveForSupportingSentence(supportSentence, p, cache, scorer, steps = numberOfSteps, numberOfSolutions = numberOfSolution, fieldNames = fieldNames, supportSentence.last.relation)
        } else {
          solveForSupportingSentence(supportSentence, p, cache, scorer, steps = numberOfSteps, numberOfSolutions = numberOfSolution, fieldNames = fieldNames, supportSentence.last.relation)
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

  def solvePair(
    querySentence    : FewShotSentence,
    supportSentence  : SupportSentence,
    cache            : String,
    scorer           : DynamicWeightScorer,
    fieldNames       : Set[String] = Set("word", "tag", "lemma"),
    numberOfSolutions: Option[Int] = None,
    numberOfSteps    : Option[Int] = Some(1000),
    fakeSyntax       : Boolean     = false,
    extra            : String      = ""
  ): Seq[TacredPattern] = {
    val tokens = Seq(querySentence.getSentenceWithTypes, supportSentence.sentence.getSentenceWithTypes)

    val sentences = Seq(querySentence, supportSentence.sentence)

    val ft         = sentences.map(it => f"word=${it.getFirstType.toLowerCase()}").mkString("[", " | ", "]")
    val st         = sentences.map(it => f"word=${it.getSecondType.toLowerCase()}").mkString("[", " | ", "]")

    val firstType  = Parser.parseBasicQuery(ft)
    val secondType = Parser.parseBasicQuery(st)

    if(fakeSyntax) {
      
      val newWords = sentences.map { s =>
        val doc = DocumentFromSentences.documentFromSentencesAndCache(Seq(s.getSentenceWithTypes), p, cache)
        wordsOnSyntacticPathBetween(doc, s.getFirstType, s.getSecondType)
      }
              
      val newDoc = DocumentFromSentences.documentFromSentencesAndCache(newWords, p, cache)

      val newSpecs = newWords.zipWithIndex.map { case (newWords, idx) => Spec(newDoc.id, idx, 1, newWords.size-1) }.toSet
      if (newSpecs.size == 1) {
        println(newWords)
        println(querySentence)
        println(supportSentence.sentence)
        System.exit(1)
      }

      if (newSpecs.exists(s => s.start == s.end)) {
        Seq.empty
      } else {
        val searcher   = new TacredSearcher(Seq(newDoc), newSpecs, Set("word", "tag", "lemma"), numberOfSteps, None, scorer, false, firstType, secondType)

        val solution = if(Seq("\n", "\t", " ", "\u00a0").exists(supportSentence.sentence.tokens.contains) || Seq("\n", "\t", " ", "\u00a0").exists(querySentence.tokens.contains)) { // "\n" is not correctly tokenized by huggingface tokenizers; "\u00a0" is a no-break space
          Seq.empty[TacredPattern]
        } else {
          if(numberOfSolutions.isDefined) {
            searcher.getAll(numberOfSolutions.get).map(it => TacredPattern(it.rule.pattern, supportSentence.relation, supportSentence.sentence.getDirectionality, it.currentSteps.toDouble))
          } else {
            searcher.findFirst().map(it => TacredPattern(it.rule.pattern, supportSentence.relation, supportSentence.sentence.getDirectionality, it.currentSteps.toDouble)).map { it => Seq(it) }.getOrElse(Seq.empty)
          }
        }
        println(f"${querySentence.hashCode};${supportSentence.sentence.hashCode()};${solution.size};$extra;${supportSentence.relation};${newSpecs.toSeq.map { it => f"[${it.start},${it.end}]" }.mkString("[", ",", "]")};${solution.map(it => f""""${it.pattern.replace("'", "_q1_").replace("\"", "_q2_")}"""").mkString("[", ",", "]")};${solution.map(_.weight).mkString("[", ",", "]")}")
        solution
      }


    } else {
      val (tokens, specs) = sentences.zipWithIndex.map { case (ss, idx) => ss.getSentenceWithTypesAndSpecWithoutTypes(idx) }.unzip
      val doc             = DocumentFromSentences.documentFromSentencesAndCache(tokens, p, cache)

      if(specs.exists(s => s.start == s.end)) { // A spec with nothing highlighted causes us problems; But in such cases it is very unlikely to obtain a rule that would match both (when appending entities)
        Seq.empty
      } else {

        val searcher   = new TacredSearcher(Seq(doc), specs.map(_.copy(docId = doc.id)).toSet, fieldNames, numberOfSteps, None, scorer, false, firstType, secondType)

        val solution = if(Seq("\n", "\t", " ", "\u00a0").exists(supportSentence.sentence.tokens.contains) || Seq("\n", "\t", " ", "\u00a0").exists(querySentence.tokens.contains)) { // "\n" is not correctly tokenized by huggingface tokenizers; "\u00a0" is a no-break space
          Seq.empty[TacredPattern]
        } else {
          if(numberOfSolutions.isDefined) {
            searcher.getAll(numberOfSolutions.get).map(it => TacredPattern(it.rule.pattern, supportSentence.relation, supportSentence.sentence.getDirectionality, it.currentSteps.toDouble))
          } else {
            searcher.findFirst().map(it => TacredPattern(it.rule.pattern, supportSentence.relation, supportSentence.sentence.getDirectionality, it.currentSteps.toDouble)).map { it => Seq(it) }.getOrElse(Seq.empty)
          }
        }
        println(f"${querySentence.hashCode};${solution.size};$extra;${supportSentence.relation};${specs.toSeq.map { it => f"[${it.start},${it.end}]" }.mkString("[", ",", "]")};${solution.map(it => f""""${it.pattern.replace("'", "_q1_").replace("\"", "_q2_")}"""").mkString("[", ",", "]")};${solution.map(_.weight).mkString("[", ",", "]")}")
        solution

      }

    }
  
  }

  def loadMultiRulesPerSentenceFromFile(loadpath: String): Seq[(Seq[SupportSentence], Seq[TacredPattern])] = {
    implicit val rwPD: RW[PatternDirection] = macroRW
    implicit val rwTP: RW[TacredPattern]    = macroRW
    implicit val rwSS: RW[SupportSentence]  = macroRW

    using(Source.fromFile(loadpath)) { it => 
      it.getLines().toList.map { it =>
        val string = it.mkString
        read[(Seq[SupportSentence], Seq[TacredPattern])](string)
      }
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
