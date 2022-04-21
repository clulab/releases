package org.clulab.odinsynth.evaluation.tacred

import scala.io.Source
import org.clulab.odinsynth.EnhancedType
import org.clulab.odinsynth.using
import org.clulab.odinsynth.Spec
import scala.collection.mutable
import ai.lum.odinson.ExtractorEngine
import org.clulab.dynet.Utils.initializeDyNet
import org.clulab.processors.fastnlp.FastNLPProcessor
import ai.lum.odinson.extra.ProcessorsUtils
import java.io.PrintWriter
import java.io.File
import org.clulab.odinsynth.EnhancedColl
import org.clulab.odinsynth.Parser
import org.clulab.odinsynth.ConcatQuery
import org.clulab.odinsynth.evaluation.PandasLikeDataset
import org.clulab.odinsynth.evaluation.DocumentFromSentences
import com.typesafe.scalalogging.LazyLogging
import ai.lum.odinson.Document
import scala.util.Try
import org.clulab.odinsynth.Query
import org.clulab.odinsynth.RepeatQuery
import org.clulab.odinsynth.TokenQuery
import org.clulab.odinsynth.MatchAllQuery
import org.clulab.odinsynth.HoleQuery
import org.clulab.odinsynth.OrQuery
import org.clulab.odinsynth.TokenConstraint
import org.clulab.odinsynth.MatchAllConstraint
import org.clulab.odinsynth.HoleConstraint
import org.clulab.odinsynth.FieldConstraint
import org.clulab.odinsynth.NotConstraint
import org.clulab.odinsynth.AndConstraint
import org.clulab.odinsynth.OrConstraint
import org.clulab.odinsynth.Matcher
import org.clulab.odinsynth.MatchAllMatcher
import org.clulab.odinsynth.HoleMatcher
import org.clulab.odinsynth.StringMatcher
import com.typesafe.config.Config
import com.typesafe.config.ConfigFactory
import com.typesafe.config.ConfigValue
import com.typesafe.config.ConfigValueFactory

/**
 * Using the rules generated using TacredRuleGeneration, evaluate their performance on train/dev/test splits of TACRED dataset
 * The splits used here are preprocessed. Their format is (\t separated):
 * subj_start subj_end subj_type obj_start obj_end obj_type highlighted tokens reversed relation
 * 
 * It's similar to pandas. One way to read it is to use PandasLikeDataset
 * 
 * Running command
 *        $ sbt -J-Xmx32g 'runMain org.clulab.odinsynth.evaluation.tacred.TacredEvaluation static_rule_evaluation.conf'
 *        $ sbt -J-Xmx32g 'runMain org.clulab.odinsynth.evaluation.tacred.TacredEvaluation dynamic_rule_evaluation.conf' -Dodinsynth.evaluation.tacred.relationsPath=path_to_all_solutions.tsv -Dodinsynth.evaluation.tacred.weightingScheme="logspecsize" -Dodinsynth.evaluation.tacred.distinctRules=true
 */
object TacredEvaluation extends App with LazyLogging {

  // run(args)

  override def main(args: Array[String]): Unit = {
    val config = TacredConfig.from(args.head)
    run(config)
  }

  def handleQuotesInPatterns(p: String): String = {
    // Replace ""<things>"" with "<things>"
    val output = p.replace("\"\"", "\"")
    if (output.length > 2 && output.head == '"' && output.last == '"') {
      f"""${output.tail.init}"""
    } else {
      output
    }
  }

  /**
    * If there is [x=y]* or [x=y]+ we cannot remove it
    *
    * @param pattern
    * @param init
    * @param end
    * @return
    */
  def transformPattern(pattern: String, firstObject: String, lastObject: String): Query = {
    var patternParsed = Parser.parseBasicQuery(handleQuotesInPatterns(pattern)).asInstanceOf[ConcatQuery].queries
    val head = patternParsed.head
    val last = patternParsed.last
    // if (head.isInstanceOf[RepeatQuery] || last.isInstanceOf[RepeatQuery]) {
    //   return Parser.parseBasicQuery(handleQuotesInPatterns(pattern))
    // } else if (head.isInstanceOf[RepeatQuery]) {
    //   return Parser.parseBasicQuery(f"${handleQuotesInPatterns(ConcatQuery(patternParsed.init).pattern)} $lastObject")
    // } else if (last.isInstanceOf[RepeatQuery]) {
    //   return Parser.parseBasicQuery(f"$firstObject ${handleQuotesInPatterns(ConcatQuery(patternParsed.tail).pattern)}")
    // } else {
    //   return Parser.parseBasicQuery(f"$firstObject ${handleQuotesInPatterns(ConcatQuery(patternParsed.tail.init).pattern)} $lastObject")
    // }


    val f = if (last.isInstanceOf[RepeatQuery]) {
      val firstRQ = last.asInstanceOf[RepeatQuery]
      if (firstRQ.max != None) {
        RepeatQuery(Parser.parseBasicQuery(firstObject), 0, 1)
      } else {
        ConcatQuery(Vector(Parser.parseBasicQuery(firstObject), firstRQ))
      }
    } else {
      Parser.parseBasicQuery(firstObject)
    }

    val l = if (last.isInstanceOf[RepeatQuery]) {
      val lastRQ = last.asInstanceOf[RepeatQuery]
      if (lastRQ.max != None) {
        RepeatQuery(Parser.parseBasicQuery(firstObject), 0, 1)
      } else {
        ConcatQuery(Vector(lastRQ, Parser.parseBasicQuery(lastObject)))
      }
    } else {
      Parser.parseBasicQuery(firstObject)
    }

    val patternWithoutEntities = ConcatQuery(Parser.parseBasicQuery(handleQuotesInPatterns(pattern)).asInstanceOf[ConcatQuery].queries.tail.init)
    return ConcatQuery((f +: patternWithoutEntities.queries.toList :+ l).toVector)
    // return ConcatQuery(Parser.parseBasicQuery +: patternWithoutEntities.queries.toList)
    
    // Parser.parseBasicQuery(f"${handleQuotesInPatterns(f)} ${handleQuotesInPatterns(patternWithoutEntities.pattern)} ${handleQuotesInPatterns(l)}")

    // patternParsed = if (last.isInstanceOf[RepeatQuery] && last.asInstanceOf[RepeatQuery].max == None) {
    //   patternParsed
    // } else {
    //   patternParsed.init
    // }
    // val query = new ConcatQuery(patternParsed)
    

    Parser.parseBasicQuery(f"$firstObject $lastObject")
  }

  def readPatterns(rules: PandasLikeDataset, config: TacredConfig): Seq[TacredPattern] = {
    // Read the patterns. Optionally, augment them such that the first constraint is subj/obj 
    // and the second constraint is obj/subj (based on directionality)
    // Seq[(String, String, PatternDirection)] ==> corresponding to (pattern, relation, direction)
    // println(rules      .filter { it => Try(Parser.parseBasicQuery(it("pattern"))).isSuccess }.length())
    // println(rules      .filter { it => Try(Parser.parseBasicQuery(it("pattern"))).isFailure }.length())
    // System.exit(1)
    val weightingScheme = config.weightingScheme
    val patterns = rules      .filter(_("pattern").nonEmpty)
                              // .filter(_("num_steps").toInt > 0)
                              .filter { it => Try(Parser.parseBasicQuery(handleQuotesInPatterns(it("pattern")))).isSuccess }//.let { it => println(it.length()); it.map(_("pattern")).take(20).foreach(println); System.exit(1); it }
                              .flatMap { it => 
                                val cluster = PandasLikeDataset(it("cluster_path"))
                                val subjType = cluster.lines.map(it => it("subj_type").toLowerCase()).distinct.map(it => f"word=$it").mkString(start="[", sep=" | ", end = "]")
                                val objType = cluster.lines.map(it => it("obj_type").toLowerCase()).distinct.map(it => f"word=$it").mkString(start="[", sep=" | ", end = "]")                              
                                val direction = PatternDirection.fromIntValue(it("direction").toInt)
                                
                                val patternWithoutEntities = ConcatQuery(Parser.parseBasicQuery(handleQuotesInPatterns(it("pattern"))).asInstanceOf[ConcatQuery].queries.tail.init)
                                val adjustedPattern = direction match {
                                  case SubjObjDirection => Parser.parseBasicQuery(f"$subjType ${handleQuotesInPatterns(patternWithoutEntities.pattern)} $objType")
                                  case ObjSubjDirection => Parser.parseBasicQuery(f"$objType ${handleQuotesInPatterns(patternWithoutEntities.pattern)} $subjType")
                                }
                                val processedPattern = direction match {
                                  case SubjObjDirection => transformPattern(it("pattern"), subjType, objType).pattern
                                  case ObjSubjDirection => transformPattern(it("pattern"), objType, subjType).pattern
                                }

                                val pattern = it("pattern")
                                // val pattern = TestTestTest.correctSeq2SeqTree(Parser.parseBasicQuery(it("pattern").stripPrefix("\"").stripSuffix("\""))).pattern
                                // val pattern = TestTestTest.correctSeq2SeqTree(Parser.parseBasicQuery(it("pattern"))).pattern
                                // if(subjType == objType) {
                                  // Seq(
                                    // TacredPattern(pattern, it("relation"), PatternDirection.fromIntValue(it("direction").toInt), Math.log(it("spec_size").toInt + 1)/Math.log(2.0)), 
                                    // TacredPattern(Parser.parseBasicQuery(pattern).asInstanceOf[ConcatQuery].let { it => it.copy(it.queries.reverse).pattern }, it("relation"), PatternDirection.fromIntValue(it("direction").toInt), Math.log(it("spec_size").toInt + 1)/Math.log(2.0)), 
                                    // 
                                  // )
                                // } else {
                                  // Seq(TacredPattern(pattern, it("relation"), PatternDirection.fromIntValue(it("direction").toInt), Math.log(it("spec_size").toInt + 1)/Math.log(2.0)))
                                // }
                                // """[word=person] [word=","] [word=and] [word=person]"""
                                // val pattern = adjustedPattern.pattern
                                // val pattern = processedPattern 
                                // if (it("wrong_matches").toInt == 0 || it("relation") == "no_relation") {
                                // if (it("trials").toInt != 0) {
                                  // Some(TacredPattern(pattern, it("relation"), PatternDirection.fromIntValue(it("direction").toInt), 1.0))
                                  Some(TacredPattern(pattern, it("relation"), PatternDirection.fromIntValue(it("direction").toInt), weightingScheme.weight(it)))
                                  // Some(TacredPattern(pattern, it("relation"), PatternDirection.fromIntValue(it("direction").toInt), it("spec_size").toInt))
                                // } else {
                                  // None
                                // }
                              }
                              // .sortBy(_._4)
                              // .let { it =>
                                // val perfectPatterns = it.count(_._4 == 0)
                                // val imperfectSize = it.size - perfectPatterns
                                // it.take(perfectPatterns) //++ it.drop(perfectPatterns).take(imperfectSize/10)
                              // }
                              // .map(it => (it._1, it._2, it._3))
                              .sortBy(-_.weight)
                              // .distinctBy(it => (it.pattern, it.direction))
                              // .map { it => it.copy(weight=1) }
                              // .groupBy(it => (it.pattern, it.relation, it.direction)).map { case ((pattern, relation, direction), group) => TacredPattern(pattern, relation, direction, group.map(_.weight).sum) }.toSeq
                              // .distinctBy(it => (it.pattern, it.relation))
                            
    // println(patterns.groupBy(_.pattern).map(_._2.map(_.relation).distinct.size).max)
    // println(patterns.groupBy(_.pattern).map(_._2.map(_.direction).distinct.size).max)
    // System.exit(1)
    if(config.distinctRules) {
      patterns.distinctBy(it => (it.pattern, it.direction))
    } else {
      patterns
    }

  }

  def predict(data: Seq[(Seq[String], Spec, Document, String, PatternDirection)], patterns: Seq[TacredPattern]): Seq[(String, String)] = {
    data.map { case (tokens, spec, doc, goldRelation, direction) =>
      val ee = ExtractorEngine.inMemory(doc)
      val predictions = predictRelations(patterns, ee, spec, direction)
      if (predictions.isEmpty) {
        (goldRelation, "no_relation")
      } else {
        (goldRelation, predictions.maxBy(_._2)._1)
      }
    }.toIndexedSeq
  }

  def runPatternByPattern(tacredConfig: TacredConfig): Seq[(Double, Double, Double)] = {
    // These are the relations that we will use currently
    
    val relationsIndexToKeep = tacredConfig.relationsToUse 
    val relationsToUse       = tacredConfig.allRelations.filter(it => relationsIndexToKeep.contains(it._2)).map(_._1)

    // Read the data as a PandasLikeDataset
    logger.info(f"Read the data to evaluate on from ${tacredConfig.devProcessed}")
    val pld = PandasLikeDataset(tacredConfig.devProcessed, index=false)
    
    val pth = tacredConfig.patternsPath
    logger.info(f"Evaluate the patterns from $pth")

    logger.info("Read the rules")
    val rules = PandasLikeDataset(pth, index=false)

    logger.info("Rules were read")

    // Read the patterns. Optionally, augment them such that the first constraint is subj/obj 
    // and the second constraint is obj/subj (based on directionality)
    val patterns = readPatterns(PandasLikeDataset(rules.lines.sortBy(-_("spec_size").toInt)), tacredConfig)
    // val patterns = readPatterns(PandasLikeDataset(rules.lines.sortBy(it => -it("spec_size").toInt)), tacredConfig)

    logger.info("Patterns were read, in the order of their cluster size, from big clusters to small clusters")

    logger.info("Initialize DyNet")
                              
    val p = {
      initializeDyNet()
      new FastNLPProcessor
    }

    logger.info("DyNet was initialized")

    logger.info(f"Process the data (${tacredConfig.devProcessed})")
    // Seq[(Seq[String], Spec, Document, String, PatternDirection)] (tokens, spec, document, relation, direction)
    val data = (0 until pld.lines.size)
                      .map { idx => 
                        val (tokens, spec) = pld.getSentenceWithTypesAndSpec(idx)
                        val relation = pld.get(idx, "relation")
                        val direction = PatternDirection.fromIntValue(pld.get(idx, "reversed").toInt)
                        val doc = DocumentFromSentences.documentFromSentencesAndCache(Seq(tokens), p, tacredConfig.odinsonDocumentCache) //ProcessorsUtils.convertDocument(p.annotateFromTokens(Seq(tokens.map(_.toLowerCase()))))
                        (tokens, spec, doc, relation, direction)
                      }
                      // .toList
    logger.info(f"The data was processed (${tacredConfig.devProcessed})")

    println("Generating the predictions of each pattern")
    val perClusterPredictions = data.par.map { case(tokens, spec, doc, goldRelation, dataDirection) =>
      val ee = ExtractorEngine.inMemory(doc)
      val results = patterns.par.map { p =>
        val predictions = predictRelations(Seq(p), ee, spec, dataDirection)
        if(predictions.isEmpty) {
          ("no_relation", 0.0)
        } else {
          val prediction = predictions.head
          assert(predictions.size == 1)
          (prediction._1, prediction._2)
        }

      }.toIndexedSeq.toSeq
      (goldRelation, results)
    }.toIndexedSeq.toSeq
    // using(new PrintWriter(new File("/home/rvacareanu/projects/results/odinsynth_tacred_generated_rules/learning_rates_dev_predictions_8_512_all"))) { it =>
    //   it.println(f"${tacredConfig.distinctRules}, ${tacredConfig.weightingScheme.getClass().getName()}, ${tacredConfig.patternsPath}")
    //   perClusterPredictions.foreach { case (goldRelation, results) =>
    //     it.println(f"$goldRelation\t${results.map { case (relation, weight) => f"$relation $weight" }.mkString("\t")}")
    //   }
    // }
    println("The predictions of each pattern were finished")

    
    val f1ScorePatternByPattern = (1 until patterns.size+1).par.map { idx =>
      // We store the perClusterPredictions as a list of pairs: (gold, <the prediction of each pattern on this cluster>)
      // We take the corresponding number of patterns, then we group by the prediction and take the max by score
      val currentPredictions = perClusterPredictions.map { case (gold, eachPatternPrediction) => 
        val predictions = eachPatternPrediction.take(idx).groupBy(_._1).mapValues(_.map(_._2).sum).toList
        
        (gold, predictions.maxBy(_._2)._1, predictions.maxBy(_._2)._2)         
      }
      val (gold, predicted, _) = currentPredictions.unzip3
      val (p, r, f1) = score(gold, predicted)

      (p, r, f1)
    }.toIndexedSeq
    
    // println(f1ScorePatternByPattern)
    // using(new PrintWriter("/home/rvacareanu/projects/results/odinsynth_tacred_generated_rules/bert_models/normal/8_512/pattern_by_pattern")) { it =>
      // f1ScorePatternByPattern.foreach(it.println)
    // }
    return f1ScorePatternByPattern.toList
    
  }

  def run(tacredConfig: TacredConfig): (Double, Double, Double, Int) = {
    // These are the relations that we will use currently
    // val tacredConfig = TacredConfig.from(args.head)
    
    val relationsIndexToKeep = tacredConfig.relationsToUse 
    val relationsToUse       = tacredConfig.allRelations.filter(it => relationsIndexToKeep.contains(it._2)).map(_._1)

    // Read the data as a PandasLikeDataset
    logger.info(f"Read the data to evaluate on from ${tacredConfig.devProcessed}")
    val pld = PandasLikeDataset(tacredConfig.devProcessed, index=false)
    
    val pth = tacredConfig.patternsPath
    logger.info(f"Evaluate the patterns from $pth")

    logger.info("Read the rules")
    val rules = PandasLikeDataset(pth, index=false)

    logger.info("Rules were read")

    // Read the patterns. Optionally, augment them such that the first constraint is subj/obj 
    // and the second constraint is obj/subj (based on directionality)
    // Seq[(String, String, PatternDirection)]
    val patterns = readPatterns(rules, tacredConfig)

    println(patterns.length)
    logger.info("Patterns were read")

    logger.info("Initialize DyNet")
                              
    val p = {
      initializeDyNet()
      new FastNLPProcessor
    }

    logger.info("DyNet was initialized")

    logger.info(f"Process the data (${tacredConfig.devProcessed})")
    // Seq[(Seq[String], Spec, Document, String, PatternDirection)] (tokens, spec, document, relation, direction)
    val data = (0 until pld.lines.size)
                      .map { idx => 
                        val (tokens, spec) = pld.getSentenceWithTypesAndSpec(idx)
                        val relation = pld.get(idx, "relation")
                        val direction = PatternDirection.fromIntValue(pld.get(idx, "reversed").toInt)
                        val doc = DocumentFromSentences.documentFromSentencesAndCache(Seq(tokens), p, tacredConfig.odinsonDocumentCache) //ProcessorsUtils.convertDocument(p.annotateFromTokens(Seq(tokens.map(_.toLowerCase()))))
                        (tokens, spec, doc, relation, direction)
                      }//.filter(_._4!="no_relation")//.take(15)
                      // .toList
    logger.info(f"The data was processed (${tacredConfig.devProcessed})")

    
    val resultsWithNoRel = predict(data, patterns)

    val resultsWithoutNoRel = predict(data.filter { it => it._4 != "no_relation" }, patterns)
    
    val isResultIn = data.par.filter { it => it._4 != "no_relation" }.map { case (tokens, spec, doc, goldRelation, direction) =>
      val ee = ExtractorEngine.inMemory(doc)

      val predictions = predictRelations(patterns, ee, spec, direction)
      
      if(predictions.map(_._1).contains(goldRelation)) {
        1
      } else {
        0
      }
    }
    
    println(pth)
    println("#"*100)
    println("Relation vs no relation (binary)")
    println("#"*100)
    val (binaryP, binaryR, binaryF1) = {
      val tp = resultsWithNoRel.count { case (gold, predicted) => gold == predicted && gold != "no_relation" }.toDouble
      val tn = resultsWithNoRel.count { case (gold, predicted) => gold == predicted && gold == "no_relation" }.toDouble
      val fp = resultsWithNoRel.count { case (gold, predicted) => gold != predicted && predicted != "no_relation" }.toDouble
      val fn = resultsWithNoRel.count { case (gold, predicted) => gold != predicted && predicted == "no_relation" }.toDouble
      
      val p  = tp / (tp + fp)
      val r  = tp / (tp + fn)
      val f1 = 2 * (p*r)/(p+r)
      (p, r, f1)
    }
    println(f"P:  ${binaryP}%2.5f")
    println(f"R:  ${binaryR}%2.5f")
    println(f"F1: ${binaryF1}%2.5f")

    println("\n\n")

    println("#"*100)
    println("Micro average (with no_relation)")
    println("#"*100)
    val microWithNoRelation = microScores(resultsWithNoRel.toList)
    println(f"P:  ${microWithNoRelation._1}%2.5f")
    println(f"R:  ${microWithNoRelation._2}%2.5f")
    println(f"F1: ${microWithNoRelation._3}%2.5f")

    println("\n\n")

    println("#"*100)
    println("Micro average (without no_relation)")
    println("#"*100)
    val microWithoutNoRelation = microScores(resultsWithoutNoRel.toList)

    println(f"P:  ${microWithoutNoRelation._1}%2.5f")
    println(f"R:  ${microWithoutNoRelation._2}%2.5f")
    println(f"F1: ${microWithoutNoRelation._3}%2.5f")

    println("\n\n")
    println(f"Correctly predicted (with no_relation):    ${resultsWithNoRel.count(it => it._1 == it._2)}")
    println(f"Predicted as a relation ${resultsWithNoRel.count(it => it._2 != "no_relation")}")

    val precision = resultsWithoutNoRel.count(it => it._1 == it._2).toDouble / resultsWithoutNoRel.count(_._2 != "no_relation").toDouble
    val recall    = resultsWithoutNoRel.count(it => it._1 == it._2).toDouble / resultsWithoutNoRel.count(_._1 != "no_relation").toDouble
    val f1        = 2 * (precision * recall) / (precision + recall)
    println(f"Correctly predicted (without no_relation): ${resultsWithoutNoRel.count(it => it._1 == it._2)} (p=$precision%2.5f r=$recall%2.5f f1=$f1%2.5f)")
    println(f"Predicted as a relation ${resultsWithoutNoRel.count(it => it._2 != "no_relation")}")
    println(f"Total (with no_relation):    ${resultsWithNoRel.size}")
    println(f"Total (without no_relation): ${resultsWithoutNoRel.size}")
    println("")
    println(f"Is gold inside (without no_relation): ${isResultIn.sum}")
    println("\n\n")
    println("#"*100)
    println("\n\n")
    println("-"*100)

    val (gold, predicted) = resultsWithNoRel.toList.unzip
    val (precisionMicro, recallMicro, f1Micro) = score(gold, predicted)
    println("\n\n")
    println("#"*100)
    println("TACRED scoring function")
    println("#"*100)
    println(f"P (micro):  ${precisionMicro}%2.5f")
    println(f"R (micro):  ${recallMicro}%2.5f")
    println(f"F1 (micro): ${f1Micro}%2.5f")
    println("\n\n")
    
    (precisionMicro, recallMicro, f1Micro, patterns.size)
  }

  /**
    * 
    *
    * @param patterns the collection of our rules (generated with TacredRuleGeneration)
    * @param ee an @see ExtractorEngine, holding the test sentence in memory
    * @param spec the specification of what to match; used to make sure that we match exactly that with the pattern
    * @param direction the direction of the current sentence (for which we want to predict the relation) used to
    *                  filter the patterns we apply, such that we apply only the patterns that match the direction
    * @return a sequence of (String, Int), where the String represents the relation predicted and the Int represents
    *         the number of times it was predicted (how many rules with relation equal to the String matched the sentence)
    */
  def predictRelations(patterns: Seq[TacredPattern], ee: ExtractorEngine, spec: Spec, direction: PatternDirection): Seq[(String, Double)] = {
    val predictions = patterns.par.filter(_.direction == direction).flatMap { case TacredPattern(pattern, relation, direction, weight) => 

      val odinsonPattern = ee.compiler.compile(pattern)
      val results        = ee.query(odinsonPattern)

      val matches = results.scoreDocs.flatMap(_.matches).map(it => (it.start, it.end))

      if (matches.contains((spec.start, spec.end))) {
        Some(relation, weight)
      } else {
        None
      }
    }.groupBy(_._1).mapValues(_.map(_._2).sum).toSeq
    
    predictions.toList.toSeq
  }

  def predictRelation(pattern: TacredPattern, ee: ExtractorEngine, spec: Spec, direction: PatternDirection): String = {
    if (pattern.direction != direction) {
      "no_relation"
    } else {
      val odinsonPattern = ee.compiler.compile(pattern.pattern)
      val results        = ee.query(odinsonPattern)

      val matches = results.scoreDocs.flatMap(_.matches).map(it => (it.start, it.end))

      if (matches.contains((spec.start, spec.end))) {
        pattern.relation
      } else {
        "no_relation"
      }
    }
  }
  
 
  /**
    * 
    *
    * @param goldAndPrediction a sequence of 2-tuple, where _1 is the gold label and _2 is the predicted label
    * @return 3-tuple, where _1 is precision, _2 is recall and _3 is f1
    */
  def microScores(goldAndPrediction: Seq[(String, String)]): (Double, Double, Double) = {
    val tp = goldAndPrediction.count { case (gold, predicted) => gold == predicted }.toDouble
    val fp = goldAndPrediction.count { case (gold, predicted) => gold != predicted }.toDouble
    val fn = goldAndPrediction.count { case (gold, predicted) => gold != predicted }.toDouble

    val p  = tp / (tp + fp)
    val r  = tp / (tp + fn)
    val f1 = 2 * (p*r)/(p+r)
    (p, r, f1)
    
  }

  /**
    * 
    *
    * @param key the collection of the gold relations
    * @param prediction the collection of the predicted relation
    * There should be a 1:1 match between them, each corresponding to the same text (same order)
    * 
    * The implementation is translated from https://github.com/yuhaozhang/tacred-relation/blob/834ee87b861c08979cd0b3ccfe65a63d0a7d70d8/utils/scorer.py
    * for consistency
    */
  def score(key: Seq[String], predictions: Seq[String]): (Double, Double, Double) = {
    val correctByRelation = mutable.Map.empty[String, Int].withDefaultValue(0)
    val guessedByRelation = mutable.Map.empty[String, Int].withDefaultValue(0)
    val goldByRelation    = mutable.Map.empty[String, Int].withDefaultValue(0)
    key.zip(predictions).foreach { case (gold, guess) =>
      if (gold == "no_relation" && guess == "no_relation") {
        // Nothing to do. Here for 1:1 consistency
      } else if (gold == "no_relation" && guess != "no_relation") {
        guessedByRelation(guess) += 1
      } else if (gold != "no_relation" && guess == "no_relation") {
        goldByRelation(gold) += 1
      } else if (gold != "no_relation" && guess != "no_relation") {
        guessedByRelation(guess) += 1
        goldByRelation(gold) += 1
        if (gold == guess) {
          correctByRelation(guess) += 1
        }
      }
    }

    val cbrSum  = correctByRelation.values.sum
    val gbrSum  = guessedByRelation.values.sum
    val goldSum = goldByRelation.values.sum

    val precMicro = if (gbrSum > 0) {
      cbrSum.toDouble / gbrSum.toDouble
    } else {
      1.0
    }

    val recallMicro = if (goldSum > 0) {
      cbrSum.toDouble / goldSum.toDouble
    } else {
      0.0
    }

    val f1Micro = (2.0 * precMicro * recallMicro) / (precMicro + recallMicro)

    return (precMicro, recallMicro, f1Micro)

  }

  def correctSeq2SeqTree(q: Query): Query = q match {
    case it@MatchAllQuery => it
    case it@HoleQuery => it
    case it@TokenQuery(constraint) => TokenQuery(correctSeq2SeqTree(constraint))
    case it@ConcatQuery(queries) => ConcatQuery(queries.map(correctSeq2SeqTree))
    case it@OrQuery(queries) => OrQuery(queries.map(correctSeq2SeqTree))
    case it@RepeatQuery(query, min, max) => it.copy(query=correctSeq2SeqTree(query))
  }
  def correctSeq2SeqTree(q: TokenConstraint): TokenConstraint = q match {
    case it@MatchAllConstraint =>  it
    case it@HoleConstraint => it
    case it@FieldConstraint(name, value) => { 
      (name, value) match {
        case (StringMatcher(n), StringMatcher(v)) => {
          if (n.toLowerCase.contains("tag")) FieldConstraint(StringMatcher(n.trim().toLowerCase()), StringMatcher(v.trim().replace(" ", "").toUpperCase()))
          else FieldConstraint(StringMatcher(n.trim().toLowerCase()), StringMatcher(v.trim().replace(" ", "").toLowerCase()))
        }
        case _ => throw new IllegalStateException("The pattern is incomplete")
      }
      // FieldConstraint(correctSeq2SeqTree(name), correctSeq2SeqTree(value)
    }
    case it@NotConstraint(constraint) => NotConstraint(correctSeq2SeqTree(constraint))
    case it@AndConstraint(constraints) => AndConstraint(constraints.map(correctSeq2SeqTree))
    case it@OrConstraint(constraints) => OrConstraint(constraints.map(correctSeq2SeqTree))
  }
  def correctSeq2SeqTree(q: Matcher): Matcher = q match {
    case it@MatchAllMatcher =>  it
    case it@HoleMatcher => it
    case it@StringMatcher(s) => StringMatcher(s.trim())
  }

}

object GenerateAllResults extends App {
  val resultLocations = Map(
    "2_128"              -> "/home/rvacareanu/projects/results/odinsynth_tacred_generated_rules/bert_models/normal/2_128/aggregated/all_solutions.tsv",
    "4_256"              -> "/home/rvacareanu/projects/results/odinsynth_tacred_generated_rules/bert_models/normal/4_256/aggregated/all_solutions.tsv",
    "4_512"              -> "/home/rvacareanu/projects/results/odinsynth_tacred_generated_rules/bert_models/normal/4_512/aggregated/all_solutions.tsv",
    "8_512"              -> "/home/rvacareanu/projects/results/odinsynth_tacred_generated_rules/bert_models/normal/8_512/aggregated/all_solutions.tsv",
    "bert_base"          -> "/home/rvacareanu/projects/results/odinsynth_tacred_generated_rules/bert_models/normal/bert_base/aggregated/all_solutions.tsv",
    "static"             -> "/home/rvacareanu/projects/results/odinsynth_tacred_generated_rules/static/normal/all_solutions.tsv",
    "static with reward" -> "/home/rvacareanu/projects/results/odinsynth_tacred_generated_rules/static/with_reward/all_solutions.tsv",
    "no learning (0, 1)" -> "/home/rvacareanu/projects/results/odinsynth_tacred_generated_rules/no_learning/aggregated/all_solutions.tsv",
    "seq2seq"            -> "/home/rvacareanu/projects/results/odinsynth_tacred_generated_rules/seq2seq_bert-base-uncased/all_solutions.tsv",
  )
  val distinctValues = Seq(
    true, 
    false
    )
  val scoringSchema = Seq(
    "equal",
    "specsize",
    "logspecsize",  
  )
  val pw = new PrintWriter(new File("/home/rvacareanu/projects/results/odinsynth_tacred_generated_rules/dev_results_bert_models_static_seq2seq_nolearning"))
  // pw.println("name\trules\tdistinct\tscoringSchema\tpatterns_size\tp\tr\tf1")
  pw.println(Seq("name", "rules", "distinct", "scoringSchema", "tpatterns_size", "p", "r", "f1").mkString("\t"))//"name\trules\tdistinct\tscoringSchema\tpatterns_size\tp\tr\tf1")
  for {
    (key, rl) <- resultLocations.toList
    dv        <- distinctValues
    ss        <- scoringSchema
  } {
    val config = ConfigFactory.empty
                           .withValue("odinsynth.evaluation.tacred.relationsPath", ConfigValueFactory.fromAnyRef(rl))
                           .withValue("odinsynth.evaluation.tacred.weightingScheme", ConfigValueFactory.fromAnyRef(ss))
                           .withValue("odinsynth.evaluation.tacred.distinctRules", ConfigValueFactory.fromAnyRef(dv))
                           .withValue("odinsynth.evaluation.tacred.odinsonDocumentCache", ConfigValueFactory.fromAnyRef("/home/rvacareanu/projects/odinsynth_cache/"))
                           .withValue("odinsynth.evaluation.tacred.devProcessed", ConfigValueFactory.fromAnyRef("/data/nlp/corpora/odinsynth/data/TACRED/tacred/data/json/dev_processed.json"))
    
    val (p, r, f1, patternSize) = TacredEvaluation.run(new TacredConfig(config))
    pw.println(Seq(key, rl, if (dv) "1" else "0", ss, patternSize.toString, p.toString(), r.toString(), f1.toString()).mkString("\t"))//"name\trules\tdistinct\tscoringSchema\tpatterns_size\tp\tr\tf1")
  }
  pw.close()
}
object GenerateLearningRatePlot extends App {
  val resultLocations = Map(
    // "8_512"              -> "/home/rvacareanu/projects/results/odinsynth_tacred_generated_rules/bert_models/normal/8_512/aggregated/all_solutions.tsv",
    "bert_base"          -> "/home/rvacareanu/projects/results/odinsynth_tacred_generated_rules/bert_models/normal/bert_base/aggregated/all_solutions.tsv",

  )
  val distinctValues = Seq(
    true, 
    // false
    )
  val scoringSchema = Seq(
    // "equal",
    // "specsize",
    "logspecsize",  
  )
  val path = Map(
    "dev" -> "/data/nlp/corpora/odinsynth/data/TACRED/tacred/data/json/dev_processed.json",
    "test" -> "/data/nlp/corpora/odinsynth/data/TACRED/tacred/data/json/test_processed.json",
  )

  for {
    (key, rl) <- resultLocations.toList
    dv        <- distinctValues
    ss        <- scoringSchema
    (k,p)     <- path
  } {
    val pw = new PrintWriter(new File(f"/home/rvacareanu/projects/results/odinsynth_tacred_generated_rules/bert_models/normal/bert_base/pattern_by_pattern_$k"))
    val config = ConfigFactory.empty
                           .withValue("odinsynth.evaluation.tacred.relationsPath", ConfigValueFactory.fromAnyRef(rl))
                           .withValue("odinsynth.evaluation.tacred.weightingScheme", ConfigValueFactory.fromAnyRef(ss))
                           .withValue("odinsynth.evaluation.tacred.distinctRules", ConfigValueFactory.fromAnyRef(dv))
                           .withValue("odinsynth.evaluation.tacred.odinsonDocumentCache", ConfigValueFactory.fromAnyRef("/home/rvacareanu/projects/odinsynth_cache/"))
                           .withValue("odinsynth.evaluation.tacred.devProcessed", ConfigValueFactory.fromAnyRef(p))
    
    val f1s = TacredEvaluation.runPatternByPattern(new TacredConfig(config))
    f1s.toList.foreach { f1 =>
      pw.println(f1)
    }
    pw.close()
  }
}
