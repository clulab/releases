package org.clulab.clint

import java.io.{BufferedWriter, File, FileWriter}

import com.typesafe.config.ConfigFactory
import org.clulab.clint.EPB.Index
import org.clulab.embeddings.word2vec.Word2Vec
import ai.lum.common.ConfigUtils._
import ai.lum.common.StringUtils._

import scala.io.Source

/**
  * Created by ajaynagesh on 3/8/18.
  */
object QualitativeInterpretability extends App {

  // -----------------------------------------------------------------------------------------
  //1. Load entity,pattern embed, entity,pattern pools, Load dataset and maps
  val entityEmbedFile = "results/Mar8_SEM/Dec15_2017_conll_debug/emboot_I_global_promotion.txt_entemb.txt" // "results/Mar8_SEM/Dec15_2017_onto_debug/emboot_I_global_promotion.txt_entemb.txt" //
  val patternEmbedFile = "results/Mar8_SEM/Dec15_2017_conll_debug/emboot_I_global_promotion.txt_ctxemb.txt" // "results/Mar8_SEM/Dec15_2017_onto_debug/emboot_I_global_promotion.txt_ctxemb.txt" //
  val entPoolFile = "results/Mar8_SEM/Dec15_2017_conll_debug/emboot_I_global_promotion.txt" // "results/Mar8_SEM/Dec15_2017_onto_debug/emboot_I_global_promotion.txt" //
  val patPoolFile = "results/Mar8_SEM/Dec15_2017_conll_debug/emboot_I_global_promotion.txt_patterns.txt" // "results/Mar8_SEM/Dec15_2017_onto_debug/emboot_I_global_promotion.txt_patterns.txt" //
  val outputFile = "results/Mar8_SEM/QualitativeInterpretability_EPB_Conll.txt"
  val isEPB: Boolean = true

  val config = ConfigFactory.load()
  val indexDir = config[File]("clint.index-dir") // Directory which contains the index of entities and patterns
  val goldLabelsFile:File =  config[File]("clint.goldLabelsFile")

  val w2vVectorsFile = config[String]("clint.w2vVectors")
  lazy val w2v = new Word2Vec(w2vVectorsFile, None)

  // -----------------------------------------------------------------------------------------

  val wordLexicon = LexiconBuilder.loadLexicon(new File(indexDir, "word.lexicon"))
  val entityLexicon = LexiconBuilder.loadLexicon(new File(indexDir, "entity.lexicon.filtered"))
  val patternLexicon = LexiconBuilder.loadLexicon(new File(indexDir, "patterns.lexicon.filtered"))
  val entityToPatterns = Index.loadFrom(new File(indexDir, "entityToPatterns.index.filtered"))
  val patternToEntities = Index.loadFrom(new File(indexDir, "patternToEntities.index.filtered"))

  // -----------------------------------------------------------------------------------------

  val w2v_entityEmbed = if (isEPB) { //load lexicon from goldberg embeddings
    println("[entity] Constructing the embeddings from pretrained vectors")
    new Word2Vec( entityLexicon.inverseLexicon.map { entity_id =>
      val entity = entity_id._1
      val entityVector = w2v.makeCompositeVector(entity.split(" +").map( Word2Vec.sanitizeWord(_) )) //IMPT: to sanitise the words before comparing with goldberg embeddings
      (entity,entityVector)
    })
  } else { //construct lexicon from the custom embeddings learned from Emboot
    println("[entity] Constructing the embeddings from Emboot-trained vectors")
    val entityEmbed = Source.fromFile(entityEmbedFile).getLines().toArray.map {line =>
      val fields = line.split("\t")
      val entity = fields(0)
      val embed = fields(1).split(" ").map(_.toDouble)
      (entity, embed)
    }.toMap
    new Word2Vec(entityEmbed)
  }

  val w2v_patEmbed = if(isEPB){ //load lexicon from goldberg embeddings
    println("[pattern] Constructing the embeddings from pretrained vectors")
    new Word2Vec( patternLexicon.inverseLexicon.map{ pat_id =>
      val pat = pat_id._1.splitOnWhitespace.map {
        case "@" => "@ENTITY"
        case n => wordLexicon(n.toInt)
      }.mkString(" ")
      val patVector = w2v.makeCompositeVector(pat.split(" +").map( Word2Vec.sanitizeWord(_) )) //IMPT: to sanitise the words before comparing with goldberg embeddings
      (pat, patVector)
    } )
  } else { //construct lexicon from the custom embeddings learned from Emboot
    println("[pattern] Constructing the embeddings from Emboot-trained vectors")
    val patternEmbed = Source.fromFile(patternEmbedFile).getLines().toArray.map {line =>
      val fields = line.split("\t")
      val pattern = fields(0)
      val embed = fields(1).split(" ").map(_.toDouble)
      (pattern, embed)
    }.toMap
    new Word2Vec(patternEmbed)
  }

  val entPool = Source.fromFile(entPoolFile).getLines.toArray.filterNot(_.contains("Epoch")).map {line =>
    val fields = line.split("\t")
    val lbl = fields.head
    val entities = fields.tail
    (lbl, entities)
  }.groupBy(_._1).map{ x =>
    val lbl = x._1
    val entities = x._2.flatMap(_._2)
    (lbl, entities)
  }

  val patPool = Source.fromFile(patPoolFile).getLines.toArray.filterNot(_.contains("Epoch")).map {line =>
    val fields = line.split("\t")
    val lbl = fields.head
    val entities = fields.tail
    (lbl, entities)
  }.groupBy(_._1).map{ x =>
    val lbl = x._1
    val entities = x._2.flatMap(_._2)
    (lbl, entities)
  }

  val goldLabels = Source.fromFile(goldLabelsFile).getLines.map {l =>
    val tmp = l.split("\t")
    val k = tmp(0)
    val v = tmp.tail
    k -> v }.toMap

  val categories = goldLabels.keys.toArray

  // 2. Compute the avg entity vectors for each category

  val  avgEntityVectors = entPool.map{ label_entities =>
    val avgVector = w2v_entityEmbed.makeCompositeVector(label_entities._2) // So that we get normalized averaged vectors
    (label_entities._1, avgVector)
  }

  // 3. For each candidate entity, get its matching (patterns, score), where score is a prob. distrib,
  //    computed from softmax(cos(pattern, category_centroid)), (category centroid from step 2)

  val entityMatchingPatternScores = entityLexicon.inverseLexicon.map{ entity_id =>
    val entity = entity_id._1
    val entityId = entity_id._2

    val patternIds_matching_entity = entityToPatterns.index.get(entityId).get
    val patterns_matching_entity = patternIds_matching_entity.map { patId =>
      val pattern_wordIds = patternLexicon.get(patId).get
      val patternString = pattern_wordIds.splitOnWhitespace.map {
        case "@" => "@ENTITY"
        case n => wordLexicon(n.toInt)
      }.mkString(" ")
      patternString
    }
    // Patterns matching entity and present in the pattern pool
    val patterns_in_pool_matching_entity = patterns_matching_entity.intersect(patPool.values.flatten.toSet)

    val patterns_SoftMaxScores =  patterns_in_pool_matching_entity.map { pat =>
      val patternVector = w2v_patEmbed.makeCompositeVector(Seq(pat)) // So that we get normalized vectors
      val cosineScore = avgEntityVectors.map{ lbl_vector =>
        val lbl = lbl_vector._1
        val avgVector = lbl_vector._2
        val dotprod = Word2Vec.dotProduct(avgVector, patternVector)
        (lbl, dotprod)
      }
      val expScore = cosineScore.map {lbl_score =>
        val lbl = lbl_score._1
        val score = lbl_score._2
        (lbl, math.exp(score))
      }
      val denom = expScore.map(_._2).sum
      val softMaxScore = expScore.map(x => (x._1, x._2/denom))

      //(pat, softMaxScore.toArray.sortBy(- _._2))
      (pat, softMaxScore)
    }

    val noisyOrScore = categories.map { c =>
      val one_minus_prob_c = patterns_SoftMaxScores.map(x => (1 - x._2.get(c).get)).toArray
      (c, 1 - one_minus_prob_c.product)
    }.maxBy(_._2)

    (entity, patterns_SoftMaxScores, noisyOrScore)
  }

  val entityGoldLabels = goldLabels.map{lbl_entities =>
    val lbl = lbl_entities._1
    val entities = lbl_entities._2.map(x => (x,lbl))
    entities
  }.flatten.toArray.groupBy(_._1).map{x =>
    val entity = x._1
    val lbl_counts = x._2.groupBy(_._2).map(lbl_cnt => (lbl_cnt._1, lbl_cnt._2.length))
    val max_lbl = lbl_counts.toArray.sortBy(- _._2).head._1
    //(entity, (lbl_counts, max_lbl))
    (entity, max_lbl)
  }

    // 4. Return:
      //  1. prediction (computed using noisy-or),
      //  2. patterns from step 3. (also score in decreasing order of category?)
      //  3. gold mention label from the dataset
  val bw = new BufferedWriter(new FileWriter(new File(outputFile)))
  bw.write("Entity\tPrediction\tGoldLabel\tPatternScores\n")
  println(s"Writing the output file to ${outputFile}")
  for ((entity, matchingPatternsScores, noisyor_score) <- entityMatchingPatternScores){

    val prediction = noisyor_score._1
    val goldLabel = entityGoldLabels.get(entity).get
    val patternsPrintable = matchingPatternsScores.map{pat_scores =>
      s"${pat_scores._1}_[${pat_scores._2.toArray.sortBy(- _._2).mkString(",")}]"
    }.toArray

    val filter_entities = Seq("Leonid Kuchma", "Sofia", "Irani")

    if (filter_entities.contains(entity)) {
      println(s"${entity}: ")
      for (((pattern, scores), idx) <- matchingPatternsScores.zipWithIndex){
        println(s"  ${idx+1})  ${pattern}\n  \t${scores}\n----------")
      }
      val full_noisyOr = categories.map { c =>
        val one_minus_prob_c = matchingPatternsScores.map(x => (1 - x._2.get(c).get)).toArray
        (c, 1 - one_minus_prob_c.product)
      }.map(x => (s"${x._1}, ${x._2}", x._2 ))
      println(s"NoisyOr-score: [${full_noisyOr.map(_._1).mkString("; ")}]\tMax: ${full_noisyOr.maxBy(_._2)._1}")

    }

    bw.write(s"${entity}\t${prediction}\t${goldLabel}\t${patternsPrintable.mkString(", ")}\n")

  }
  bw.close()
}
