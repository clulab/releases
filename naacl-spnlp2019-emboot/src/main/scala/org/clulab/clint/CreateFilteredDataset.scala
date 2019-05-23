package org.clulab.clint

import java.io.{BufferedWriter, File, FileWriter}

import com.typesafe.config.ConfigFactory
import com.typesafe.scalalogging.LazyLogging
import ai.lum.common.ConfigUtils._

import scala.io.Source

/**
  * Created by ajaynagesh on 10/1/17.
  */
object CreateFilteredDataset extends App with LazyLogging {

  val config = ConfigFactory.load()
  val indexDir = config[File]("clint.index-dir")
  val patternCountThreshold = config[Int]("clint.patternCountThreshold")

  logger.info("loading pattern lexicon")
  val patCounts = LexiconBuilder.loadCounts(new File(indexDir, "patterns.counts"))
  val patLexicon = LexiconBuilder.loadLexicon(new File(indexDir, "patterns.lexicon")).inverseLexicon.toArray.map(x => (x._2,x._1) ).toMap

  logger.info("Loading the indices")
  val entToPatIndex = InvertedIndex.loadFrom(new File(indexDir, "entityToPatterns.index"))
  val patToEntIndex = InvertedIndex.loadFrom(new File(indexDir, "patternToEntities.index"))

  logger.info("loading the entity lexicon")
  val entCounts = LexiconBuilder.loadCounts(new File(indexDir, "entity.counts"))
  val entLexicon = LexiconBuilder.loadLexicon(new File(indexDir, "entity.lexicon")).inverseLexicon.toArray.map(x => (x._2,x._1) ).toMap
  // -----------------------------------------------------------------------------------------------------------
  // Filter out patterns which have counts < `patternCountThreshold`
  // -----------------------------------------------------------------------------------------------------------
  val patCountsFiltered = patCounts
                                .toArray
                                .filter(_._2 >= patternCountThreshold) // only select those patterns whose count >= threshold
                                .sortBy(_._1)

  val patTotalCountsFiltered = patCountsFiltered
    .unzip._2 // Get all the pattern counts
    .sum
  // -----------------------------------------------------------------------------------------------------------

  // -----------------------------------------------------------------------------------------------------------
  // "Writing `patterns.total.filtered`, `patterns.lexicon.filtered` and the `patterns.counts.filtered` files"
  // -----------------------------------------------------------------------------------------------------------
  logger.info("Writing `patterns.total.filtered`, `patterns.lexicon.filtered` and the `patterns.counts.filtered` files")
  val patCountsFilteredFile = new BufferedWriter(new FileWriter(new File(indexDir, "patterns.counts.filtered")))
  val patLexiconFilteredFile = new BufferedWriter(new FileWriter(new File(indexDir, "patterns.lexicon.filtered")))

  val patTotalCountsFilteredFile = new BufferedWriter(new FileWriter(new File(indexDir, "patterns.total.filtered")))
  patTotalCountsFilteredFile.write(s"${patTotalCountsFiltered}")
  patTotalCountsFilteredFile.close

  patLexiconFilteredFile.write(s"${patCountsFiltered.size}\n")
  val patLexiconFiltered = (for((pat, cnt) <- patCountsFiltered) yield {
    patLexiconFilteredFile.write(s"${pat}\t${patLexicon(pat)}\n")
    patCountsFilteredFile.write(s"${pat}\t${cnt}\n")
    (pat,patLexicon(pat))
  }).toMap
  patLexiconFilteredFile.close
  patCountsFilteredFile.close
  // -----------------------------------------------------------------------------------------------------------

  // -----------------------------------------------------------------------------------------------------------
  // Writing the `entityToPatterns.index.filtered` , `patternToEntities.index.filtered` and `entityId.patternId.counts`
  // -----------------------------------------------------------------------------------------------------------
  logger.info(s"Writing the `entityToPatterns.index.filtered` , `patternToEntities.index.filtered` and `entityId.patternId.counts`")
  // NOTE: Need to read the `entityId.patternId.counts` file and retain the counts of the filtered (entityId, patternId)
  val entPatternCounts = Source.fromFile(new File(indexDir, "entityId.patternId.counts")).getLines.toArray.map { line =>
                                                                        val fields = line.split(" ")
                                                                        val (entId, patId, cnt) = (fields(0).toInt, fields(1).toInt, fields(2).toInt)
                                                                        (entId, patId) -> cnt
                                                                    }.toMap
  val entToPatIndexFiltered = new InvertedIndex
  // [ (entId, patId, cnt) , (...), (...), ... ]
  val entPatternCountsFiltered = (for((entId, pats) <- entToPatIndex.toSeq) yield {
    val t2 = for(pat <- pats.toSeq) yield {
      val t1 = if(patLexiconFiltered.contains(pat)) {
        //Only if the filtered pattern lexicon contains this pattern ass to the index
        entToPatIndexFiltered.add(entId, pat)
        Some(entId, pat, entPatternCounts.get((entId,pat)).get)
      }
      else None

      t1
    }
    t2.flatten
  }).flatten


  entToPatIndexFiltered.saveTo(new File(indexDir, "entityToPatterns.index.filtered"))
  // NOTE: Not the right way to generate the `entityId.patternId.counts.filtered` file. (see above)
  //entToPatIndexFiltered.writeCounts(new File(indexDir, "entityId.patternId.counts.filtered"))
  val entPatternCountsFilteredFile =  new BufferedWriter(new FileWriter(new File(indexDir, "entityId.patternId.counts.filtered")))
  for ((eid,pid,cnt) <- entPatternCountsFiltered)
    entPatternCountsFilteredFile.write(s"${eid} ${pid} ${cnt}\n")
  entPatternCountsFilteredFile.close

  val patToEntIndexFiltered = new InvertedIndex
  for((patId, ents) <- patToEntIndex){
    if(patLexiconFiltered.contains(patId)) // Only if the filtered pattern lexicon contains this pattern ass to the index
      for(ent <- ents)
        patToEntIndexFiltered.add(patId, ent)
  }
  patToEntIndexFiltered.saveTo(new File(indexDir, "patternToEntities.index.filtered"))
  // -----------------------------------------------------------------------------------------------------------

  // -----------------------------------------------------------------------------------------------------------
  // IMPT NOTE (w.r.t CoNLL dataset)
  // -----------------------------------------------------------------------------------------------------------
  // Total number of entities in lexicon = 8082
  // Entities that appear in the inverted index = 7543 (remaining entities are missing since they do not appear with any pattern: TODO: confirm this)
  // Entities that appear in the filtered lexicon = 5522 (since the patterns have been filtered and the entities that do not make it have 0 patterns associated with it after the filtering of patterns)
  // -----------------------------------------------------------------------------------------------------------

  // -----------------------------------------------------------------------------------------------------------
  // Generate the `entity.lexicon.filtered`, `entity.counts.filtered` and `entity.total.filtered`
  // -----------------------------------------------------------------------------------------------------------
  logger.info(s"Regenerating the `entity.lexicon.filtered`, `entity.counts.filtered` and `entity.total.filtered`")
  // [ (entId, entStr, cnt) , (...) , (...) ]
  val entLexiconFiltered = InvertedIndex.loadFrom(new File(indexDir, "entityToPatterns.index.filtered")) // Load the newly generated index of ent -> patterns
                              .keys.toArray // Get all the keys -- these are the entities that are remaining and that should go into the filtered lexicon
                                .map{ entId =>
                                  val entStr = entLexicon(entId)
                                  val cnt = entCounts(entId)
                                  (entId, entStr, cnt)
                                }.sortBy(_._1) // sort by entity id to compare the filtered and unfiltered files easily

  val entLexiconFilteredTotal = entLexiconFiltered // from the new filtered lexicon
                          .map(_._3) // get the counts
                          .sum // add them up

  val entCountsFilteredFile = new BufferedWriter(new FileWriter(new File(indexDir, "entity.counts.filtered")))
  val entLexiconFilteredFile = new BufferedWriter(new FileWriter(new File(indexDir, "entity.lexicon.filtered")))

  val entTotalCountsFilteredFile = new BufferedWriter(new FileWriter(new File(indexDir, "entity.total.filtered")))
  entTotalCountsFilteredFile.write(s"${entLexiconFilteredTotal}")
  entTotalCountsFilteredFile.close

  entLexiconFilteredFile.write(s"${entLexiconFiltered.size}\n")

  for((entId, entStr, entCnt) <- entLexiconFiltered) {
    entLexiconFilteredFile.write(s"${entId}\t${entStr}\n")
    entCountsFilteredFile.write(s"${entId}\t${entCnt}\n")
  }
  entLexiconFilteredFile.close
  entCountsFilteredFile.close
  // -----------------------------------------------------------------------------------------------------------

  // -----------------------------------------------------------------------------------------------------------
  // Regenerate the Emboot files
  // -----------------------------------------------------------------------------------------------------------
  logger.info("Begin pruning the emboot files")

  // -----------------------------------------------------------------------------------------------------------
  //pattern_vocabulary_emboot.txt
  logger.info("Creating `pattern_vocabulary_emboot.filtered.txt`")
  val patternVocabEmbootFiltered = Source.fromFile(new File(indexDir,"pattern_vocabulary_emboot.txt")).getLines.toArray
               .tail //do not consider the 1st line "</s>\t0"
               .map{ line => //map each line to (patString, count)
                  val fields = line.split("\t")
                  val (patString:String, count:String) = (fields(0), fields(1))
                  (patString,count.toInt)
               }
               .filter( _._2 >= patternCountThreshold) //filter those patterns whose counts are above the threshold
               .toMap

  val patternVocabEmbootFilteredFile = new BufferedWriter(new FileWriter(new File(indexDir, "pattern_vocabulary_emboot.filtered.txt")))
  patternVocabEmbootFilteredFile.write("</s>\t0\n")
  for ((pat,cnt) <- patternVocabEmbootFiltered.toArray.sortBy(- _._2)){
    patternVocabEmbootFilteredFile.write(s"${pat}\t${cnt}\n")
  }
  patternVocabEmbootFilteredFile.close
  // -----------------------------------------------------------------------------------------------------------
  //entity_vocabulary.emboot.txt
  logger.info("Creating `entity_vocabulary.emboot.filtered.txt`")
  val entVocabEmbootFiltered = entLexiconFiltered.map(x => (x._2,x._3)).toMap

  val entVocabEmbootFilteredFile = new BufferedWriter(new FileWriter(new File(indexDir, "entity_vocabulary.emboot.filtered.txt")))
  entVocabEmbootFilteredFile.write("</s>\t0\n")
  for((ent,cnt) <- entVocabEmbootFiltered.toArray.sortBy(- _._2)){
    entVocabEmbootFilteredFile.write(s"${ent}\t${cnt}\n")
  }
  entVocabEmbootFilteredFile.close
  // -----------------------------------------------------------------------------------------------------------
  //training_data_with_labels_emboot.txt
  logger.info("Creating `training_data_with_labels_emboot.filtered.txt`")

  val trainingDataEmbootFiltered = Source.fromFile(new File(indexDir, "training_data_with_labels_emboot.txt")).getLines.toArray
                .map { line =>
                  val fields = line.split("\t")
                  val (lbl,entity,patterns) = (fields(0), fields(1), fields.slice(2,fields.length))
                  val patternsNew = patterns.filter(patternVocabEmbootFiltered.contains(_)) // Only those patterns which are present in the filtered pattern lexicon
                  (lbl, entity, patternsNew)
                }.filter(trainingDatum => entVocabEmbootFiltered.contains(trainingDatum._2)) // Only those datums that have entity in the filtered entity lexicon
                 .filter(datum => datum._3.length > 0) // Also only those mentions which have a non-zero number of patterns associated with it in its context

  val trainingDataEmbootFilteredFile = new BufferedWriter(new FileWriter(new File(indexDir, "training_data_with_labels_emboot.filtered.txt")))
  for ((lbl, ent, patterns) <- trainingDataEmbootFiltered) {
    trainingDataEmbootFilteredFile.write(s"${lbl}\t${ent}\t${patterns.mkString("\t")}\n")
  }
  trainingDataEmbootFilteredFile.close
  // -----------------------------------------------------------------------------------------------------------
  //pattern_labels_emboot.txt
  logger.info("Creating `pattern_labels_emboot.filtered.txt`")
  val patLabelsEmbootFiltered = Source.fromFile(new File(indexDir,"pattern_labels_emboot.txt" )).getLines.toArray
                .map {line =>
                  val fields = line.split("\t")
                  val (pat, lbl) = (fields(0), fields(1))
                  (pat, lbl)
                }.filter(p => patternVocabEmbootFiltered.contains(p._1)) // Only those patterns which are present in the filtered pattern lexicon

  val patLabelsEmbootFilteredFile = new BufferedWriter(new FileWriter(new File(indexDir, "pattern_labels_emboot.filtered.txt")))
  for((pat,lbl) <- patLabelsEmbootFiltered) {
    patLabelsEmbootFilteredFile.write(s"${pat}\t${lbl}\n")
  }
  patLabelsEmbootFilteredFile.close
  // -----------------------------------------------------------------------------------------------------------
  //entity_labels_emboot.txt
  logger.info("Creating `entity_labels_emboot.filtered.txt`")
//  val entLabelsEmbootFiltered = Source.fromFile(new File(indexDir, "entity_labels_emboot.txt" )).getLines.toArray
//                                    .map{ line =>
//                                      val fields = line.split("\t")
//                                      val (ent, lbl) = (fields(0), fields(1))
//                                      (ent, lbl)
//                                    }.filter(e => entVocabEmbootFiltered.contains(e._1)) // Only those entities are present in the filtered entity lexicon

  val entLabelsEmbootFilteredFile = new BufferedWriter(new FileWriter(new File(indexDir, "entity_labels_emboot.filtered.txt")))
  for((lbl,ent,_) <- trainingDataEmbootFiltered){
    entLabelsEmbootFilteredFile.write(s"${ent}\t${lbl}\n")
  }
  entLabelsEmbootFilteredFile.close
  // -----------------------------------------------------------------------------------------------------------

  //entity_label_counts_emboot.txt
  logger.info("Creating `entity_label_counts_emboot.filtered.txt`")
  val entLabelCountsEmbootFiltered = trainingDataEmbootFiltered.map(datum => (datum._2,datum._1)) // (entity, label)
                                                .map(x => (x,1)) // used to get the counts
                                                .groupBy(_._1) //group by (entity, label)
                                                .map(y => (y._1, y._2.length)) // get (entity,label) --> counts

  val entLabelCountsEmbootFilteredFile = new BufferedWriter(new FileWriter(new File(indexDir, "entity_label_counts_emboot.filtered.txt")))
  for (((ent,lbl),cnt) <- entLabelCountsEmbootFiltered) {
    entLabelCountsEmbootFilteredFile.write(s"${cnt}\t${ent}\t${lbl}\n")
  }
  entLabelCountsEmbootFilteredFile.close
  // -----------------------------------------------------------------------------------------------------------

  // TODO: Do I have to close the files opened using Source.fromFile() ?

}