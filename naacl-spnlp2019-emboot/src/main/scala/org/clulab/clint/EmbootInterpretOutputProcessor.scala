package org.clulab.clint

import java.io._

import scala.io._

/**
  * Created by ajaynagesh on 10/20/17.
  */
object EmbootInterpretOutputProcessor extends App {

  //// ---------- TO CHANGE (Conll/Ontonotes) ----------------------------
  //// TODO: Parameterize this
  val interpretableOutputFile ="emboot/pools_output_interpretable.txt_interpretable_model.txt"
  // TODO: not sure what this is...
  val toScoreFileStratified = "emboot/Emboot_int"

  // Conll
   val categories = Array("PER", "ORG", "LOC", "MISC").sorted
  // Ontonotes
//   val categories = Array(
//   "EVENT",
//   "FAC",
//   "GPE",
//   "LANGUAGE",
//   "LAW",
//   "LOC",
//   "NORP",
//   "ORG",
//   "PERSON",
//   "PRODUCT",
//   "WORK_OF_ART").sorted

  // TODO: Only for Ontonotes runs of the Interpretable output
  val file = "emboot/emboot_I_modpmi_logfreq.txt_interpretable_model.txt"
  val linesToWrite = Source.fromFile(interpretableOutputFile).getLines.toArray.grouped(2).toArray.map(m => s"${m(0)}${m(1)}")

  val writer = new FileWriter(new File(interpretableOutputFile))

  for (l <- linesToWrite) {
    writer.write(s"${l}\n")
  }

  writer.close
  // ---------- TO CHANGE (Conll/Ontonotes) ----------------------------

  // Each line is of the form
  //  "1       Richard Breeden PERSON  [ 0.16640232  0.16082305  0.15665299  0.15998637  0.15986372  0.15794857  0.15606606  0.23227195  0.2330059   0.16163346  0.16502617]"
  // epochid <tab> entity <tab> prediction <tab> noisy_or scores
  val predictions =  Source.fromFile(interpretableOutputFile).getLines().toArray.map{ line =>
    val fields = line.split("\t")
    val epoch = fields(0).toInt
    val entity = fields(1)
    val predLabel = fields(2)
    val scoresArray =  categories.zip(fields(3).replace("[","").replace("]", "").trim.split(" +").map(_.toDouble)).toMap
    (epoch, entity, predLabel, scoresArray.get(predLabel).get)
  }

  // group predictions by epoch
  val predictionPerEpoch = predictions
                              .groupBy(_._1) // group by the epoch id
                              .map(p => (p._1, p._2.map( x => (x._2, x._3, x._4) )) ) // convert this to a map from epoch_id -> (entity, predLabel, predScore)
                              .toArray // convert to array
                              .sortBy(_._1) // sort by epoch id

  // remove predictions which are already present in previous epochs
  val uniqPredictionPerEpoch =
    (for (i <- Range(0,predictionPerEpoch.size)) yield {
      val epoch = predictionPerEpoch(i)._1
      val preds = predictionPerEpoch(i)._2

      // all the entities predicted in the previous epochs
      val prevPredEntities = (for(j <- Range(0,i)) yield {
          predictionPerEpoch(j)._2.map(_._1)
        }).toArray.flatten

      // remove those entities which are already predicted in the previous epochs and sort by their noisy-or score
      val newPreds = preds.filterNot(p => prevPredEntities.contains(p._1)).sortBy(- _._3)
      (epoch, newPreds)
  }).toArray

  val globalPreds = uniqPredictionPerEpoch
                      .flatMap(_._2) // flatten all the predictions from the different epochs in one big array
                      .groupBy(_._2) // groupBy each predLabel (category)
                      .map(p => (p._1, p._2.map( x => (x._1, x._3))  )) // map from predLabel -> (entity, score)

  val globalPredsInFakeEpochs = globalPreds.map{ cat_preds =>
    val cat = cat_preds._1 // for each predLabel
    val preds = cat_preds._2.grouped(10).toArray // bin the set of predictions into groups of 10 and assign fake epoch-ids to them (next step)

    (cat, preds.zipWithIndex.map(p => (p._2, p._1))) // create a map from predLabel -> [1 -> [(ent, score), (ent, score), (ent, score) ...] [2 -> ...]  stratified predictions with fake epoch-ids
  }

  // determine how many stratifications (epochs) are created by look at the maximum sized bin for all categories
  val maxFakeEpochs = globalPredsInFakeEpochs.map(_._2.size).max

  val outWriterStratified = new BufferedWriter(new FileWriter(new File(toScoreFileStratified)))
  println(s"maxFakeEpochs : ${maxFakeEpochs}")
  for (i <- Range(0, maxFakeEpochs)){
    outWriterStratified.write(s"Epoch ${i}\n")
    for (c <- categories){
//      println(s"Category : $c")
      if(globalPredsInFakeEpochs.get(c).isDefined) {
        val preds = globalPredsInFakeEpochs.get(c).get
        if (i < preds.size) {
          val preds_i = (preds(i))._2.map(_._1).mkString("\t")
          outWriterStratified.write(s"${c}\t${preds_i}\n")
        }
        else { // if the epoch-id overshoots max number of bins for the category
          outWriterStratified.write(s"${c}\n")
        }
      }
      else{ // if no entity is predicted against the category
        outWriterStratified.write(s"${c}\n")
      }
    }
    outWriterStratified.write(s"\n")
  }
  outWriterStratified.close

}
