package org.clulab.clint

import scala.io.Source

/**
  * Created by ajaynagesh on 12/12/17.
  */
object AnalyseGPEBug extends App{

  val training_data = "/Users/ajaynagesh/Research/code/research/clint/emboot/data-ontonotes/training_data_with_labels_emboot.filtered.txt"
//  val promotedPatternsInterpretEmboot = "/Users/ajaynagesh/Research/code/papers/naacl2018-emboot/results/Nov29_2017.globalpromo/Nov27_2017_onto/emboot_I_global_promotion.txt_patterns.txt"
//  val promotedEntInterpretEmboot = "/Users/ajaynagesh/Research/code/papers/naacl2018-emboot/results/Nov29_2017.globalpromo/Nov27_2017_onto/emboot_I_global_promotion.txt"

  val promotedPatternsInterpretEmboot = "/Users/ajaynagesh/Research/code/papers/naacl2018-emboot/results/Dec13_gpelogs/Oct31_2017/emboot_interpretable_goldberg.log_patterns.txt"
  val promotedEntInterpretEmboot = "/Users/ajaynagesh/Research/code/papers/naacl2018-emboot/results/Dec13_gpelogs/Oct31_2017/emboot_interpretable_goldberg.log"

  val interpretOutputFile = "/Users/ajaynagesh/Research/code/papers/naacl2018-emboot/results/Dec13_gpelogs/Oct31_2017/emboot_interpretable_goldberg.log_interpretable_model.txt"

  val trainingDataset = Source.fromFile(training_data).getLines.toArray.map{ line =>
    val fields = line.split("\t")
    val label = fields(0)
    val entity = fields(1)
    val patterns = fields.takeRight(fields.length - 2) // Remaining are patterns
    (label, entity, patterns)
  }

  val promotedPatternsPerEpoch = Source.fromFile(promotedPatternsInterpretEmboot)
                                       .getLines.toArray.grouped(12).toArray//split into epochs
                                       .map {epochArray =>
                                                val epochId = epochArray(0)
                                                val labelPatternArray = epochArray.tail
                                                val label_patterns = labelPatternArray.map {lbl_pat=>
                                                  val fields = lbl_pat.split("\t")
                                                  val lbl = fields(0)
                                                  val patterns = fields.tail
                                                  (lbl,patterns)
                                                }
                                                (epochId, label_patterns)
                                        }

  val promotedEntPerEpoch = Source.fromFile(promotedEntInterpretEmboot)
    .getLines.toArray.grouped(12).toArray//split into epochs
    .map {epochArray =>
    val epochId = epochArray(0)
    val labelPatternArray = epochArray.tail
    val label_patterns = labelPatternArray.map {lbl_pat=>
      val fields = lbl_pat.split("\t")
      val lbl = fields(0)
      val patterns = fields.tail
      (lbl,patterns)
    }
    (epochId, label_patterns)
  }

  val trainSetFlatmap = trainingDataset.flatMap { data =>
    val (lbl, entity, patterns) = (data._1, data._2, data._3)
    patterns.map { p =>
      p -> (entity, lbl)
    }
  }

  val trainSetFlatmapEnt = trainingDataset.flatMap { data =>
    val (lbl, entity, patterns) = (data._1, data._2, data._3)
    patterns.map { p =>
      entity -> (p, lbl)
    }
  }

  val gpePatternsInPool = promotedPatternsPerEpoch.flatMap(_._2).filter(_._1.equals("GPE")).flatMap(_._2)
  val gpeEntitiesInPool = promotedEntPerEpoch.flatMap(_._2).filter(_._1.equals("GPE")).flatMap(_._2).map((_, "GPE"))

  val gpeEntTrainMatchinggpePatterns:Array[(String, String)] = trainSetFlatmap
                                                                  .filter(x => gpePatternsInPool.contains(x._1))
                                                                  .map((_._2))
                                                                  .toSet.toArray


  val gpeNotPresentInPool:Array[(String, String)] =  gpeEntTrainMatchinggpePatterns.filterNot(gpeEntitiesInPool.contains(_))
  val gpePresentInPool =  gpeEntTrainMatchinggpePatterns.filter(gpeEntitiesInPool.contains(_))


  for ( e_lbl <- gpeNotPresentInPool.groupBy(_._2).get("GPE").get) {
    val x = trainingDataset.filter(_._2.equals(e_lbl._1))
    println(s"${e_lbl}\t${x.length}")
  }

  for ( e_lbl <- gpePresentInPool.groupBy(_._2).get("GPE").get) {
    val x = trainingDataset.filter(_._2.equals(e_lbl._1))
    println(s"${e_lbl}\t${x.length}")
  }

  val patMatchingTaiwan = trainSetFlatmap.filter( _._2.equals  ("Taiwan", "GPE")).map(_._1)

  val patTaiwanCommon = gpePatternsInPool.intersect(patMatchingTaiwan)


  val xyx = promotedEntPerEpoch.map{epochInfo =>
    val entitiesInEpoch = epochInfo._2
    val hist = entitiesInEpoch.map(x => (x._1, x._2.length))
    (epochInfo._1, hist)
  }


  promotedEntPerEpoch.flatMap(_._2).map(x => (x._1, x._2.length)).groupBy(_._1).map(y => (y._1, y._2.map(_._2).sum))
  // Res:  Map(NORP -> 14, FAC -> 48, EVENT -> 22, WORK_OF_ART -> 24, GPE -> 514, PRODUCT -> 26, LANGUAGE -> 37, ORG -> 97, LAW -> 13, PERSON -> 893, LOC -> 72)

  val x = Source.fromFile(interpretOutputFile).getLines.toArray.map {line =>
    val fields = line.split("\t")
    val (epoch,entity,pred) = (fields(0),fields(1), fields(2))
    val scores = fields(3).replace("[ ", "").replace("]", "").split(" +").map(_.toDouble)
    (epoch, entity, pred, scores)
  }

  val interpretPredictions = promotedEntPerEpoch.flatMap(_._2).flatMap {x =>
    val lbl = x._1
    val ents = x._2.map(y => (y,lbl))
    ents
  }.toMap

  promotedPatternsPerEpoch.flatMap(_._2).flatMap {x =>
    val lbl = x._1
    val pats = x._2.map(y => (y, lbl))
    pats
  }

  val gpeEntsPreds = for (e_lbl <- gpeNotPresentInPool) yield {
    val e = e_lbl._1
    val truelbl = e_lbl._2
    val pred = if (interpretPredictions.contains(e)) interpretPredictions.get(e).get else "NIL"
    (e, truelbl, pred)
  }

  for (i <- gpeEntsPreds){
    println(i)
  }

  // Australia --> EVENT .. why ??
  val eventPatterns = trainingDataset.filter(_._1.equals("EVENT")).flatMap(_._3) // All event patterns
  gpePatternsInPool.intersect(eventPatterns)
  //Array(defeat in @ENTITY)
  // Now "defeat in @ENTITY" occurs as a GPE pattern but is present an EVENT entity as well "EVENT	Vietnam	defeat in @ENTITY"

  // Tokyo --> LAW .. why ??
  val lawPatterns = trainingDataset.filter(_._1.equals("LAW")).flatMap(_._3) // All law patterns
  gpePatternsInPool.intersect(lawPatterns)
  // Tokyo initally predicted as LAW (randomness of the emb initially ?? )
  // later predicted as GPE .. due to stratification we pick LAW
  // grep "\tTokyo\t" Oct31_2017/emboot_interpretable_goldberg.log_interpretable_model.txt

  //(Denver,GPE,PRODUCT)
  //  (Korea,GPE,PRODUCT)
  val prodPatterns = trainingDataset.filter(_._1.equals("PRODUCT")).flatMap(_._3)
  // patterns which occur with Denver
  // A bad pattern in PRODUCT ("News , @ENTITY") matches the entity Denver .. so Denver is assigned PRODUCT

  val epoch2Pat = promotedPatternsPerEpoch(1)._2
  val epoch1ProdEnt = promotedEntPerEpoch(1)._2.flatMap{x =>
    val lbl = x._1
    val ents = x._2.map((_,lbl))
    ents
  }.filter(_._2.equals("PRODUCT")).map(_._1)

  val p = promotedPatternsPerEpoch(1)._2.filter(_._1.equals("PRODUCT")).head._2.last // "News , @ENTITY"
  val epoch1ProdPatternsInTrain = trainSetFlatmapEnt.filter(x => epoch1ProdEnt.contains(x._1))

  epoch1ProdPatternsInTrain.filter(_._2._1.equals(p))
  //res9: Array[(String, (String, String))] =
  //      Array((Aden,(News , @ENTITY,GPE)),
  //            (Ramallah,(News , @ENTITY,GPE)),
  //            (Aden,(News , @ENTITY,GPE)))

}
