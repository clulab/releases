package edu.arizona.sista.qa.word2vec

import scala.collection.mutable.ArrayBuffer

/**
  * Created by bsharp on 5/9/16.
  */
class CNNRelation (filenameTrainScores:String, filenameTestScores:String, filenameTrainInfo:String, filenameTestInfo:String) {

  val (trainScores, trainLabels) = loadDataFromFile(filenameTrainScores)
  val (testScores, testLabels) = loadDataFromFile(filenameTestScores)
  val trainCandidateIds = loadInfoFromFile(filenameTrainInfo)
  val testCandidateIds = loadInfoFromFile(filenameTestInfo)
  val allScores = trainScores ++ testScores
  val allLabels = trainLabels ++ testLabels
  val allIDs = trainCandidateIds ++ testCandidateIds


  // Retrieves the score for a particular docid in either the train or test folds
  def getScore(docid:String):Double = {
    val index = allIDs.indexOf(docid)
    allScores(index)
  }

  // Retrieves the score for a particular docid in either the train or test folds
  def getLabel(docid:String):Double = {
    val index = allIDs.indexOf(docid)
    allLabels(index)
  }


  // Returns the parallel arrays for the scores and the labels
  def loadDataFromFile(fn:String):(Array[Double], Array[Double]) = {
    val scores = new ArrayBuffer[Double]
    val labels = new ArrayBuffer[Double]
    // Open the file and read the info
    val source = scala.io.Source.fromFile(fn)
    val lines = source.getLines()
    // Parse and store
    for (line <- lines) {
      val fields = line.split("\t")
      if (fields.length != 2) println ("ERROR with line: " + line)
      assert (fields.length == 2)
      scores.append(fields(0).toDouble)
      labels.append(fields(1).toDouble)
    }

    source.close()
    (scores.toArray, labels.toArray)
  }

  // Loads the question/candidate id information, should be parallel to what is loaded in loadDataFromFile
  // Loads from the A files
  def loadInfoFromFile(fn:String):Array[String] = {
    val canidateIds = new ArrayBuffer[String]
    // Open the file and read the info
    val source = scala.io.Source.fromFile(fn)
    val lines = source.getLines()
    // Parse and store
    for (line <- lines) {
      val fields = line.split("\t")
      assert (fields.length > 2)
      canidateIds.append(fields(0))
    }

    source.close()
    canidateIds.toArray
  }

}


