package edu.arizona.sista.embeddings.postprocessing

import edu.arizona.sista.struct.Counter

/**
  * Created by bsharp on 4/22/16.
  */
object AnalyzeInference extends App {

  def getOccur(filename:String, side:Int):Counter[String] = {
    val coOccur = new Counter[String]
    val source = scala.io.Source.fromFile(filename)
    val lines = source.getLines()

    for (line <- lines) {
      val sideChosen = line.split("-->")(side)
      // Grab just the words (not the POS)
      val tokens = sideChosen.split(" ").map(wl => wl.split("_")(0).toLowerCase().trim())
      tokens.foreach(coOccur.incrementCount(_))
    }

    source.close()
    coOccur
  }


  val causefile = "/lhome/bsharp/platform.txt"
  val effectfile = "/lhome/bsharp/scaffold.txt"

  val coCauses = getOccur(causefile, 0).toSeq.sortBy(- _._2)
  val coEffects = getOccur(effectfile, 1).toSeq.sortBy(- _._2)
  val causesScaffold = getOccur(effectfile, 0).toSeq.sortBy(- _._2)
  val platformCauses = getOccur(causefile, 1).toSeq.sortBy(- _._2)

  println ("CoCauses: " + coCauses.mkString(", "))
  println ("CoEffects: " + coEffects.mkString(", "))
  println ("Causes scaffold: " + causesScaffold.mkString(", "))
  println ("Platform causes: " + platformCauses.mkString(", "))

}
