package edu.arizona.sista.embeddings.postprocessing

import java.io.PrintWriter

import preprocessing.agiga.ProcessAgiga._

/**
  * Created by bsharp on 9/16/16.
  */
object ProcessPairsForAI2 extends App {

  val mentionsDir = "/lhome/bsharp/causal/causalOut_mar30"
  val mentionsFiles = findFiles(mentionsDir, ".argsC")

  for (file <- mentionsFiles) {
    // Read the file
    val source = scala.io.Source.fromFile(file)
    val lines = source.getLines().toArray
    source.close()

    val outfileName = file.getAbsolutePath + ".tsv"
    val pw = new PrintWriter(outfileName)
    for (line <- lines) {
      val cols = line.split("\t")
      val causes = cols(0).split(",").map(cause => cause.replaceAll(" ", ","))
      val effects = cols(2).split(",").map(effect => effect.replaceAll(" ", ","))
      for {
        c <- causes
        e <- effects
      } pw.println(s"$c\t$e")
    }
    pw.close()
  }


}
