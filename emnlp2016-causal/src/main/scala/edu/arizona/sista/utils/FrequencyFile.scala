package edu.arizona.sista.utils

/**
 * Created by dfried on 12/18/14.
 */
object FrequencyFile {
  /**
   * read in a file of space-separated word count pairs, of the type produced by GigawordContentFreq,
   * and return a set of the words.  if minCount is defined, only take those words with frequency >= that count
   * If numToTake is defined, only take the first N words (with frequency >= the threshold, if that is also defined)
   */
  def parseFrequencyFile(fileName: String, numToTake: Option[Int] = None, minCount: Option[Int] = None): Set[String] = {
    val lines = io.Source.fromFile(fileName).getLines
    val it = for {
      line <- lines
      spl = line.split("\\s+")
      if (minCount.isEmpty || spl(1).toInt >= minCount.get)
    } yield spl(0)
    (numToTake match {
      case None => it
      case Some(k) => it.take(k)
    }).toSet
  }

}
