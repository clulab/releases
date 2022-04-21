package org.clulab.odinsynth.evaluation.tacred

/**
  * Trait defining a "weight" method
  * The weight method takes one line in the pattern .tsv file
  * The header of that file can change, but one example of the header is:
  * \tspec_size\tmax_highlight_length\tmin_highlight_length\tsolved\tcluster_path\tpattern\tnum_steps\ttrials\twrong_matches\tdirection\trelation\tscorerVersion\tsolution_status"
  * Which is the same as TacredSolution.getHeaderWithIndex()
  * 
  * The methods, generally, care only about the "spec_size" field, which tells the number of specifications,
  * that is, the number of sentences used to generate the rule
  * 
  */
trait WeightingScheme {
  def weight(patternLine: Map[String, String]): Double
}

object EqualWeight extends WeightingScheme {
    def weight(patternLine: Map[String, String]): Double = 1.0
}

object SpecSizeWeight extends WeightingScheme {
    def weight(patternLine: Map[String, String]): Double = patternLine("spec_size").toDouble
}

object LogSpecSizeWeight extends WeightingScheme {
    def weight(patternLine: Map[String, String]): Double = Math.log(patternLine("spec_size").toDouble + 1)/Math.log(2)
}
