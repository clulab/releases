package org.clulab.odinsynth.evaluation.tacred

// import scala.util.Try
import org.clulab.odinsynth._
// import org.clulab.odinsynth.HoleQuery


/**
  *
  * 
  * @param numberOfSentences    - how many sentences did the cluster contain
  * @param maxHighlightedLength - max number of tokens highlighted
  * @param minHighlightedLength - min number of tokens highlighted
  * @param clusterPath          - path to the claster of which this tentative solution is created
  * @param numberOfSteps        - number of steps it took to obtain solution (can be None, case in which we reached the maximum number of allowed is reached)
  * @param solution             - the querry pattern. It is an optional, as when it is None we didn't find a solution
  * @param wrongMatches         - the number of wrong matches this solution achieves. It can be None
  * @param relation             - the relation associated with this cluster
  * @param trials               - how many times we have tried this cluster (@see clusterPath)
  * @param direction            - either subj <..> obj or obj <..> subj
  * @param relation             - the relation with which will this solution be associated
  * @param scorerVersion        - the scorer version (either Static, in which case we print the weights, either Dynamic, in which case we print the checkpoint)
  * @param solutionStatus       - whether this solution failed or not; A solution can be failed when an exception was thrown, case in which we should retry;
  *                               It is not failed if do not find a solution because we exhausted the maximum number of steps allowed
  */
case class TacredSolution(
                        numberOfSentences: Int, 
                        maxHighlightedLength: Int, 
                        minHighlightedLength: Int, 
                        clusterPath: String,
                        numberOfSteps: Int,
                        solution: Option[String],
                        wrongMatches: Option[Int],
                        trials: Int,
                        direction: PatternDirection,
                        relation: String,
                        scorerVersion: String,
                        solutionStatus: SolutionStatus,
                        ) {
                      
  def getString(): String = {
    val solved = if(solution.isDefined) 1                             else 0
    val p      = if(solution.isDefined) solution.get                  else ""
    val wm     = if(wrongMatches.isDefined) wrongMatches.get.toString else ""
    return f"${numberOfSentences}\t${maxHighlightedLength}\t${minHighlightedLength}\t${solved}\t${clusterPath}\t${p}\t${numberOfSteps}\t${trials}\t${wm}\t${direction.intValue}\t${relation}\t${scorerVersion}\t${solutionStatus.intValue}"
  }

}
object TacredSolution {
  def getHeaderWithIndex(): String = {
    f"\tspec_size\tmax_highlight_length\tmin_highlight_length\tsolved\tcluster_path\tpattern\tnum_steps\ttrials\twrong_matches\tdirection\trelation\tscorerVersion\tsolution_status"
  }

  def getHeaderWithoutIndex(): String = getHeaderWithIndex().tail

  /**
    * Checks if the string received as parameter is a valid solution
    * 
    * @param string: The string we wish to chech for validity
    * @return
    */
  def checkValidString(string: String): Boolean = {
    val split = string.split("\t")
    // Check the number of items matches. If yes, it is valid
    split.size == getHeaderWithoutIndex().split("\t").size
    // Alternatively, a more comprehensive (and more expensive) check would be: 
    // val wrongMatchesValid = if (split(8) == "") true else Try(split(8).toInt).isSuccess
    // val isValidSolution = Try(split(0).toInt).isSuccess && Try(split(1).toInt).isSuccess && Try(split(2).toInt).isSuccess && Try(split(3).toInt).isSuccess && Try(split(6).toInt).isSuccess && Try(split(7).toInt).isSuccess && wrongMatchesValid && Try(split(9).toInt).isSuccess
    // split.size == getHeaderWithoutIndex().split("\t").size && isValidSolution
  }

  def fromString(string: String): TacredSolution = {
    val split = string.split("\t")

    val numberOfSentences    = split(0).toInt
    val maxHighlightedLength = split(1).toInt
    val minHighlightedLength = split(2).toInt
    val solved               = split(3).toInt
    val clusterPath          = split(4)
    val p                    = if (split(5) == "") None else Some(split(5))
    val numberOfSteps        = split(6).toInt
    val trials               = split(7).toInt
    val wrongMatches         = if (split(8) == "") None else Some(split(8).toInt)
    val direction            = PatternDirection.fromIntValue(split(9).toInt)
    val relation             = split(10)
    val scorerVersion        = split(11)
    val solutionStatus       = SolutionStatus.fromIntValue(split(12).toInt)
    TacredSolution(
      numberOfSentences = numberOfSentences,
      maxHighlightedLength = maxHighlightedLength,
      minHighlightedLength = minHighlightedLength,
      clusterPath = clusterPath,
      numberOfSteps = numberOfSteps,
      solution = p,
      wrongMatches = wrongMatches,
      trials = trials,
      direction = direction,
      relation = relation,
      scorerVersion = scorerVersion,
      solutionStatus = solutionStatus,
    )
  }

  def fromPreviousVersion(string: String, scorerDefaultValue: String, solutionStatusDefaultValue: Int): TacredSolution = {
    val split = string.split("\t")

    val numberOfSentences    = split(0).toInt
    val maxHighlightedLength = split(1).toInt
    val minHighlightedLength = split(2).toInt
    val solved               = split(3).toInt
    val clusterPath          = split(4)
    val p                    = if (split(5) == "") None else Some(split(5))
    val numberOfSteps        = split(6).toInt
    val trials               = split(7).toInt
    val wrongMatches         = if (split(8) == "") None else Some(split(8).toInt)
    val direction            = PatternDirection.fromIntValue(split(9).toInt)
    val relation             = split(10)

    TacredSolution(
      numberOfSentences = numberOfSentences,
      maxHighlightedLength = maxHighlightedLength,
      minHighlightedLength = minHighlightedLength,
      clusterPath = clusterPath,
      numberOfSteps = numberOfSteps,
      solution = p,
      wrongMatches = wrongMatches,
      trials = trials,
      direction = direction,
      relation = relation,
      scorerVersion = scorerDefaultValue,
      solutionStatus = SolutionStatus.fromIntValue(solutionStatusDefaultValue),
    )
  }

}
sealed trait SolutionStatus {
  /**
    * 
    * The SolutionStatus construction is used only on the scala side
    * When it is saved to the file, we saved it as an integer for easier 
    * manipulation on python, for example
    *
    * @return an integer corresponding to the code for the current directionality
    */
  def intValue: Int
}
object SolutionStatus {
  def fromIntValue(intValue: Int): SolutionStatus = intValue match {
    case 0 => SolutionFailed
    case 1 => SolutionSucceeded
    case _ => throw new IllegalArgumentException(f"We got $intValue, but that value is not associated with any SolutionStatus.")
  }
}

/**
  * A solution can fail when something throws an exception during the searcher find call.
  * We differentiate between TacredSolution objects that finished successfully (even if no pattern was found) and TacredSolution objects
  * that finished with an exception (as it may be possible to find a solution)
  */
final case object SolutionFailed extends SolutionStatus {
  override val intValue = 0
}
/**
  * A solution is successful when the Try(searcher.find<..>) is successful. It does not matter if the searcher did not find a pattern
  */
final case object SolutionSucceeded extends SolutionStatus {
  override val intValue = 1
}


