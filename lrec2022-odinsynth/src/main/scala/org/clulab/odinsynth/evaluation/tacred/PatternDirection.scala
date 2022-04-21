package org.clulab.odinsynth.evaluation.tacred

/**
  * Specifies the direction that the pattern is to be applied
  */
sealed trait PatternDirection {
  /**
    * 
    * The PatternDirection construction is used only on the scala side
    * When it is saved to the file, we saved it as an integer for easier 
    * manipulation on python, for example
    *
    * @return an integer corresponding to the code for the current directionality
    */
  def intValue: Int
}
object PatternDirection {
  def fromIntValue(intValue: Int): PatternDirection = intValue match {
    case 0 => SubjObjDirection
    case 1 => ObjSubjDirection
    case _ => throw new IllegalArgumentException(f"We got $intValue, but that value is not associated with any PatternDirection.")
  }
}
final case object SubjObjDirection extends PatternDirection {
  override val intValue = 0
}
final case object ObjSubjDirection extends PatternDirection {
  override val intValue = 1
}