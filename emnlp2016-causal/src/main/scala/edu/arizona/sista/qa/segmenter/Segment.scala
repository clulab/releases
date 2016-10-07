package edu.arizona.sista.qa.segmenter

/**
 * Storage class for one text segment
 * User: peter
 * Date: 4/26/13
 */

// Storage class
class Segment (val label:String,
               val doc:edu.arizona.sista.processors.Document,
               val startOffset:Tuple2[Int, Int],      // Tuple2 (sentenceOffset, tokenOffset in sentence)
               val endOffset:Tuple2[Int, Int])  {       // Tuple2 (sentenceOffset, tokenOffset in sentence)


  override def toString:String = {
    var os = new StringBuilder
    os.append ("\nSegment:[label=" + label + " startOffset=(" + startOffset._1 + "," + startOffset._2 + ") endOffset=(" + endOffset._1 + "," + endOffset._2 + ")]")
    os.toString()
  }

}