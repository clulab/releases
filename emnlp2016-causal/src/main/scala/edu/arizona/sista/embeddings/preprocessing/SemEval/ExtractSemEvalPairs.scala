package preprocessing.SemEval

import java.io.PrintWriter

import scala.collection.mutable.ArrayBuffer

/**
  * Created by bsharp on 3/1/16.
  */
object ExtractSemEvalPairs {

  def loadEntityPairs(filename:String, taskRelation:String): Array[EntityPair] = {
    // Sample text:
//    32	"He had chest pains and <e1>headaches</e1> from <e2>mold</e2> in the bedrooms."
//    Cause-Effect(e2,e1)
//    Comment:
//
//    33	"The silver-haired author was not just laying India's politician saint to rest but healing a generations-old rift in the family of the <e1>country</e1>'s founding <e2>father</e2>."
//    Product-Producer(e1,e2)
//    Comment: in an abstract way the country is built/founded/produced by the father
    val out = new ArrayBuffer[EntityPair]

    val lines = scala.io.Source.fromFile(filename, "UTF-8").getLines()

    while (lines.hasNext) {
      val textLine = lines.next()
      val relationLine = lines.next()
      val commentLine = lines.next()
      lines.next()  // blank line

      // Extract the text and the index
      val tlsp = textLine.split("\t")
      assert (tlsp.length == 2)
      val index = tlsp(0).toInt
      val text = tlsp(1)

      // Extract the entities from the text
      val (e1, e2) = extractEntityTexts(text)

      // Extract the Relation label and the ordered labeled entities
      val relationOption = extractRelation(relationLine, e1, e2)
      // If there is a relation:  (i.e. if Option isn't None)
      if (relationOption.getOrElse(-1) != -1) {
        // Retrieve the elements
        val (relation, src, dst) = relationOption.get
        // Create a gold/not-gold label for the entity pair in regards to the current task
        val taskLabel = if (relation == taskRelation) 1 else 0
        // Make an EntityPair
        val ep = new EntityPair(src, dst, relation, index, taskLabel)
        // Display and Store
        println ("Extracted -- \n" + ep.toString)
        out.append(ep)
      }
    }

    out.toArray
  }

  // Extracts the text of the two entities
  def extractEntityTexts(s:String): (String, String) = {

    val e1Pattern = "<e1>(.*)</e1>".r.unanchored
    val e1Pattern(e1) = s

    val e2Pattern = "<e2>(.*)</e2>".r.unanchored
    val e2Pattern(e2) = s

    (e1, e2)
  }

  // Returns the Relation, the labels for the members of the relation, and the ordering
  def extractRelation(rText:String, e1: String, e2: String): Option[(String, LabeledEntity, LabeledEntity)] = {
    // Case 1: If there is no relation...
    if (rText == "Other") return None
    // Case 2: There is a relation, extract the relevant parts and organize the previously extracted entities accordingly
    // Example: "Cause-Effect(e2,e1)"
    val pattern = "(.*)-(.*)\\(e(.),e(.)\\)".r
    val pattern(srcLabel, dstLabel, src, _) = rText

    // Determine the source and destination text (i.e. the ordering of the relation, which is the first/src and second/dst)
    val srcEntity = if (src == "1") e1 else e2
    val dstEntity = if (src == "1") e2 else e1

    // Return the ordered relation
    val relation = s"$srcLabel-$dstLabel"
    val labeledSrc = new LabeledEntity(srcEntity, srcLabel)
    val labeledDst = new LabeledEntity(dstEntity, dstLabel)

    Some(relation, labeledSrc, labeledDst)
  }

  def main(args:Array[String]): Unit = {

    val semEvalFile = "/home/bsharp/SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT"

    //val taskRelation = "Cause-Effect"
    val taskRelation = "Component-Whole"

    val entityPairs = loadEntityPairs(semEvalFile, taskRelation)
    val pwTarget = new PrintWriter(s"/home/bsharp/causal/SEMEVAL_${taskRelation}_pairs.txt")
    val pwOther = new PrintWriter(s"/home/bsharp/causal/SEMEVAL_NON_${taskRelation}_pairs.txt")
    for (ep <- entityPairs){
      if (ep.relation == taskRelation) pwTarget.println(ep.toStringMinimal)
      else pwOther.println(ep.toStringMinimal)
    }

    pwTarget.close()
    pwOther.close()

  }

}

case class EntityPair(src:LabeledEntity, dst:LabeledEntity, relation:String, index:Int, taskLabel:Int) {
  override def toString = {
    s"Entity Pair $index\n " +
      s"\tRelation: $relation \n" +
      s"\tSRC: ${src.text} (${src.label}) ==> DST: ${dst.text} (${dst.label})\n" +
      s"\ttaskLabel: $taskLabel"
  }

  def toStringMinimal = {
    s"${src.text} (${src.label}) ==> ${dst.text} (${dst.label})"
  }

}

case class LabeledEntity(text:String, label:String)
