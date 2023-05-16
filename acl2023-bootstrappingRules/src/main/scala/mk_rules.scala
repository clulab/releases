//import ai.lum.regextools.RegexBuilder
//import ai.lum.regextools.OdinPatternBuilder
//import scala.util.parsing.json._
//
//object mk_rules extends App {
//  val input_file = io.Source.fromURL(getClass.getResource("clusters.json"))
//  val jsonString = input_file.mkString
//  input_file.close()
//  val list:Map[String, Any] = JSON.parseFull(jsonString).get.asInstanceOf[Map[String, Any]]
//  for ((k, d)<-list){
//    println(k)
//    val cluster = d.asInstanceOf[Map[String, List[Any]]]
//    for ((c, ns)<-cluster){
//      val trigger_buildr = new RegexBuilder
//      val subj_builder = new OdinPatternBuilder
//      val obj_builder = new OdinPatternBuilder
//      for (n<-ns){
//        val node = n.asInstanceOf[Map[String, String]]
//        val trigger = node.get("trigger").get
//        val subj= node.get("subj").get
//        val obj = node.get("obj").get
//
//        trigger_buildr.add(trigger)
//        subj_builder.add(subj)
//        obj_builder.add(obj)
//      }
//      println("trigger = "+trigger_buildr.mkPattern)
//      println("subject: ${subject_type} = "+subj_builder.mkPattern)
//      println("object: ${object_type} = "+obj_builder.mkPattern)
//      println()
//      trigger_buildr.clear()
//      subj_builder.clear()
//      obj_builder.clear()
//    }
//  }
//
//}
