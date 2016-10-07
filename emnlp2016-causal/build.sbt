name := "emnlp2016-causal"

version := "1-0-SNAPSHOT"

scalaVersion := "2.11.6"


libraryDependencies ++= Seq(
  "org.scalatest" %% "scalatest" % "2.2.1" % "test",
  "junit" % "junit" % "4.10" % "test",
  "com.novocode" % "junit-interface" % "0.11" % "test",
  "org.clulab" %% "processors" % "5.8.2-SNAPSHOT",
  "org.clulab" %% "processors" % "5.8.2-SNAPSHOT" classifier "models",
  "org.json" % "json" % "20090211",
  "org.json4s" %% "json4s-native" % "3.2.11",
  "com.googlecode.json-simple" % "json-simple" % "1.1",
  "org.apache.lucene" % "lucene-core" % "3.0.3",
  "com.fasterxml.jackson.module" %% "jackson-module-scala" % "2.5.1",
  "edu.jhu.agiga" % "agiga" % "1.4"

)

