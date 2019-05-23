name := "clint"

version := "0.0.1-SNAPSHOT"

organization := "org.clulab"

scalaVersion := "2.12.4"

resolvers += "Typesafe Repo" at "http://repo.typesafe.com/typesafe/releases/"

scalacOptions ++= Seq(
  "-feature",
  "-unchecked",
  "-deprecation"
)

libraryDependencies ++= Seq(
  "org.scalatest" %% "scalatest" % "3.0.4" % "test",
  "org.clulab" %% "processors-main" % "7.1.0",
  "org.clulab" %% "processors-corenlp" % "7.1.0",
  "org.clulab" %% "processors-modelsmain" % "7.1.0",
  "org.clulab" %% "processors-modelscorenlp" % "7.1.0",
  "ai.lum" %% "common" % "0.0.8",
  "ai.lum" %% "nxmlreader" % "0.1.2",
  "com.typesafe.scala-logging" %% "scala-logging" % "3.5.0",
  "ch.qos.logback" % "logback-classic" % "1.1.7",
  "org.clulab" %% "processors-odin" % "7.1.0",
  //"com.mnemotix" %% "stringmetric-core" % "0.28.0-SNAPSHOT",
  "me.tongfei" % "progressbar" % "0.4.0",
  "io.spray" %%  "spray-json" % "1.3.3"
)
