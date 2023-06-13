

name := "natlog"
organization := "org.natlog"

version := "0.1.0-SNAPSHOT"


lazy val root = (project in file("."))
  .settings(
    name := "natlog"
  )
resolvers ++= Seq(
  "jitpack" at "https://jitpack.io", // This provides access to regextools straight from github.
  ("Artifactory" at "http://artifactory.cs.arizona.edu:8081/artifactory/sbt-release").withAllowInsecureProtocol(true)
)

libraryDependencies ++= {
  val procVer = "8.5.2"

  Seq(
    "org.clulab" %% "processors-main" % procVer,
    "org.clulab" %% "processors-corenlp" % procVer,
    // other dependencies here
    "org.scalanlp" %% "breeze"                                                                                                                                                                                  % "1.1",
    // native libraries are not included by default. add this if you want them (as of 0.7)
    // native libraries greatly improve performance, but increase jar sizes.
    // It also packages various blas implementations, which have licenses that may or may not
    // be compatible with the Apache License. No GPL code, as best I know.
    "org.scalanlp" %% "breeze-natives" % "1.1",
    // the visualization library is distributed separately as well.
    // It depends on LGPL code.
    "org.scalanlp" %% "breeze-viz" % "1.1",
    "com.github.tototoshi" %% "scala-csv" % "1.3.10",
    "org.scalatest"              %% "scalatest"          % "3.0.9" % Test,
    "org.scalatestplus.play" %% "scalatestplus-play" % "3.0.0" % Test
  )
}

