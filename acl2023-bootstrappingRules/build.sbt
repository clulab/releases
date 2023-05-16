name := "TACRED_ODIN"

version := "1.0"

scalaVersion := "2.12.13"

resolvers += ("Artifactory" at "http://artifactory.cs.arizona.edu:8081/artifactory/sbt-release").withAllowInsecureProtocol(true)

libraryDependencies ++= {
  val procVer = "8.3.1"

  Seq(
    "org.clulab" %% "processors-main" % procVer,
    "org.clulab" %% "processors-corenlp" % procVer,
    "org.clulab" %% "processors-odin" % procVer
  )
}

//libraryDependencies += "ai.lum" %% "regextools" % "0.1.0-SNAPSHOT"
