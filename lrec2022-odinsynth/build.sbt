import com.typesafe.sbt.packager.docker.DockerChmodType

resolvers += "Sonatype OSS Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots"
resolvers += ("Artifactory" at "http://artifactory.cs.arizona.edu:8081/artifactory/sbt-release").withAllowInsecureProtocol(true)

name := "odinsynth"

ThisBuild / organization := "org.clulab"
ThisBuild / scalaVersion := "2.12.12"
ThisBuild / homepage := Some(url("https://github.com/clulab/odinsynth"))

lazy val sharedDeps = {
  libraryDependencies ++= {
    val odinsonVersion  = "0.3.2-SNAPSHOT"
    Seq(
      "org.scalatest" %% "scalatest"          % "3.0.5" % Test,
      "ai.lum"        %% "common"             % "0.1.1",
      "ai.lum"        %% "odinson-core"       % odinsonVersion,
      "ai.lum"        %% "odinson-extra"      % odinsonVersion,
      "com.lihaoyi"   %% "fastparse"          % "2.1.0",
      "com.lihaoyi"   %% "pprint"             % "0.5.6",
      "com.lihaoyi"   %% "ujson"              % "0.7.1",
      "com.lihaoyi"   %% "upickle"            % "0.7.1",
      "com.lihaoyi"   %% "requests"           % "0.6.5",
      "org.clulab"    %% "processors-main"    % "8.2.3",
      "org.clulab"    %% "processors-corenlp" % "8.2.3",
      "org.scala-lang.modules" %% "scala-parser-combinators" % "1.1.2",
    )
  }
}

lazy val assemblySettings = Seq(
  // Trick to use a newer version of json4s with spark (see https://stackoverflow.com/a/49661115/1318989)
  assembly / assemblyShadeRules := Seq(
    ShadeRule.rename("org.json4s.**" -> "shaded_json4s.@1").inAll
  ),
  assembly / assemblyMergeStrategy := {
    case refOverrides if refOverrides.endsWith("reference-overrides.conf") => MergeStrategy.first
    case logback if logback.endsWith("logback.xml") => MergeStrategy.first
    case netty if netty.endsWith("io.netty.versions.properties") => MergeStrategy.first
    case "messages" => MergeStrategy.concat
    case PathList("META-INF", "terracotta", "public-api-types") => MergeStrategy.concat
    case PathList("play", "api", "libs", "ws", xs @ _*) => MergeStrategy.first
    case PathList("org", "apache", "lucene", "analysis", xs @ _ *) => MergeStrategy.first
    case x =>
      val oldStrategy = (assembly / assemblyMergeStrategy).value
      oldStrategy(x)
  }
)

lazy val core = (project in file("."))
  .settings(sharedDeps)

lazy val webapp = project
  .enablePlugins(PlayScala)
  .enablePlugins(DockerPlugin)
  .aggregate(core)
  .dependsOn(core)
  .settings(
    assembly / test := {},
    assembly / mainClass := Some("play.core.server.ProdServerStart"),
    assembly / fullClasspath += Attributed.blank(PlayKeys.playPackageAssets.value),
    PlayKeys.devSettings ++= Seq(
      "play.server.akka.requestTimeout" -> "infinite",
      //"play.server.akka.terminationTimeout" -> "10 seconds",
      "play.server.http.idleTimeout" -> "2400 seconds",
    )
  )
  .settings(
    // see https://www.scala-sbt.org/sbt-native-packager/formats/docker.html
    dockerUsername := Some("odinson"),
    dockerAliases ++= Seq(
      // see https://github.com/sbt/sbt-native-packager/blob/master/src/main/scala/com/typesafe/sbt/packager/docker/DockerAlias.scala
      dockerAlias.value.withTag(Option("latest")),
    ),
    Docker / daemonUser  := "odinsynth",
    Docker / packageName := "odinsynth-backend",
    dockerBaseImage := "eclipse-temurin:11-jre-focal", // arm46 and amd64 compat
    Docker / maintainer := "Gus Hahn-Powell <gus@parsertongue.org>",
    Docker / dockerExposedPorts := Seq(9000),
    Universal / javaOptions ++= Seq(
      "-J-Xmx2G",
      // avoid writing a PID file
      "-Dplay.server.pidfile.path=/dev/null",
      "-Dplay.server.pidfile.path=/dev/null",
      "-Dplay.secret.key=odinsynth-is-not-production-ready",
      // NOTE: bind mount odison dir to /data/odinson
      "-Dodinson.dataDir=/data/odinson",
      // communicate with the rule gen. engine (PyTorch)
      "-Dodinsynth.scorer.scorerType=DynamicWeightScorer",
      // NOTE: assumes engine is running on or accessible from host's localhost 8000
      // see https://docs.docker.com/desktop/mac/networking/#use-cases-and-workarounds
      "-Dodinsynth.scorer.endpoint=http://host.docker.internal:8000",
      //"-Dplay.server.akka.requestTimeout=20s"
      //"-Dlogger.resource=logback.xml"
    )
  )

addCommandAlias("dockerfile", ";webapp/docker:stage")
addCommandAlias("dockerize", ";webapp/docker:publishLocal")
