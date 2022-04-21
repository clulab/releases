package org.clulab.odinsynth.evaluation.fstacred

import upickle.default.{ReadWriter => RW, macroRW, write, read}
import java.io.PrintWriter
import java.io.File
import scala.io.Source
import ai.lum.odinson.Document
import org.clulab.odinsynth.scorer.StaticWeightScorer
import org.clulab.odinsynth.{Searcher, Spec, Parser}
import scala.collection.mutable.ListBuffer

// import mutable map
import scala.collection.mutable.Map

case class Step(
    current_rule: String,
    next_correct_rule: String,
    next_incorrect_rules: Seq[String]
)

/** An application to generate next actions for rules extracted by the oracle
  */
object NewOracleConversionApp extends App {
  implicit val rw: RW[Step] = macroRW

  def getListOfFiles(dir: String): List[File] = {
    val d = new File(dir)
    if (d.exists && d.isDirectory) {
      d.listFiles.filter(_.isFile).toList
    } else {
      List[File]()
    }
  }

  //
  val scorer = StaticWeightScorer()
  val dataFolder = "/home/gcgbarbosa/repos/research/odinsynth/scripts/"

  // list all files inside specs folder
  val specsFiles = getListOfFiles(dataFolder + "specs/")

  specsFiles.map { specPath =>
    {
      // open the specs file
      val evalFile = Source.fromFile(specPath).getLines().toList
      val specsJson = evalFile(0)
      val jsonDoc = evalFile(1)
      val odinsonDoc = Document.fromJson(jsonDoc)

      //
      val fieldNames = Set("word", "tag", "lemma")

      // create the search object
      val searcher = new Searcher(
        doc = odinsonDoc,
        specs = Set(Spec.apply("test", 0, 0, 1)), //specs.toSet,
        fieldNames = fieldNames,
        maxSteps = None,
        writer = None,
        scorer = scorer,
        withReward = false
      )

      // make the name of the specs file
      // open the steps file
      val stepsPath =
        (specPath.toString().dropRight(9) + "steps.json")
          .replace("specs", "v2.0-steps")

      // open steps file
      val stepsFile = Source.fromFile(stepsPath).getLines().toList

      val correctSteps = Map.empty[String, Seq[String]]

      // TODO: this needs to return a map
      stepsFile.foreach { step =>
        {
          val objStep = read(step)
          // check if key exists
          if (correctSteps.contains(objStep.current_rule)) {
            // just add new key
            correctSteps +=
              (objStep.current_rule -> (
                correctSteps(objStep.current_rule) ++ Seq(
                  objStep.next_correct_rule
                )
              ))
          } else {
            correctSteps += (objStep.current_rule -> Seq(
              objStep.next_correct_rule
            ))
          }
          objStep.next_correct_rule
        }
      }
      correctSteps.foreach(println)

      val filledSteps: List[String] = stepsFile.map { step =>
        {
          // parse specs file
          val objStep = read(step)
          // add step to correctSteps

          val currentRule = Parser.parseBasicQuery(objStep.current_rule)

          // call next to get next rules
          val nextRules = currentRule.nextNodes(searcher.vocabularies)

          // remove the rule that is correct from the incorrect ones
          val nextIncorrectRules =
            nextRules.filter(r =>
              r.pattern != objStep.next_correct_rule
                && !correctSteps(objStep.current_rule).contains(r.pattern)
            )

          write(
            Step(
              objStep.current_rule,
              objStep.next_correct_rule,
              nextIncorrectRules.map(r => r.pattern)
            )
          )
        }
      }

      // generate the path of the file to be saved
      //
      val newStepsPath = stepsPath.replace("v2.0-steps", "v2.1-steps")

      // save rules to a file
      val outputFile = new PrintWriter(new File(newStepsPath))
      outputFile.print(filledSteps.mkString("\n") + "\n")
      outputFile.close()
    }
  }
}
