package org.clulab.odinsynth

import ai.lum.odinson.{Document}

/**
  * Orchestrates the communication with an external API
  *
  * @param endpoint - address of the endpoint
  */
class ApiCaller(endpoint: String) {

  val scoreEndpoint   = f"$endpoint/score"
  val versionEndpoint = f"$endpoint/version"

  // Data holder class
  case class ApiData(@upickle.implicits.key("sentences") sentencesWithHighlight: Seq[Seq[String]], specs: Seq[SpecWrapper], patterns: Seq[String], @upickle.implicits.key("current_pattern") currentPattern: String)
  case class SpecWrapper(docId: String, sentId: Int, start: Int, end: Int)
  object SpecWrapper {
    def apply(s: Spec) = new SpecWrapper(s.docId, s.sentId, s.start, s.end)
  }
  implicit val specWrapperRW = upickle.default.macroRW[SpecWrapper]
  implicit val apiDataRW = upickle.default.macroRW[ApiData]

  def getScores(
      sentencesWithHighlight: Seq[Seq[String]],
      specs: Set[Spec],
      patterns: Seq[String],
      currentPattern: String
  ): Seq[Float] = {
    // case class for specs is not working with upickle
    val specsStr =
      specs.map(it => SpecWrapper(it)).toSeq
    //.mkString("[", ",", "]")
    // instantiate case class
    // to make json creation easier

    // Filter out negative examples to minimize the amount of data that has to be transferred
    val specSentenceId = specs.map(_.sentId).toSeq.sorted
    
    val apiData = ApiData(sentencesWithHighlight, specsStr, patterns, currentPattern)
    // convert the call data to json
    val jsonApiData = upickle.default.write(apiData)
    // make the api call
    // val time = System.nanoTime()
    // print(f"Send ${jsonApiData.length()} (${patterns.size} - ${sentencesWithHighlight.length})")
    // println(jsonApiData)
    val returnedData = requests.post(scoreEndpoint, data = jsonApiData, readTimeout = 50000 * sentencesWithHighlight.length, connectTimeout = 50000 * sentencesWithHighlight.length, headers = Seq(("Content-Type", "application/json")))
    // val elapsed = ((System.nanoTime() - time) / 1e9d).toFloat
    // convert the return to json
    val json = ujson.read(returnedData.text)
    // println(f". Receive ${json.arr.length} ($elapsed)")
    //println(s"returned ${returnedData.text}")
    // convert the output to num
    json.arr.map(v => v.num.toFloat)
  }

  def getVersion(): String = {
    val returnedData = requests.get(versionEndpoint, readTimeout=50000, connectTimeout=50000)

    val json = ujson.read(returnedData.text)

    // println(json.str)
    return json.str

  }

}
