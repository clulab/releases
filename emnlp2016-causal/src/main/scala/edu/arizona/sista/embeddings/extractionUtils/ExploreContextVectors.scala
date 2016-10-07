package extractionUtils

import edu.arizona.sista.embeddings.word2vec.Word2Vec

/**
  * Created by bsharp on 3/19/16.
  */
object ExploreContextVectors extends App {

  val causeVectors = "/home/bsharp/yoavgo-word2vecf-90e299816bcd/simpleWiki_mar19/simpleWiki_dim200vecs"
  val effectVectors = "/home/bsharp/yoavgo-word2vecf-90e299816bcd/simpleWiki_mar19/simpleWiki_dim200context-vecs"
  val causeW2V = new Word2Vec(causeVectors)
  val effectW2V = new Word2Vec(effectVectors)

  val mode = scala.io.StdIn.readLine("Mode (1 for most similar CAUSES, 2 for most similar EFFECTS: ").toInt match {
    case 1 => {
      var input = ""
      while (input != "EXIT") {
        input = scala.io.StdIn.readLine("Enter CAUSE (or EXIT to exit): ")
        println (s"Top 20 most similar CAUSES to $input:")
        val top = causeW2V.mostSimilarWords(Set(Word2Vec.sanitizeWord(input)), 20)
        for(t <- top) {
          println(t._1 + " " + t._2)
        }
      }
    }
    case 2 => {
      var input = ""
      while (input != "EXIT") {
        input = scala.io.StdIn.readLine("Enter CAUSE (or EXIT to exit): ")
        println (s"Top 20 most similar EFFECTS to $input:")
        val top = effectW2V.mostSimilarWords(Set(Word2Vec.sanitizeWord(input)), 20)
        for(t <- top) {
          println(t._1 + " " + t._2)
        }
      }
    }
    case _ => println("Error: invalid mode")
  }

}
