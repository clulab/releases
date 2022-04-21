package org.clulab.odinsynth

import edu.cmu.dynet.Initialize
import org.clulab.processors.Processor
import org.clulab.processors.fastnlp.FastNLPProcessor
import scala.util.Random
import scala.collection.mutable
import org.clulab.dynet.Utils.initializeDyNet


object SearcherUtils {

  // FIXME: Why must this be a FastNLPProcessor?
  def mkProcessor(): FastNLPProcessor = {
    initializeDyNet()
    new FastNLPProcessor
  }

}