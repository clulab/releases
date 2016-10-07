package edu.arizona.sista.qa.segmenter

import edu.arizona.sista.processors.{Sentence, Document}
import edu.arizona.sista.qa.ranking.ProcessedQuestionSegments
import collection.mutable.{ListBuffer, ArrayBuffer}
import edu.arizona.sista.struct.{Counter}
import edu.arizona.sista.qa.index.TermFilter
import java.util.Properties
import org.slf4j.LoggerFactory
import QuestionProcessor.logger
import edu.arizona.sista.struct.Tree


// NOTE: This could use some cleanup

/**
 * Basic question segmenter for discourse models, where the fragmentation point is centered around the question's head verb
 * User: peter
 * Date: 5/31/13
 */

class QuestionProcessor(props:Properties) {
  val termFilter = new TermFilter()
  val indexDir = props.getProperty("index")


  def mkProcessedQuestionOneArgument(question:Document):ProcessedQuestionSegments = {
    // Separate the question using the head verb as the segmentation point
    val segmentMatcher = new SegmentMatcherBOW(termFilter, indexDir)
    val questionType = question.sentences(0).words(0).toLowerCase  // TODO: Dumb method -- simply hopes the first word will be "HOW" or "WHY"

    var argSegs = List[Segment]()
    val argSegsLTN = new ArrayBuffer[Counter[String]]()
    val numSentences = question.sentences.size - 1

    // Step 3: Assemble question segments (here, just one big segment, containing all sentences in question)
    argSegs = List(new Segment("QSEG", question, (0, 0), (numSentences, question.sentences(numSentences).words.size)))

    // Step 4: Precompute LTN on question segments to greatly speed any BOW scoring later on
    argSegsLTN.append ( segmentMatcher.buildVectorFromSegmentLTN(indexDir, argSegs(0))  )

    // return new processed question
    new ProcessedQuestionSegments(questionType, argSegs, argSegsLTN.toList)

  }


  def mkProcessedQuestionSVO(question:Document):ProcessedQuestionSegments = {

    // Separate the question using the head verb as the segmentation point
    val segmentMatcher = new SegmentMatcherBOW(termFilter, indexDir)
    val questionType = question.sentences(0).words(0).toLowerCase  // TODO: Dumb method -- simply hopes the first word will be "HOW" or "WHY"

    var argSegs = new ArrayBuffer[Segment]()
    val argSegsLTN = new ArrayBuffer[Counter[String]]()

    // Step 1: Check that syntactic tree information is populated
    if (question.sentences(0).syntacticTree.isEmpty) {
      throw new RuntimeException ("mkProcessedQuestionSVO(): ERROR: Question passed does not have syntactic tree populated. ")
    }

    // Step 2: Use syntactic tree information to determine head verb
    var syntacticTree = question.sentences(0).syntacticTree.get
    val verbPosition = findMainVerbPosition(syntacticTree, question.sentences(0))
    if (verbPosition == -1) {
      // Return empty segments
      logger.debug ("mkProcessedQuestionSVO(): ERROR: Could not detect verb")
      return new ProcessedQuestionSegments("failed", argSegs.toList, argSegsLTN.toList)
    }

    logger.debug("The main verb is \"" + question.sentences(0).words(verbPosition) + "\" at position " + verbPosition)

    // Step 3: Use syntactic tree and head verb information to find preposition phrase after head verb
    val offsetsPP = findAllOffsets(Some(syntacticTree), "PP", ArrayBuffer[Int]())
    var ppPosition = -1
    for (i <- 0 until offsetsPP.size) {
      if ((offsetsPP(i) > verbPosition+1) && (ppPosition == -1)){     // +1 to skip over cases that are VB PP (eg. HOW DO I SCROLL TO THE TOP OF...)
        ppPosition = offsetsPP(i)
      }
    }

    // Step 4: Assemble question segments
    // 0   1  2+   3    .
    // HOW DO SUBJ VERB OBJ INDOBJ
    var qLength = question.sentences(0).words.size
    if (question.sentences(0).tags.get(qLength-1) == ".") qLength -= 1      // remove any punctuation at the end of the sentence

    // Check for subject
    if (verbPosition > 2) {
      val argSubjSeg = new Segment("S", question, (0, 2), (0, verbPosition))       // Start at position 2  (HOW,0) (DO*, 1), (X, 2..N)
      argSegs.append(argSubjSeg)
      argSegsLTN.append ( segmentMatcher.buildVectorFromSegmentLTN(indexDir, argSubjSeg) )
    }

    // Always a verb
    val argVerbSeg = new Segment("V", question, (0, verbPosition), (0, verbPosition+1))
    argSegs.append(argVerbSeg)
    argSegsLTN.append ( segmentMatcher.buildVectorFromSegmentLTN(indexDir, argVerbSeg) )

    // Check for object
    if ((qLength-1) > verbPosition) {
      // Check if we also have an indirect object
      if (ppPosition != -1) {
        val argObjSeg = new Segment("O", question, (0, verbPosition+1), (0, ppPosition))
        val argIndObjSeg = new Segment("I", question, (0, ppPosition), (0, qLength))
        argSegs.append(argObjSeg)
        argSegs.append(argIndObjSeg)
        argSegsLTN.append ( segmentMatcher.buildVectorFromSegmentLTN(indexDir, argObjSeg) )
        argSegsLTN.append ( segmentMatcher.buildVectorFromSegmentLTN(indexDir, argIndObjSeg) )
      } else {
        val argObjSeg = new Segment("O", question, (0, verbPosition+1), (0, qLength))
        argSegs.append(argObjSeg)
        argSegsLTN.append ( segmentMatcher.buildVectorFromSegmentLTN(indexDir, argObjSeg) )
      }
    }

    // debug
    logger.debug ("questionProcessor.mkProcessedQuestionSVO Segments:")
    for (i <- 0 until argSegs.size)  logger.debug(i + ": " + argSegs(i))

    // return new processed question
    new ProcessedQuestionSegments(questionType, argSegs.toList, argSegsLTN.toList)

  }


  def findFirstSubtreeExactMatch(rootOpt: Option[Tree], label:String):Option[Tree] = {
    findFirstSubtree(rootOpt, label, 0, 0)._1
  }

  def findFirstSubtreeStartsWith(rootOpt: Option[Tree], label:String):Option[Tree] = {
    findFirstSubtree(rootOpt, label, 0, 1)._1
  }

  def findFirstSubtree(rootOpt: Option[Tree], label:String, depth:Int = 0, mode:Int = 0):(Option[Tree], Int) = {
    // Mode = 0 : Match 'label' exactly
    // Mode = 1 : Match 'label' as start of node value
    val LARGENUM = 999999

    // Empty case
    if (rootOpt.isEmpty) return (None, LARGENUM)
    val root = rootOpt.get

    // Match current node case
    //println("root.value = " + root.value)
    if (mode == 0) {
      if (root.value == label) return (Some(root), depth)
    } else if (mode == 1) {
      if (root.value.startsWith(label)) return (Some(root), depth)
    }

    // Recursive Case
    if (!root.isLeaf) {
      var topSubTree: Option[Tree] = None
      var topDepth: Int = LARGENUM // Arbitrarily large depth
      // Traverse all subtrees
      if (!root.children.isEmpty) {
        val children = root.children.get
        for (child <- children) {
          val (subtree, stDepth) = findFirstSubtree(Some(child), label, depth + 1, mode)
          // Take the shallowest tree
          if (stDepth < topDepth) {
            topDepth = stDepth
            topSubTree = subtree
          }
        }
      }

      // Return tree
      return (topSubTree, topDepth)
    }

    return (None, LARGENUM)
  }

  //## TEST
  def findAllOffsets(rootOpt: Option[Tree], label:String, in:ArrayBuffer[Int]):ArrayBuffer[Int] = {
    var offsets = in

    // Empty case
    if (rootOpt.isEmpty) return ArrayBuffer[Int]()
    val root = rootOpt.get

    // Match current node case
    if (root.value == label) offsets.append(root.startOffset)

    // Recursive Case
    if (!root.isLeaf) {
      // Traverse all subtrees
      if (!root.children.isEmpty) {
        val children = root.children.get
        for (child <- children) {
          offsets = findAllOffsets(Some(child), label, offsets)
        }
      }
    }

    return offsets
  }


  def findMainVerbPosition(tree:Tree, sentence:Sentence):Int = {
    // stores the (position in sentence, depth in tree) for all verb head words
    val allVerbHeads = new ListBuffer[(Int, Int)]
    // stores the ids of verb head words already seen in the traversal
    val seenVerbs = new collection.mutable.HashSet[Int]
    // find all verbs that are head words in this sentence
    findVerbHeads(tree, sentence, 0, seenVerbs, allVerbHeads)

    //println("Tree:\n" + tree)
    //println("Verb heads:")
    //for(v <- allVerbHeads) {
    //  println("\tposition:" + v._1 + ", depth:" + v._2)
    //}

    // sort verb heads by their depth, i.e., how deep in the syntactic tree they are
    val sortedHeads = allVerbHeads.toList.sortBy(_._2)

    // pick the non-auxiliary verb with the smallest depth
    for(v <- sortedHeads) {
      // do NOT accept verbs in positions < 2; these are just auxiliary verbs
      if(v._1 > 1) {
        return v._1
      }
    }

    // no verb found; shouldn't happen
    -1
  }

  def findVerbHeads(
                     tree:Tree,
                     sentence:Sentence,
                     depth:Int,
                     seenVerbs:collection.mutable.HashSet[Int],
                     allVerbHeads:ListBuffer[(Int, Int)]) {

    val headPos = tree.headOffset
    assert(headPos >= 0)
    if(sentence.tags.get(headPos).startsWith("VB") && ! seenVerbs.contains(headPos)) {
      seenVerbs += headPos
      allVerbHeads += new Tuple2[Int, Int](headPos, depth)
    }
    if(! tree.isLeaf) {
      for(c <- tree.children.get) {
        findVerbHeads(c, sentence, depth + 1, seenVerbs, allVerbHeads)
      }
    }
  }


}

object QuestionProcessor {
  val logger = LoggerFactory.getLogger(classOf[QuestionProcessor])
}
