package extractionUtils

/**
 * Helper functions for working with tagged lemmas (of the form "lemma_TAG")
 * Created by peter on 2/3/16.
 */

object TaggedLemmaHelper {
  val contentTags = Array("NN", "VB", "JJ", "RB", "IN")

  // Make tagged lemma string format
  def mkTLemma(lemma:String, tag:String):String = {
    lemma.trim().toLowerCase() + "_" + mkGroupedTag(tag)
  }

  // Group tags -- all nouns to NN, all verbs to VB, etc.
  def mkGroupedTag(tag:String):String = {
    var groupedTag:String = tag
    if (groupedTag.length > 2) groupedTag = groupedTag.substring(0, 2)
    groupedTag.toUpperCase
  }

  // Check if a string is already likely a tagged lemma (e.g. lemma_TAG, cat_NN)
  // Note: Currently a simple test
  def isTLemmaFormat(text:String):Boolean = {
    if (!text.contains("_")) return false
    true
  }

  def hasContentTag(queryTag:String):Boolean = {
    for (contentTag <- contentTags) {
      if (queryTag.startsWith(contentTag)) {
        return true
      }
    }
    false
  }


}