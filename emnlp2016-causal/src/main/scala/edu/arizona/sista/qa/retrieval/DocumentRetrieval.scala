package edu.arizona.sista.qa.retrieval

import edu.arizona.sista.processors.Document

/**
 * 
 * User: mihais
 * Date: 3/15/13
 */
trait DocumentRetrieval {
  def retrieve:List[DocumentCandidate]
}
