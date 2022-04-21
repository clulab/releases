package org.clulab.odinsynth.scorer

/**
 * 
 * Hold the weights for each type of AstNode (@see org.clulab.odinsynth.AstNode)
 */
final case class AstCost(
    holeMatcher: Float        = 2f,
    holeConstraint: Float     = 3f,
    holeQuery: Float          = 2f,
    stringMatcher: Float      = 1f,
    fieldConstraint: Float    = 0f,
    notConstraint: Float      = 10f,
    andConstraint: Float      = 4f,
    orConstraint: Float       = 4f,
    tokenQuery: Float         = 0f,
    concatQuery: Float        = 4f,
    orQuery: Float            = 4f,
    repeatQuery: Float        = 15f,
    matchAllMatcher: Float    = 0f,
    matchAllQuery: Float      = 0f,
    matchAllConstraint: Float = 0f,
)
object AstCost {
    // Get the original weights
    def getStandardWeights(): AstCost = AstCost(
        holeMatcher        = 2f,
        holeConstraint     = 3f,
        holeQuery          = 2f,
        stringMatcher      = 1f,
        fieldConstraint    = 0f,
        notConstraint      = 10f,
        andConstraint      = 4f,
        orConstraint       = 4f,
        tokenQuery         = 0f,
        concatQuery        = 4f,
        orQuery            = 4f,
        repeatQuery        = 15f,
        matchAllMatcher    = 0f,
        matchAllQuery      = 0f,
        matchAllConstraint = 0f,
    )
}
