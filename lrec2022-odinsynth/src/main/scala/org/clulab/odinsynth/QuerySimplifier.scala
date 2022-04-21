package org.clulab.odinsynth

// Simplifies a query by grouping consecutive constraints of the same underlying type
object QuerySimplifier {
  /**
    * For reduction, we compare the "underlying" type.
    * That is, we want to group consecutive things like:
    *   - [tag=NNP]
    *   - [tag=NNP]?
    *   - [tag=NNP]+
    *   - [tag=NNP]*
    *   - [tag=NNP]{<digit>}
    * As such, we consider them equal
    */
  implicit val underlyingQueryEquiv = new Equiv[Query] {
    def equiv(x: Query, y: Query): Boolean = {
      (x, y) match {
        case (RepeatQuery(q1, min1, max1), RepeatQuery(q2, min2, max2)) => q1 == q2
        case (RepeatQuery(q1, min1, max1), _) => q1 == y
        case (_, RepeatQuery(q2, min2, max2)) => x == q2
        case _ => x == y
      }
    }
  }

  /**
    * 
    *
    * @param query                       -> the query to be simplified
    * @param overApproximateMaxThreshold -> the maximum number of same type; above this we simply add a * or a +
    * @return                            -> the simplified query
    */
  def simplifyQuery(query: Query, overApproximateMaxThreshold: Option[Int] = Some(2)): Query = {

    /**
      * Reduce a sequence of queries to a single query
      * NOTE: We use the fact that this is called only with a sequence of the
      * "same" type ("same" because we consider [word=this], [word=this]?, [word=this]* etc to be the same type)
      *
      * @param lq
      * @return
      */
    def queryReducer(lq: Seq[Query]): Query = {
      // Get the underlying query by unwrapping the repeat query (if exists)
      val underlyingQuery = lq.head match {
        case RepeatQuery(query, _, _) => query
        case it@_                     => it
      }
      if (lq.size == 1){
        lq.head
      } else {
        // map to min and sum
        val min = lq.map { _ match {
            case RepeatQuery(_, min, _) => min
            case _                      => 1
          } 
        }.sum
        // map to max and sum; handle None
        val max = lq.map { _ match {
            case RepeatQuery(_, _, max) => max
            case _                      => Some(1)
          } 
        }.let { it =>
          if (it.exists(_.isEmpty) || it.flatMap(it => it).sum > overApproximateMaxThreshold.getOrElse(Int.MaxValue)) {
            None
          } else {
            Some(it.flatMap(it => it).sum)
          }
        }
        RepeatQuery(underlyingQuery, min, max) 
      }
    }

    query match {
      case ConcatQuery(queries)        => {
        val consecutives = groupConsecutives(queries.toList.map(it => simplifyQuery(it, overApproximateMaxThreshold)))
        val result = consecutives.map(queryReducer)
        if(result.size == 1) {
          result.head
        } else {
          ConcatQuery(result.toVector)
        }
      }
      case OrQuery(queries)            => {
        val consecutives = groupConsecutives(queries.toList.map(it => simplifyQuery(it, overApproximateMaxThreshold)))
        val result = consecutives.map(queryReducer)
        if(result.size == 1) {
          result.head
        } else {
          OrQuery(result.toVector)
        }
      }
      case ncq@NamedCaptureQuery(q, _) => ncq.copy(query=simplifyQuery(q, overApproximateMaxThreshold)) 
      case _ => query
    }
  }


  def groupConsecutives[CT, T](coll: CT)(implicit collToList: CT => List[T], equiv: Equiv[T]): List[List[T]] = {
    coll.toList match {
      case head :: tail =>
        val (t1, t2) = tail.span(it => equiv.equiv(it, head))
        (head :: t1) :: groupConsecutives(t2)
      case _ => Nil  
    }
  }


}
