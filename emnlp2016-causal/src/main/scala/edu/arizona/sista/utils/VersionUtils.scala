package edu.arizona.sista.utils

import scala.util.Try
import scala.sys.process._

/**
 * Created by dfried on 7/19/14.
 */
object VersionUtils {
  def gitRevision: Option[String] = Try({
    "git rev-parse HEAD".!!
  }).toOption
}
