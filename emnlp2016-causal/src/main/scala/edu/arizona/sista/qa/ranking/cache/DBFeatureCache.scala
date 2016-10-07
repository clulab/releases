package edu.arizona.sista.qa.ranking.cache

import java.sql._

import edu.arizona.sista.struct.Counter

/**
 * So the DB feature cache is basically pointless - the expected advantage over JSON backing store was that we could
 * have simultaneous writes from separate processes, but as it turns out SQLite does not allow this. Could change to
 * a different database backend but at this point marginal cost > marginal benefit
 *
 * Created by dfried on 8/7/14.
 */
class DBFeatureCache(val filename: String, val tablename: String = "features",
                     val busyTimeoutSeconds: Int = 180) extends FeatureCache {
  import edu.arizona.sista.qa.ranking.cache.FeatureCache.{Cache, QAKey, QAValue}

  /*
  commented this out because of some weird version stuff, can't find the proper version where this is defined
  val config = new SQLiteConfig
  config.setBusyTimeout((busyTimeoutSeconds * 1000).toString)
  */

  def getConnection: Connection = {
    // val connection = DriverManager.getConnection(s"jdbc:sqlite:$filename?dontTrackOpenResources=true")
    val connection = DriverManager.getConnection(s"jdbc:sqlite:$filename")
    // writes are faster if we batch them
    connection.setAutoCommit(false)
    connection
  }

  // cache the columns that we know exist
  private val knownColumns = new collection.mutable.HashSet[String]

  val cache = {
    val connection = getConnection
    if (getColumnNames(connection).isEmpty) {  // create the db
      withStatement(connection)(_.executeUpdate(s"create table $tablename (question string, answer string, kernel string, primary key(question, answer))"))
    }
    val ret = readCache(connection)
    connection.commit
    connection.close
    ret
  }

  def filterFeatures(colsToKeep: Set[String])(newFilename: String, newTablename: String = tablename,
                                             newBusyTimeout: Int = busyTimeoutSeconds): DBFeatureCache = {
    val newCache = new DBFeatureCache(newFilename, newTablename, newBusyTimeout)
    for ((key, (counter, kernel)) <- cache) {
      newCache.cache(key) = (counter.filter(pair => colsToKeep.contains(pair._1)), kernel)
    }
    newCache
  }

  def readFeatures(rs: ResultSet): (QAKey, QAValue) = {
    val rmd = rs.getMetaData
    val counter = new Counter[String]
    // question, answer, kernel should be first three columns
    for (idx <- 4 to rmd.getColumnCount; colName = rmd.getColumnName(idx); if (rs.getString(colName) != null))
      counter.setCount(colName, rs.getDouble(colName))
    val kernel = rs.getString("kernel")
    val question = rs.getString("question")
    val answer = rs.getString("answer")
    ((question, answer), (counter, kernel))
  }

  def readCache(connection: Connection) : Cache = {
    val statement = connection.createStatement
    val rs = statement.executeQuery(s"select * from $tablename")
    val cache = FeatureCache.mkCache
    for ((key, value) <-  resultSetIterator(rs, readFeatures))
      cache(key) = value
    rs.close
    statement.close
    cache
  }

  def withStatement(connection: Connection)(block: Statement => Unit): Unit = {
    val statement = connection.createStatement()
    block(statement)
    statement.close
  }

  def resultSetIterator[A](resultSet: ResultSet, accessor: ResultSet => A) = new Iterator[A] {
    private var hasNextCached = resultSet.next
    def hasNext = hasNextCached
    def next(): A = {
      val r = accessor(resultSet)
      hasNextCached = resultSet.next()
      if (! hasNextCached) resultSet.close()
      r
    }
  }

  def getColumnNames(connection: Connection): Set[String] = {
    resultSetIterator(connection.getMetaData.getColumns(null, null, tablename, null), _.getString("COLUMN_NAME")).toSet
  }

  // add the column to the db if it doesn't already exist
  def ensureColumn(connection: Connection)(columnName: String): Unit = {
    knownColumns.synchronized {
      if (! knownColumns.contains(columnName)) {
        knownColumns ++= getColumnNames(connection)
        if (! knownColumns.contains(columnName)) {
          withStatement(connection)(_.executeUpdate(s"alter table $tablename add column $columnName double"))
        }
      }
    }
  }

  val sqlCreateRow = s"insert into $tablename (question, answer) values (?, ?)"
  def createRow(connection: Connection)(qaPair: QAKey): Unit = {
    val (question, answer) = qaPair
    val psUpdate = connection.prepareStatement(sqlCreateRow)
    psUpdate.setString(1, question)
    psUpdate.setString(2, answer)
    psUpdate.executeUpdate
  }

  val sqlQueryPair =s"select * from $tablename where question = ? and answer = ?"
  def queryPair(connection: Connection)(qaPair: QAKey): Option[ResultSet] = {
    val (question, answer) = qaPair
    val psQuery = connection.prepareStatement(sqlQueryPair)
    psQuery.setString(1, question)
    psQuery.setString(2, answer)
    val rs = psQuery.executeQuery
    if (rs.next)
      Some(rs)
    else
      None
  }

  def ensureRow(connection: Connection)(qaPair: QAKey): Unit = {
    if (! dbContains(connection)(qaPair))
      createRow(connection)(qaPair)
  }

  def setFeatures(connection: Connection)(qaPair: QAKey, qaValue: QAValue): Unit = {
    val (question, answer) = qaPair
    val (counter, kernel) = qaValue
    val (features, values) = counter.toSeq.unzip
    val colString = if (features.nonEmpty)
      "kernel = ?, " + features.map(_ + " = ?").mkString(", ")
    else
      "kernel = ?"
    val updateString = s"update $tablename set $colString where question = ? and answer = ?"
    val psUpdate = connection.prepareStatement(updateString)
    val n_features = values.size
    psUpdate.setString(1, kernel)
    for ((v, i) <- values.zipWithIndex)
      psUpdate.setDouble(i + 2, v)
    psUpdate.setString(n_features + 2, question)
    psUpdate.setString(n_features + 3, answer)
    psUpdate.executeUpdate()
    psUpdate.close
  }

  def dbContains(connection: Connection)(key: QAKey): Boolean = this.synchronized {
    val maybeRs = queryPair(connection)(key)
    maybeRs.foreach(_.close())
    maybeRs.nonEmpty
  }

  // dump everything in the cache to the db file
  def writeCache: Unit = this.synchronized {
    val connection = getConnection
      for (column <- cache.values.map(_._1.keySet).reduce(_ ++ _))
        ensureColumn(connection)(column)
      for ((pair, value) <- cache) {
        ensureRow(connection)(pair)
        setFeatures(connection)(pair, value)
      }
      connection.commit()
      connection.close
  }
}

object DBFeatureCache {
  // this is necessary to load the JDBC interface
  Class.forName("org.sqlite.JDBC")
}
