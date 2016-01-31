package utils

import org.apache.commons.logging.LogFactory
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.hadoop.mapred._
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.SparkContext._

class DirectOutputCommitter extends OutputCommitter {
  private final val LOG = LogFactory.getLog("org.apache.spark.examples.DirectOutputCommitter")

  override def setupJob(jobContext: JobContext): Unit = {
    LOG.info("Nothing to do in setupJob")
  }

  override def needsTaskCommit(taskContext: TaskAttemptContext): Boolean = {
    LOG.info("Nothing to do in needsTaskCommit"); false
  }

  override def setupTask(taskContext: TaskAttemptContext): Unit = {
    LOG.info("Nothing to do in setupTask")
  }

  override def commitTask(taskContext: TaskAttemptContext): Unit = {
    LOG.info("Nothing to do in commitTask")
  }

  override def abortTask(taskContext: TaskAttemptContext): Unit = {
    LOG.info("Nothing to do in abortTask")
  }
}
