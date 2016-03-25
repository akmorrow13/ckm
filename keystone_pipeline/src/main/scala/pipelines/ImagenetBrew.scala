package pipelines.imagenet

import java.net.URI

import scala.reflect.ClassTag
import scala.util.Random
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.HashMap

import org.apache.hadoop.fs.{Path, FileSystem}
import org.apache.hadoop.conf.Configuration

import breeze.linalg._
import breeze.numerics._

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import nodes._
import pipelines._
import utils.{ImageMetadata, Stats, Image, ImageUtils}

object ImageNetBrew extends Serializable {

  def main(args: Array[String]) {

    if (args.length < 3) {
      println("Usage: ImageNetBrew <trainDir> <testDir> <trainingLabelsFile>")
      System.exit(0)
    }

    val trainingDirName = args(0)
    val testingDirName = args(1)
    val trainingLabelsPath = args(2)

    val conf = new SparkConf()
      .setAppName("ImageNetBrew")
      .setJars(SparkContext.jarOfObject(this).toSeq)
      .set("spark.hadoop.validateOutputSpecs", "false") // overwrite hadoop files

    val sc = new SparkContext(conf)
    sc.addSparkListener(new org.apache.spark.scheduler.JobLogger())

    Thread.sleep(10000)

    val trainFilePaths = FileSystem.get(new URI(trainingDirName), new Configuration(true))
      .listStatus(new Path(trainingDirName))
      .filter(x => !x.isDir())
      .map(x => x.getPath().toUri())

    val trainNumParts = trainFilePaths.length
    val outputTrainPath = s"/home/eecs/vaishaal/ckm/keystone_pipeline/bin/brew-imagenet.sh ${trainingDirName}-brewed"
    println(s"Output train path ${outputTrainPath}")
    sc.parallelize(trainFilePaths, trainNumParts).pipe(outputTrainPath).count

    val testFilePaths =  FileSystem.get(new URI(testingDirName), new Configuration(true))
      .listStatus(new Path(testingDirName))
      .filter(x => !x.isDir())
      .map(x => x.getPath().toUri())

    val outputTestPath = s"/home/eecs/vaishaal/ckm/keystone_pipeline/bin/brew-imagenet.sh ${testingDirName}-brewed"
    println(s"Output test path ${outputTestPath}")

    val testNumParts = testFilePaths.length
    sc.parallelize(testFilePaths, testNumParts).pipe(outputTestPath).count

    sc.stop()
    sys.exit(0)
  }
}
