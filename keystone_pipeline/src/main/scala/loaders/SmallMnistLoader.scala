package loaders


import java.io.{ File, FileInputStream, FileOutputStream, DataInputStream }
import java.net.URL
import java.nio.file.{ Files, Paths }

import java.util.zip.GZIPInputStream
import breeze.linalg._

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import utils.{Image, ImageMetadata, LabeledImage, ColumnMajorArrayVectorizedImage}

import scala.reflect._
import scala.collection.mutable.ListBuffer

object SmallMnistLoader {
  // We hardcode this because these are properties of the MNIST dataset.
  val nrow = 8
  val ncol = 8
  val nchan = 1
  val numClasses = 10
  val labelSize = 1

  def mnistToBufferedImage(mnist: Array[Double]): ColumnMajorArrayVectorizedImage = {
    ColumnMajorArrayVectorizedImage(mnist, ImageMetadata(nrow, ncol, nchan))
  }

  def apply(sc: SparkContext, path: String, partitions: Int, dataset: String): RDD[LabeledImage] = {
      var X:DenseMatrix[Double] = csvread(new java.io.File(s"${path}/X_${dataset}"))
      var y:DenseMatrix[Double] = csvread(new java.io.File(s"${path}/y_${dataset}"))
      println(s"Dataset: ${dataset}, Rows ${X.rows}, Cols ${X.cols}")
      var images = new ListBuffer[Image]()
      var labels = new ListBuffer[Int]()
      for (i <- 0 until X.rows) {
       val row = X(i, ::).inner
       images +=  mnistToBufferedImage(row.toArray)
       labels += y(i, 0).toInt
      }

      val examples = (images zip labels).map(x => LabeledImage(x._1, x._2))
      println("IMAGES SIZE " + images.size)
      println("EXAMPLES SIZE " + examples.size)
      sc.parallelize(examples)
    }
}


