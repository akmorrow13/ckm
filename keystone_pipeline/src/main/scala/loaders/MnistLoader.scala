package loaders


import java.io.{ File, FileInputStream, FileOutputStream, DataInputStream }
import java.net.URL
import java.nio.file.{ Files, Paths }

import java.util.zip.GZIPInputStream
import breeze.linalg._

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import utils.{ImageMetadata, LabeledImage, ColumnMajorArrayVectorizedImage}
import scala.reflect._


/**
 * Loads images from the MNIST Dataset.
 * MNIST image/label reading functions borrowed from: https://github.com/alno/scalann
 */

class MnistFileReader(location: String, fileName: String) {

  private[this] val path = Paths.get(location, fileName)

  protected[this] val stream = new DataInputStream((new FileInputStream(path.toString)))

}


class MnistLabelReader(location: String, fileName: String) extends MnistFileReader(location, fileName) {

  assert(stream.readInt() == 2049, "Wrong MNIST label stream magic")

  val count = stream.readInt()

  val labelsAsInts = readLabels(0)

  val labelsAsVectors = labelsAsInts.map { label =>
    DenseVector.tabulate[Double](10) { i => if (i == label) 1.0 else 0.0 }
  }

  private[this] def readLabels(ind: Int): Stream[Int] =
    if (ind >= count)
      Stream.empty
    else
      Stream.cons(stream.readByte(), readLabels(ind + 1))

}

class MnistImageReader(location: String, fileName: String) extends MnistFileReader(location, fileName) {

  assert(stream.readInt() == 2051, "Wrong MNIST image stream magic")

  val count = stream.readInt()
  val width = stream.readInt()
  val height = stream.readInt()

  val imagesAsArrays = readImages(0)

  private[this] def readImages(ind: Int): Stream[Array[Double]] =
    if (ind >= count)
      Stream.empty
    else
      Stream.cons(readImage(), readImages(ind + 1))

  private[this] def readImage(): Array[Double] = {
    val m = new Array[Double](height*width)
    for (y <- 0 until height; x <- 0 until width) {
      m(y + x * height) = stream.readUnsignedByte()/255.0
    }
    m
  }

}


object MnistLoader {
  // We hardcode this because these are properties of the MNIST dataset.
  val nrow = 28
  val ncol = 28
  val nchan = 1
  val numClasses = 10
  val labelSize = 1

  def mnistToBufferedImage(mnist: Array[Double]): ColumnMajorArrayVectorizedImage = {
    ColumnMajorArrayVectorizedImage(mnist, ImageMetadata(nrow, ncol, nchan))
  }

  def apply(sc: SparkContext, path: String, partitions: Int, dataset: String): RDD[LabeledImage] = {
    val fName =
      if (dataset == "train") {
        "train"
      } else if (dataset == "test") {
        "t10k"
      } else {
        assert(false, "Unknown dataset")
      }
    lazy val imageReader = new MnistImageReader(path, s"${fName}-images-idx3-ubyte")
    lazy val labelReader = new MnistLabelReader(path, s"${fName}-labels-idx1-ubyte")

    def imageWidth = imageReader.width
    def imageHeight = imageReader.height

    def imagesAsArrays = imageReader.imagesAsArrays
    def labelsAsInts = labelReader.labelsAsInts

    def examples = (imagesAsArrays zip labelsAsInts).map(x => LabeledImage(mnistToBufferedImage(x._1), x._2))
    sc.parallelize(examples)

    }
}


