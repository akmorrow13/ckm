package loaders

import java.io.FileInputStream

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import utils.{ImageMetadata, LabeledImage, RowMajorArrayVectorizedImage, ColumnMajorArrayVectorizedImage}


/**
 * Loads images from the CIFAR-10 Dataset.
 */
object CifarLoader2 {
  // We hardcode this because these are properties of the CIFAR-10 dataset.
  val nrow = 32
  val ncol = 32
  val nchan = 3

  val labelSize = 1

  def cifar10ToBufferedImage(cifar: Array[Byte]): ColumnMajorArrayVectorizedImage = {
    val byteLen = nrow*ncol*nchan
    // Allocate some space for the rows.
    require(cifar.length == byteLen, "CIFAR-10 Images MUST be 32x32x3.")
    ColumnMajorArrayVectorizedImage(cifar.map(_.toInt/1.0).map(x => if (x < 0) x + 256 else x).map(_/255.0), ImageMetadata(nrow, ncol, nchan))
  }

  def loadLabeledImages(path: String): Seq[LabeledImage] = {
    val imgCount = labelSize + nrow*ncol*nchan

    val imageBytes = Array.fill[Byte](imgCount)(0x00)
    var out = Array[LabeledImage]()

    val inFile = new FileInputStream(path)

    while(inFile.read(imageBytes, 0, imgCount) > 0) {
      val img = cifar10ToBufferedImage(imageBytes.tail)
      val label = imageBytes.head.toShort
      val li = LabeledImage(img, label)
      out = out :+ li
    }
    out
  }

  def apply(sc: SparkContext, path: String): RDD[LabeledImage] = {
    val images = CifarLoader2.loadLabeledImages(path)
    sc.parallelize(images)
  }
}
