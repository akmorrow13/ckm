package nodes.images

import breeze.linalg._
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import pipelines._
import utils.{ChannelMajorArrayVectorizedImage, ImageMetadata, _}
import workflow.Transformer

/**
 * Convolve Cosine and Pool (CCaP)
 * Convolves images with a bank of convolution filters. Convolution filters must be square.
 * Used for using the same label for all patches from an image. Then apply a Cosine non linearity and sum pool
 * in one implementation
 *
 * TODO: Look into using Breeze's convolve
 *
 * @param filters Bank of convolution filters to apply - each filter is an array in row-major order.
 * @param imgWidth Width of images in pixels.
 * @param imgHeight Height of images in pixels.
 */
class CCaP(
    filters: DenseMatrix[Double],
    imgWidth: Int,
    imgHeight: Int,
    imgChannels: Int,
    stride: Int,
    poolSize: Int,
    maxVal: Double = 0.0,
    alpha: Double = 0.0,
    normalizePatches: Boolean = true,
    varConstant: Double = 10.0)
  extends Transformer[Image, Image] {

  val convSize = math.sqrt(filters.cols/imgChannels).toInt
  val convolutions = filters.t

  val resWidth = imgWidth - convSize + 1
  val resHeight = imgHeight - convSize + 1

  override def apply(in: RDD[Image]): RDD[Image] = {
    in.mapPartitions(CCaP.convolvePartitions(_, resWidth, resHeight, imgChannels, stride, poolSize, maxVal, alpha, convSize,
      normalizePatches, convolutions, varConstant))
  }

  def apply(in: Image): Image = {
    var patchMat = new DenseMatrix[Double](resWidth*resHeight, convSize*convSize*imgChannels)
    val image = CCaP.convolve(in, patchMat, resWidth, resHeight,
      imgChannels, stride, poolSize, maxVal, alpha, convSize, normalizePatches, convolutions)
    image
  }
}

object CCaP {
  def convolve(img: Image,
      patchMat: DenseMatrix[Double],
      resWidth: Int,
      resHeight: Int,
      imgChannels: Int,
      stride: Int,
      poolSize: Int,
      maxVal: Double = 0.0,
      alpha: Double = 0.0,
      convSize: Int,
      normalizePatches: Boolean,
      convolutions: DenseMatrix[Double],
      varConstant: Double = 10.0): Image = {

    val imgMat = makePatches(img, patchMat, resWidth, resHeight, imgChannels, convSize,
      normalizePatches, varConstant)

    val convRes: DenseMatrix[Double] = imgMat * convolutions


    val xDim = resWidth
    val yDim = resHeight
    val numSourceChannels = convolutions.cols
    val numOutChannels = numSourceChannels
    val strideStart = poolSize / 2

    val numPoolsX = math.ceil((xDim - strideStart).toDouble / stride).toInt
    val numPoolsY = math.ceil((yDim - strideStart).toDouble / stride).toInt
    val patch = new Array[Double]( numPoolsX * numPoolsY * numOutChannels)

    // Start at strideStart in (x, y) and
    var x = strideStart
    while (x < xDim) {
      var y = strideStart
      while (y < yDim) {
        var c = 0
        while (c < numSourceChannels) {
          // Extract the pool. Then apply the pixel and pool functions
          val startX = x - poolSize/2
          val endX = math.min(x + poolSize/2, xDim)
          val startY = y - poolSize/2
          val endY = math.min(y + poolSize/2, yDim)
          val output_offset = (x - strideStart)/stride * numOutChannels +
          (y - strideStart)/stride * numPoolsX * numOutChannels
          var s = startX
          while(s < endX) {
            var b = startY
            while(b < endY) {
              val pix =  convRes(s + b*resWidth, c)
              val outvar =  math.cos(pix)

              val pos_position = c + output_offset
              patch(pos_position) += outvar
              b = b + 1
              }
              s = s + 1
            }
            c = c + 1
            }
            y += stride
          }
          x += stride
          }

          ChannelMajorArrayVectorizedImage(patch, ImageMetadata(numPoolsX, numPoolsY, numOutChannels))
        }

  /**
   * This function takes an image and generates a matrix of all of its patches. Patches are expected to have indexes
   * of the form: c + x*numChannels + y*numChannels*xDim
   *
   * @param img
   * @return
   */
  def makePatches(img: Image,
      patchMat: DenseMatrix[Double],
      resWidth: Int,
      resHeight: Int,
      imgChannels: Int,
      convSize: Int,
      normalizePatches: Boolean,
      varConstant: Double): DenseMatrix[Double] = {
    var x,y,chan,pox,poy,py,px = 0

    x = 0
    while (x < resWidth) {
      y = 0
      while (y < resHeight) {
          poy = 0
          while (poy < convSize) {
            pox = 0
            while (pox < convSize) {
              chan = 0
              while (chan < imgChannels) {

                px = chan + pox*imgChannels + poy*imgChannels*convSize
                py = x + y*resWidth

                patchMat(py, px) = img.get(x+pox, y+poy, chan)

                chan+=1
              }
              pox+=1
            }
            poy+=1

          }
        y+=1
      }
      x+=1
    }

    val patchMatN = if(normalizePatches) Stats.normalizeRows(patchMat, varConstant) else patchMat

    patchMatN
  }

  def convolvePartitions(
      imgs: Iterator[Image],
      resWidth: Int,
      resHeight: Int,
      imgChannels: Int,
      stride: Int,
      poolSize: Int,
      maxVal: Double = 0.0,
      alpha: Double = 0.0,
      convSize: Int,
      normalizePatches: Boolean,
      convolutions: DenseMatrix[Double],
      varConstant: Double): Iterator[Image] = {

    var patchMat = new DenseMatrix[Double](resWidth*resHeight, convSize*convSize*imgChannels)
    imgs.map(convolve(_, patchMat, resWidth, resHeight, imgChannels, stride, poolSize, maxVal, alpha, convSize, normalizePatches,
      convolutions, varConstant))
  }
}
