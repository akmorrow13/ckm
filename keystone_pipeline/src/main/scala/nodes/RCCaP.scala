package nodes.images

import breeze.linalg._
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import pipelines._
import utils.{ChannelMajorArrayVectorizedImage, ImageMetadata, _}
import workflow.Transformer
import nodes.learning.ZCAWhitener
import breeze.stats.distributions._
import org.apache.commons.math3.random.MersenneTwister

/**
 * Random Convolve Cosine and Pool (RCCaP)
 * Convolves images with a bank of random filters
 * Entropy is passed in as seed
 * Used for using the same label for all patches from an image. Then apply a Cosine non linearity and sum pool
 * in one implementation
 *
 * A few quirks:
 *  * During the convolution, every overlapping k x k image patch is normalized to have unit norm
 *  * After the convolution and non linearity the norms are multiplied back
 *
 * @param imgWidth Width of images in pixels.
 * @param imgHeight Height of images in pixels.
 */
class RCCaP(
    numInputFeatures: Int,
    numOutputFeatures: Int,
    seed: Int,
    bandwidth: Double,
    imgWidth: Int,
    imgHeight: Int,
    imgChannels: Int,
    stride: Int,
    poolSize: Int,
    whitener: Option[ZCAWhitener] = None)
  extends Transformer[Image, Image] {
  val convSize = math.sqrt(numInputFeatures/imgChannels).toInt
  val resWidth = imgWidth - convSize + 1
  val resHeight = imgHeight - convSize + 1

  val strideStart = poolSize / 2


  val outX = math.ceil((resWidth - strideStart).toDouble / stride).toInt
  val outY = math.ceil((resHeight- strideStart).toDouble / stride).toInt


  override def apply(in: RDD[Image]): RDD[Image] = {
    in.mapPartitions(RCCaP.convolvePartitions(_, resWidth, resHeight, imgChannels, stride, poolSize, convSize, whitener, numInputFeatures, numOutputFeatures, seed, bandwidth))
  }

  def apply(in: Image): Image = {
    var patchMat = new DenseMatrix[Double](resWidth*resHeight, convSize*convSize*imgChannels)
    val image = RCCaP.convolve(in, patchMat, resWidth, resHeight,
      imgChannels, stride, poolSize, convSize, whitener, numInputFeatures, numOutputFeatures, seed, bandwidth)
    image
  }
}

object RCCaP {
  def convolve(img: Image,
      patchMat: DenseMatrix[Double],
      resWidth: Int,
      resHeight: Int,
      imgChannels: Int,
      stride: Int,
      poolSize: Int,
      convSize: Int,
      whitener: Option[ZCAWhitener] = None,
      numInputFeatures: Int,
      numOutputFeatures: Int,
      seed: Int,
      bandwidth: Double
      ): Image = {

      implicit val randBasis: RandBasis = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(seed)))
    val gaussian = new Gaussian(0, 1)
    val uniform = new Uniform(0, 1)
    val convolutions = (DenseMatrix.rand(numOutputFeatures, numInputFeatures, gaussian) :* bandwidth).t
    val phase = DenseVector.rand(numOutputFeatures, uniform) :* (2*math.Pi)

    val imgMat = makePatches(img, patchMat, resWidth, resHeight, imgChannels, convSize)
    val whitenedImage =
    whitener match  {
      case None => {
        imgMat
      }
      case Some(whitener) => {
        whitener(imgMat)
      }
    }

    val patchNorms = norm(whitenedImage :+ 1e-12, Axis._1)
    val normalizedPatches = whitenedImage(::, *) :/ patchNorms

    /*
    println("CONVOLUTION SIZE IS " + convSize)
    println("RES WIDTH IS " + resWidth)
    println("RES Height IS " + resHeight)
    println("PHASE SHAPE IS " + phase.size)
    println(s"imgMat SHAPE ${imgMat.rows}, ${imgMat.cols}")
    println(s"PATCH SHAPE ${normalizedPatches.rows}, ${normalizedPatches.cols}")
    println(s"CONVOLUTION SHAPE ${convolutions.rows}, ${convolutions.cols}")
    */
    val convRes =  normalizedPatches * convolutions
    val xDim = resWidth
    val yDim = resHeight
    val numSourceChannels = convolutions.cols
    val numOutChannels = numSourceChannels
    val strideStart = poolSize / 2

    val numPoolsX = math.ceil((xDim - strideStart).toDouble / stride).toInt
    val numPoolsY = math.ceil((yDim - strideStart).toDouble / stride).toInt
    val patch = Array.fill[Double](numPoolsX * numPoolsY * numOutChannels)(0)
    val blurSigma = poolSize/math.sqrt(2)
    val gaussianWeights = false

    // NOTE: While loops in scala are ~10x faster than for loops
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
              val weight =
              if (gaussianWeights) {
                val patchDist = math.pow(s - x, 2) + math.pow(b - y, 2)
                math.exp((-0.5/(blurSigma*blurSigma))*patchDist)
              } else {
                1.0
              }
              val patchNorm = patchNorms(s + b*resWidth)
              val pix =  convRes(s + b*resWidth, c)
              val outvar =  weight * patchNorm * math.cos(pix + phase(c))
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
      whitener: Option[ZCAWhitener] = None
      ): DenseMatrix[Double] = {
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
    patchMat
  }

  def convolvePartitions(
      imgs: Iterator[Image],
      resWidth: Int,
      resHeight: Int,
      imgChannels: Int,
      stride: Int,
      poolSize: Int,
      convSize: Int,
      whitener: Option[ZCAWhitener] = None,
      numInputFeatures: Int,
      numOutputFeatures: Int,
      seed: Int,
      bandwidth: Double
      ): Iterator[Image] = {

    var patchMat = new DenseMatrix[Double](resWidth*resHeight, convSize*convSize*imgChannels)
    imgs.map(convolve(_, patchMat, resWidth, resHeight, imgChannels, stride, poolSize, convSize, whitener, numInputFeatures, numOutputFeatures, seed, bandwidth))
  }
}
