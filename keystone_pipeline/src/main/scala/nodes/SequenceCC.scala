package nodes.images

import breeze.linalg._
import breeze.numerics._
import nodes.learning.ZCAWhitener
import nodes.learning.ZCAWhitener
import nodes.stats.Fastfood
import nodes.stats.Fastfood
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import pipelines._
import utils._
import workflow.Transformer
import breeze.stats.distributions._
import org.apache.commons.math3.random.MersenneTwister

/**
 * Convolves images with a bank of convolution filters. Convolution filters must be square.
 * Used for using the same label for all patches from an image.
 * TODO: Look into using Breeze's convolve
 *
 * @param imgWidth Width of images in pixels.
 * @param imgHeight Height of images in pixels.
 */
class SequenceCC(
          numInputFeatures: Int,
          numOutputFeatures: Int,
          seed: Int,
          bandwidth: Double,
          width: Int,
          imgChannels: Int,
          whitener: Option[ZCAWhitener] = None,
          whitenerOffset: Double = 1e-12,
          poolSize: Int = 1,
          insanity: Boolean = false,
          fastfood: Boolean = false
          )
  extends Transformer[Sequence, Sequence] {

  val convSize = (numInputFeatures/imgChannels).toInt // TODO: Alyssa removed Math.sqrt
  assert(convSize == 5)

  val resWidth = width - convSize + 1
//  val resHeight =  width - convSize + 1
  val outX = math.ceil((resWidth - (poolSize/2)).toDouble / poolSize).toInt
//  val outY = math.ceil((resHeight - (poolSize/2)).toDouble / poolSize).toInt

  override def apply(in: RDD[Sequence]): RDD[Sequence] = {
    println(s"Convolve: ${resWidth}, ${numOutputFeatures}")
    println(s"Input: ${width}, ${imgChannels}")
    println(s"First pixel ${in.take(1)(0).get(0,0)}")

    in.mapPartitions(SequenceCC.convolvePartitions(_, resWidth, imgChannels, convSize,
      whitener, whitenerOffset, numInputFeatures, numOutputFeatures, seed, bandwidth, insanity, fastfood))
  }

  def apply(in: Sequence): Sequence = {

    implicit val randBasis: RandBasis = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(seed)))
    val gaussian = new Gaussian(0, 1)
    val uniform = new Uniform(0, 1)
    val convolutions = (DenseMatrix.rand(numOutputFeatures, numInputFeatures, gaussian) :* bandwidth). t
    val phase = DenseVector.rand(numOutputFeatures, uniform) :* (2*math.Pi)
    var patchMat = new DenseMatrix[Double](resWidth, convSize*convSize*imgChannels)
    SequenceCC.convolve(in, patchMat, resWidth,
      imgChannels, convSize, whitener, whitenerOffset, convolutions.data, phase, insanity, None, numOutputFeatures, numInputFeatures)
  }
}

object SequenceCC {
  /**
   * Given an array of filters, packs the filters into a DenseMatrix[Double] which has the following form:
   * for a row i, column c+y*numChannels+x*numChannels*yDim corresponds to the pixel value at (x,y,c) in image i of
   * the filters array.
   *
   * @param filters Array of filters.
   * @return DenseMatrix of filters, as described above.
   */
  def packFilters(filters: Array[Sequence]): DenseMatrix[Double] = {
    val (xDim, numChannels) = (filters(0).metadata.dim, filters(0).metadata.numChannels)
    val filterSize = xDim*numChannels
    val res = DenseMatrix.zeros[Double](filters.length, filterSize)

    var i,x,y,c = 0
    while(i < filters.length) {
      x = 0
      while(x < xDim) {
        c = 0
        while (c < numChannels) {
          val rc = c + x*numChannels + y*numChannels*xDim
          res(i, rc) = filters(i).get(x,c)

          c+=1
        }
        x+=1
      }
      i+=1
    }
    res
  }


  def convolve(sequence: Sequence,
               patchMat: DenseMatrix[Double],
               resWidth: Int,
               imgChannels: Int,
               convSize: Int,
               whitener: Option[ZCAWhitener],
               whitenerOffset: Double,
               convolutions: Array[Double],
               phase: DenseVector[Double],
               insanity: Boolean,
               fastfood: Option[Fastfood],
               out: Int,
               in: Int
                ): Sequence = {

    val seqMat = makePatches(sequence, patchMat, resWidth, imgChannels, convSize,
      whitener)

    val whitenedSequence =
      whitener match  {
        case None => {
          seqMat
        }
        case Some(whitener) => {
          whitener(seqMat)
        }
      }

    val patchNorms = norm(whitenedSequence :+ whitenerOffset, Axis._1)
    val normalizedPatches = whitenedSequence(::, *) :/ patchNorms
    var convRes:DenseMatrix[Double] =
      fastfood.map { ff =>
        val ff_out = MatrixUtils.matrixToRowArray(normalizedPatches).map(ff(_))
        MatrixUtils.rowsToMatrix(ff_out)
      } getOrElse {
        val convRes = normalizedPatches * (new DenseMatrix(out, in, convolutions)).t
        convRes(*, ::) :+= phase
        cos.inPlace(convRes)
        if (insanity) {
          convRes(::,*) :*= patchNorms
        }
        convRes
      }

    val res = new RowMajorArrayVectorizedSequence(
      convRes.toArray,
      SequenceMetadata(resWidth, out))
    res
  }

  /**
   * This function takes an image and generates a matrix of all of its patches. Patches are expected to have indexes
   * of the form: c + x*numChannels + y*numChannels*xDim
   *
   * @param img
   * @return
   */
  def makePatches(img: Sequence,
                  patchMat: DenseMatrix[Double],
                  resWidth: Int,
                  imgChannels: Int,
                  convSize: Int,
                  whitener: Option[ZCAWhitener]
                   ): DenseMatrix[Double] = {
    var x,chan,pox,py,px = 0

    pox = 0
    while (pox < convSize) {
      x = 0
      while (x < resWidth) {
        chan = 0
        while (chan < imgChannels) {
          px = chan + pox*imgChannels
          py = x
          val k = img.get(x+pox, chan)
          patchMat(py, px) = k

          chan+=1
        }
        x+=1
      }
      pox+=1
    }

    patchMat
  }

  def convolvePartitions(
                          sequences: Iterator[Sequence],
                          resWidth: Int,
                          channels: Int,
                          convSize: Int,
                          whitener: Option[ZCAWhitener],
                          whitenerOffset: Double,
                          numInputFeatures: Int,
                          numOutputFeatures: Int,
                          seed: Int,
                          bandwidth: Double,
                          insanity: Boolean,
                          fastfood: Boolean
                          ): Iterator[Sequence] = {

    var patchMat = new DenseMatrix[Double](resWidth, convSize*channels)
    implicit val randBasis: RandBasis = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(seed)))
    val gaussian = new Gaussian(0, 1)
    val uniform = new Uniform(0, 1)
    val convolutions: Array[Double] =
      if (!fastfood) {
        (DenseMatrix.rand(numOutputFeatures, numInputFeatures, gaussian) :* bandwidth).data
      } else {
        (DenseVector.rand(numOutputFeatures, gaussian) :* bandwidth).data
      }

    val phase = DenseVector.rand(numOutputFeatures, uniform) :* (2*math.Pi)
    val ff =
      if (fastfood) {
        Some(new Fastfood(DenseVector(convolutions), phase, numOutputFeatures))
      } else {
        None
      }

    sequences.map(convolve(_, patchMat, resWidth, channels, convSize,
      whitener, whitenerOffset, convolutions, phase, insanity, ff, numOutputFeatures, numInputFeatures))
  }
}
