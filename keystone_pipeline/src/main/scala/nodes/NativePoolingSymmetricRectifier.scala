package nodes.images

import breeze.linalg.DenseVector
import pipelines._
import utils.{ImageMetadata, ChannelMajorArrayVectorizedImage, Image}
import workflow.Transformer

import utils.external.NativeRoutines

/**
 * This node takes an image and performs pooling on regions of the image.
 * based on Pooler and Symmetric Rectifier. Doubles the number of channels. 
  */



class NativePoolingSymmetricRectifier(
  stride: Int,
  poolSize: Int,
  maxVal: Double = 0.0, alpha: Double = 0.0,
  numChannels: Int = 3, xDim: Int, yDim: Int
)
  extends Transformer[Array[Double], Array[Double]] {

  @transient lazy val extLib = new NativeRoutines()

  def apply(image: Array[Double]) = {
    val out:Array[Double] = extLib.poolAndRectify(stride, poolSize, numChannels, xDim, yDim, maxVal, alpha, image)
    out
  }
}
