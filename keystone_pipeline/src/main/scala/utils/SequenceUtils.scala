package utils

import breeze.linalg.DenseVector
import org.apache.spark.rdd.RDD
import pipelines.FunctionNode
import workflow.Transformer


/**
 * @param stride How big a step to take between patches.
 * @param windowSize Size of a patch.
 */
class SequenceWindower(
                stride: Int,
                windowSize: Int) extends FunctionNode[RDD[Sequence], RDD[Sequence]] {

  def apply(in: RDD[Sequence]) = {
    in.flatMap(getSequenceWindow)
  }

  def getSequenceWindow(sequence: Sequence) = {
    val xDim = sequence.metadata.dim
    val numChannels = sequence.metadata.numChannels

    (0 until xDim - windowSize + 1 by stride).map { x =>
      // Extract the window.
      val pool = new DenseVector[Double](windowSize * numChannels)
      val startX = x
      val endX = x + windowSize
      var c = 0
      while (c < numChannels) {
        var s = startX
        while (s < endX) {
            pool(c + (s-startX)*numChannels +
              (endX-startX)*numChannels) = sequence.get(s, c)
          s = s + 1
        }
        c = c + 1
      }
      ChannelMajorArrayVectorizedSequence(pool.toArray,
        SequenceMetadata(windowSize, numChannels))

    }
  }
}

/**
 * This node takes an sequence and performs pooling on regions of the sequence.
 *
 * Divides sequences into fixed size pools, but when fed with sequences of various
 * sizes may produce a varying number of pools.
 *
 * NOTE: By default strides start from poolSize/2.
 *
 * @param stride x and y stride to get regions of the sequence
 * @param poolSize size of the patch to perform pooling on
 * @param pixelFunction function to apply on every pixel before pooling
 * @param poolFunction pooling function to use on every region.
 */
class SequencePooler(
              stride: Int,
              poolSize: Int,
              pixelFunction: Double => Double,
              poolFunction: DenseVector[Double] => Double)
  extends Transformer[Sequence, Sequence] {

  val strideStart = poolSize / 2

  def apply(sequence: Sequence) = {
    val xDim = sequence.metadata.dim
    val numChannels = sequence.metadata.numChannels

    val numPoolsX = math.ceil((xDim - strideStart).toDouble / stride).toInt
    val numPoolsY = 1
    val patch = new Array[Double]( numPoolsX * numPoolsY * numChannels)

    // Start at strideStart in (x, y) and
    for (x <- strideStart until xDim by stride) {
      // Extract the pool. Then apply the pixel and pool functions

      val pool = DenseVector.zeros[Double](poolSize * poolSize)
      val startX = x - poolSize/2
      val endX = math.min(x + poolSize/2, xDim)

      var c = 0
      while (c < numChannels) {
        var s = startX
        while (s < endX) {
            pool((s-startX)) = // TODO: Alyssa
              pixelFunction(sequence.get(s, c))
          s = s + 1
        }
        patch(c + (x - strideStart)/stride * numChannels) = poolFunction(pool)
        c = c + 1
      }
    }
    ChannelMajorArrayVectorizedSequence(patch, SequenceMetadata(numPoolsX, numChannels))
  }
}


