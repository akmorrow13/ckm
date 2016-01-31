package nodes.stats

import breeze.linalg._
import breeze.stats.distributions._
import workflow.Transformer

/**
 *  A node that takes in DenseVector[Double] and returns a smaller vector containing only the
 *  specified positions
 */
case class RandomVectorSubsampler(positions: Array[Int])
    extends Transformer[DenseVector[Double], DenseVector[Double]] {

  def apply(in: DenseVector[Double]): DenseVector[Double] = {
    val out = DenseVector.zeros[Double](positions.length)
    var i = 0
    while (i < positions.length) {
      out(i) = in(positions(i))
      i = i + 1
    }

    out
  }
}

object RandomVectorSubsampler {
  /* Create a vector sampling node */
  def apply(inputSize: Int, numToSample: Int): RandomVectorSubsampler = {
    require(inputSize >= numToSample, "Can't sample more than the input size")
    require(inputSize >= 0, "inputSize must be nonnegative")
    require(numToSample >= 0, "numToSample must be nonnegative")

    val mask = Seq.fill(numToSample)(true) ++ Seq.fill(inputSize - numToSample)(false)

    val shuffledMask = scala.util.Random.shuffle(mask)
    val maskIndices = shuffledMask.zipWithIndex.filter(_._1).map(_._2).toArray

    new RandomVectorSubsampler(maskIndices)
  }
}
