package nodes.stats

import breeze.linalg.{*, DenseMatrix, DenseVector}
import breeze.stats.distributions._
import breeze.numerics.cos
import breeze.stats.distributions.Rand
import org.apache.spark.rdd.RDD
import utils.MatrixUtils
import org.apache.commons.math3.random.MersenneTwister
import workflow.Transformer

/**
  * Transformer that extracts random cosine features from a feature vector
  * @param W A matrix of dimension (# output features) by (# input features)
  * @param b a dense vector of dimension (# output features)
  *
  * Transformer maps vector x to cos(x * transpose(W) + b).
  * Kernel trick to allow Linear Solver to learn cosine interaction terms of the input
  */
class SeededCosineRandomFeatures(numInputFeatures:Int, numOutputFeatures:Int, seed: Int, gamma: Double)
  extends Transformer[DenseVector[Double], DenseVector[Double]] {

  override def apply(in: RDD[DenseVector[Double]]): RDD[DenseVector[Double]] = {

    in.mapPartitions { part =>
      implicit val randBasis: RandBasis = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(seed)))
      val gaussian = new Gaussian(0, 1)
      val uniform = new Uniform(0, 1)
      val W = DenseMatrix.rand(numOutputFeatures, numInputFeatures, gaussian) :* gamma
      val b = DenseVector.rand(numOutputFeatures, uniform) :* (2*math.Pi)
      val data = MatrixUtils.rowsToMatrix(part)
      val features: DenseMatrix[Double] = data * W.t
      features(*,::) :+= b
      cos.inPlace(features)
      MatrixUtils.matrixToRowArray(features).iterator
    }
  }

  override def apply(in: DenseVector[Double]): DenseVector[Double] = {
    implicit val randBasis: RandBasis = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(seed)))
    val gaussian = new Gaussian(0, 1)
    val uniform = new Uniform(0, 1)
    val W = DenseMatrix.rand(numOutputFeatures, numInputFeatures, gaussian) :* gamma
    val b = DenseVector.rand(numOutputFeatures, uniform) :* (2*math.Pi)
    val features = (in.t * W.t).t
    features :+= b
    cos.inPlace(features)
    features
  }
}

/**
 * Companion Object to generate random cosine features from various distributions
 */
object SeededCosineRandomFeatures {
  /** Generate Random Cosine Features from the given distributions **/
  def apply(
               numInputFeatures: Int,
               numOutputFeatures: Int,
               gamma: Double,
               seed: Int
               ) = {
    new SeededCosineRandomFeatures(numInputFeatures, numOutputFeatures, seed, gamma)
  }
}
