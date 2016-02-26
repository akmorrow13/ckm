package nodes.images

import breeze.linalg._
import breeze.numerics._
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import workflow.Transformer
import utils.{ChannelMajorArrayVectorizedImage, ImageMetadata, _}

class CosineNode(phase: DenseVector[Double])
  extends Transformer[Image, Image] {

    def apply(in: Image): Image =  {
      val imMat =  new DenseMatrix[Double](in.metadata.yDim*in.metadata.xDim, in.metadata.numChannels, in.toArray)
      println(imMat.rows)
      println(imMat.cols)

      imMat(*,::) :+= phase
      cos.inPlace(imMat)
      new ChannelMajorArrayVectorizedImage(
        imMat.toArray,
        in.metadata)
    }

    override def apply(in: RDD[Image]): RDD[Image] = {
      in.map(apply(_))
    }
  }

