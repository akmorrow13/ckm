package nodes.images

import breeze.linalg.DenseVector
import pipelines._
import utils.{ImageMetadata, ChannelMajorArrayVectorizedImage, Image}
import workflow.Transformer

/**
 * This node takes an image and performs pooling on regions of the image.
 * based on Pooler and Symmetric Rectifier. Doubles the number of channels. 
  */



class MyPooler(
  stride: Int,
  poolSize: Int,
  maxVal: Double = 0.0, alpha: Double = 0.0

)
  extends Transformer[Image, Image] {

  val strideStart = poolSize / 2

  def apply(image: Image) = {
    val xDim = image.metadata.xDim
    val yDim = image.metadata.yDim
    val numSourceChannels = image.metadata.numChannels
    val numOutChannels = numSourceChannels * 2

    val numPoolsX = math.ceil((xDim - strideStart).toDouble / stride).toInt
    val numPoolsY = math.ceil((yDim - strideStart).toDouble / stride).toInt
    val patch = new Array[Double]( numPoolsX * numPoolsY * numOutChannels)

    // Start at strideStart in (x, y) and
    var x = strideStart

    while (x < xDim) {
      var y = strideStart
      while (y < yDim) {
        // Extract the pool. Then apply the pixel and pool functions

        //val pool = DenseVector.zeros[Double](poolSize * poolSize)
        val startX = x - poolSize/2
        val endX = math.min(x + poolSize/2, xDim)
        val startY = y - poolSize/2
        val endY = math.min(y + poolSize/2, yDim)


        val output_offset = (x - strideStart)/stride * numOutChannels +
        (y - strideStart)/stride * numPoolsX * numOutChannels

        //for(c <- 0 to (numSourceChannels-1)) {
        var s = startX
        while(s < endX) {
          //for(s <- startX to (endX -1)) {
        var b = startY
          while(b < endY) {
            var c = 0
            while (c < numSourceChannels) {

              //for (b <- startY to (endY -1)) {
              val pix = image.get(s, b, c)
              val outvar_pos =  math.max(maxVal, pix - alpha)
              val outvar_neg =  math.max(maxVal, -pix - alpha)

              val pos_position = c + output_offset
              patch(pos_position) += outvar_pos

              val neg_position = c + numSourceChannels + output_offset
              patch(neg_position) += outvar_neg

              c = c + 1
            }

            b = b + 1
          }
          s = s + 1
        }
        // val pos_position = c + output_offset
        // patch(pos_position) = outvar_pos

        // val neg_position = c + numSourceChannels + output_offset
        // patch(neg_position) = outvar_neg
        y += stride
      }
      x += stride
    }
      
    ChannelMajorArrayVectorizedImage(patch, ImageMetadata(numPoolsX, numPoolsY, numOutChannels))
  }


  def applyOtherOrder(image: ChannelMajorArrayVectorizedImage) = {
    /*
     reading through the data once, not actually an improvement. 
     
     
     */ 
    val xDim = image.metadata.xDim
    val yDim = image.metadata.yDim
    val numSourceChannels = image.metadata.numChannels
    val numOutChannels = numSourceChannels * 2
    assert(stride == poolSize)

    val numPoolsX = math.ceil((xDim - strideStart).toDouble / stride).toInt
    val numPoolsY = math.ceil((yDim - strideStart).toDouble / stride).toInt
    val patch = new Array[Double]( numPoolsX * numPoolsY * numOutChannels)

    val maxX = math.min(xDim, numPoolsX * stride)
    val maxY = math.min(yDim, numPoolsY * stride)

    var y = 0
    while(y < maxY) {

      var x = 0
      while(x < maxX) {
        var patchX = x / poolSize
        var patchY = y / poolSize

        var output_offset =  (patchY * numPoolsX + patchX) * numOutChannels
        var c = 0
        while(c < numSourceChannels) { 
          val pix = image.get(x, y, c)
          val outvar_pos =  math.max(maxVal, pix - alpha)
          val outvar_neg =  math.max(maxVal, -pix - alpha)

          patch(output_offset + c) += outvar_pos
          patch(output_offset + c + numSourceChannels) += outvar_neg

          c += 1
        }
        x += 1
      }
      y += 1
    }
    ChannelMajorArrayVectorizedImage(patch, ImageMetadata(numPoolsX, numPoolsY, numOutChannels))
  }
}
