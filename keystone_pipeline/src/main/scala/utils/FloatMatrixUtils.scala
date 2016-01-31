package utils

import breeze.linalg.{DenseMatrix, convert, max, min}
import breeze.plot._

object FloatMatrixUtils {
  def visualize(mat: DenseMatrix[Float]): Figure = {
    val f2 = Figure()
    f2.height = mat.rows
    f2.width = mat.cols
    f2.subplot(0) += image(convert(mat, Double), GradientPaintScale(min(mat), max(mat), PaintScale.BlackToWhite))
    f2.subplot(0).yaxis.setInverted(true)
    f2
  }

  /*def tiffToMatrix(fileName: String): DenseMatrix[Float] = {
    processorToMatrix(new Opener().openImage(fileName).getProcessor)
  }

  def processorToMatrix(in: ImageProcessor): DenseMatrix[Float] = {
    val data = in.getFloatArray
    val flatData = data.flatten
    new DenseMatrix[Float](in.getHeight, in.getWidth, flatData) //Todo: verify that this went in the right direction.
  }*/
}
