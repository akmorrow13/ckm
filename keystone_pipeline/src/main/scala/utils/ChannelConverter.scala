package utils

import breeze.linalg.DenseVector

object ChannelConverter {

  val bases: Array[Char] = Array('A','C','G','T')

  def apply(in: String, m: Int = 3, save: Boolean = false, filename: Option[String] = None): DenseVector[Double] = {
    // TODO: Alyssa what is m?

    val sequence = in.toCharArray

    val n = in.toArray.length
    val rows = n + 2 * m - 2
    val cols = 4 // number of bases

    var s = new DenseVector[Double](rows * cols)

    for (i <- Range(1, rows+1)) {
      for (j <- Range(1, cols+1)) {
        val pos = (i-1) * cols + j-1
        if ((i < m) ||  (i >= n + m ) || sequence(i-m) == 'N' ) //TODO: Alyssa is middle one right?
          s(pos) = 0.25
        else if (sequence(i - m) == bases(j-1))
          s(pos) = 1
        else
          s(pos) = 0
      }

    }

    return s
  }

}


