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

    val s: Array[Double] = Array.range(0, (rows * cols)).map(_.toDouble)
    val mapped = s.zipWithIndex.map { e => {
      val i = (e._2 / cols)
      val j = (e._2 % cols)
      val pos = (i) * cols + j
      if ((i+1 < m) ||  (i+1 >= n + m ) || sequence(i+1-m) == 'N' ) //TODO: Alyssa is middle one right?
          0.25
        else if (sequence(i+1 - m) == bases(j))
          1
        else
          0
      }
    }

    return new DenseVector[Double](mapped)
  }

}


