package utils

import breeze.linalg.DenseVector

object ChannelConverter {

  val bases: Array[Char] = Array('A','C','G','T')

  def apply(in: String, length: Option[Int]): DenseVector[Double] = {

    // fill in with Ns if has variable length
    val sequence =
    length match {
      case Some(_) => {
        val extras = length.get - in.length
        val ns = Array.fill[Char](extras)('N')
        val s= in.toUpperCase.toCharArray ++ ns
        assert(s.length == length.get)
        s
      }
      case None => in.toUpperCase.toCharArray
    }


    val n = length match {
      case Some(_) => length.get
      case None => in.length
    }
    val paddingRows = 2
    val rows = n + 2 * paddingRows
    val cols = 4 // number of bases

    val total = rows * cols
    val s: DenseVector[Double] = DenseVector.zeros(total)

    // fill in first and last rows with padding
    for (i <- 0 to paddingRows) {
      s(0 to paddingRows * cols) := 0.25
      s((total - paddingRows * cols) to total-1) := 0.25
    }

    // Fill in for sequence
    sequence.zipWithIndex.map { e => {
      val idx = (e._2 + paddingRows) * cols
      val baseIdx = bases.indexOf(e._1)
      if (baseIdx >= 0)
        s(idx + baseIdx) = 1.0
      else
        s(idx to idx+cols) := 0.25
      }
    }

    return s
  }

  def printConvertedSequence(sequence: String, x: DenseVector[Double], length: Int, channels: Int) = {
    println(s"printing sequence ${sequence}")
    for (i <- 0 to length) {
      for (j <- 0 to channels) {
        print(x(i * j + j))
        print("\t")
      }
      println
    }
  }
}

