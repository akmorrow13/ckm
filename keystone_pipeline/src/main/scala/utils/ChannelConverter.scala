package utils

import breeze.linalg.DenseVector

object ChannelConverter {

  val bases: Array[Char] = Array('A','C','G','T')

  def apply(in: String, save: Boolean = false, filename: Option[String] = None): DenseVector[Double] = {

    // PBM: val length = None
    val length: Option[Int] = Some(1000)
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


    // PBM: val n = in.toArray.length
    val n = length.get
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



//object ChannelConverter {
//
//  val bases: Array[Char] = Array('A','C','G','T')
//
//  def apply(in: String, m: Int = 3, save: Boolean = false, filename: Option[String] = None): DenseVector[Double] = {
//    // TODO: Alyssa what is m?
//
//    val sequence = in.toCharArray
//
//    val n = in.toArray.length
//    val rows = n + 2 * m - 2
//    val cols = 4 // number of bases
//
//    val s: Array[Double] = Array.range(0, (rows * cols)).map(_.toDouble)
//    val mapped = s.zipWithIndex.map { e => {
//      val i = (e._2 / cols)
//      val j = (e._2 % cols)
//      val pos = (i) * cols + j
//      if ((i+1 < m) ||  (i+1 >= n + m ) || sequence(i+1-m) == 'N' ) //TODO: Alyssa is middle one right?
//        0.25
//      else if (sequence(i+1 - m) == bases(j))
//        1
//      else
//        0
//    }
//    }
//
//    return new DenseVector[Double](mapped)
//  }
//
//}

