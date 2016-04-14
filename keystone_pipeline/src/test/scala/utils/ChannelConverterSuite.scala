package utils

import breeze.linalg.DenseVector
import org.scalatest._

class ChannelConverterSuite extends FunSuite {
   test("basic ATGG test") {

    val sequence = "ATGG"

    val result: DenseVector[Double] = ChannelConverter(sequence)

    assert(result(0) == 0.25)
    assert(result(8) == 1.0)
    assert(result(29) == 0.25)
    assert(result(18) == 1.0)
    assert(result(15) == 1.0)
  }
}