package loaders


import java.nio.file.{Files, Paths}

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import utils._

import scala.collection.mutable.ListBuffer


class DREAM5TFReader(sc: SparkContext,location: String, fileName: String) {
  // We hardcode this because these are properties of the MNIST dataset.
  val length = 40
  val nchan = 4 // A, T, G, C
  val labelSize = 1

  def getLength = length
  def getChannels = nchan

  val lines = sc.textFile(s"${location}/${fileName}")

//  var lines = Source.fromFile(s"${location}/${fileName}").getLines

  var X = lines.map(_.split("\t"))

  // remove labels: should be gone

  var sequences =    new ListBuffer[Array[Double]]()
  var labels =    new ListBuffer[Double]()

  val tf = X.map(r => (ChannelConverter(r(2)).toArray, r(3).toDouble))

  def get(): RDD[(Array[Double], Double)] = tf

}


object DREAM5Loader {

  def apply(sc: SparkContext, path: String, partitions: Int, dataset: String, filename: String): RDD[LabeledSequence] = {

    val fileLocation = s"/Users/akmorrow/Documents/COMPBIO294/Project/DREAM_data/SEQUENCE_INPUT/${filename}"

    val f = Paths.get(fileLocation)

    val rdd: RDD[LabeledSequence] =
      if (Files.exists(f)) {
        // load from files
        // TODO: Alyssa read all files in folder
          sc.textFile(fileLocation).map(r => r.split(","))
            .map(r =>
              LabeledSequence( RowMajorArrayVectorizedSequence(r(0).split(" ").map(_.toDouble), SequenceMetadata(r(1).toInt,r(3).toInt))
              , r(3).toDouble)
              )

      } else {
        // compute files
        val fName =
          if (dataset == "train") {
            "train.txt"
          } else if (dataset == "test") {
            "test.txt"
          } else {
            assert(false, "Unknown dataset")
          }
        val tfReader = new DREAM5TFReader(sc, path, s"${fName}")
        val rdd = tfReader.get
        println(rdd.count)
        rdd.persist
        println(s"Saving input sequences to ${fileLocation}")
        val l = tfReader.getLength
        val chan = tfReader.getChannels
//        rdd.map(r => (r._1.mkString(" "), l, chan, r._2))
//            .saveAsTextFile(fileLocation)

        rdd.map(r => LabeledSequence(RowMajorArrayVectorizedSequence(r._1, SequenceMetadata(l, chan)), r._2))
      }
    println(s"loaded ${dataset}")
    rdd
  }
}





