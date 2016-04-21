package loaders


import java.nio.file.{Files, Paths}

import breeze.linalg.DenseVector
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import utils._

import scala.collection.mutable.ListBuffer


class DREAM5TFReader(sc: SparkContext, location: String, fileName: String, sample: Boolean = true) {
  // We hardcode this because these are properties of the MNIST dataset.
  val length = 40
  val nchan = 4 // A, T, G, C
  val labelSize = 1

  def getLength = length
  def getChannels = nchan

  val loc = s"${location}/${fileName}"
  val p = Paths.get(loc)
  assert(Files.exists(p))

  val lines = sc.textFile(loc)



  var X = lines.map(_.split("\t"))
  if (sample) {
    X = X.sample(false, 0.01)
    println("samples chosen: ", X.count)
  }
  var sequences =    new ListBuffer[Array[Double]]()
  var labels =    new ListBuffer[Double]()

  val tf = X.map(r => (ChannelConverter(r(2)).toArray, r(3).toDouble))

  def getRDD(): RDD[(Array[Double], Double)] = tf

}


object DREAM5Loader {

  def apply(sc: SparkContext, path: String, partitions: Int, dataset: String, filename: String, sample: Boolean = false): RDD[LabeledSequence] = {

    val fileLocation = s"${path}/SEQUENCE_INPUT/${filename}"
    println(fileLocation)
    val f = Paths.get(fileLocation)

    val rdd: RDD[LabeledSequence] =
      if (Files.exists(f)) {
        // load from files
        // TODO: Alyssa read all files in folder
        val data = sc.objectFile[SaveableArray](fileLocation)
        data.map(r => LabeledSequence(RowMajorArrayVectorizedSequence(r.sequence, r.metadata), r.label))

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
        val tfReader = new DREAM5TFReader(sc, path, s"${fName}", sample)
        val rdd = tfReader.getRDD
        rdd.persist
        println(s"Saving input sequences to ${fileLocation}")
        val l = tfReader.getLength
        val chan = tfReader.getChannels
        val labeled = rdd.map(r => LabeledSequence(RowMajorArrayVectorizedSequence(r._1, SequenceMetadata(l, chan)), r._2))
        val saveable = labeled.map(r => SaveableArray(r.sequence.toArray, r.sequence.metadata, r.label))
        saveable.saveAsObjectFile(fileLocation)
        labeled
      }
    println(s"loaded ${dataset}")
    rdd
  }
}

case class SaveableArray(sequence: Array[Double], metadata: SequenceMetadata, label: Double) extends Serializable
case class SaveableVector(sequence: DenseVector[Double], label: Double) extends Serializable




