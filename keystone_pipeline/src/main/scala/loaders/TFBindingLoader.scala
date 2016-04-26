package loaders


import breeze.linalg.DenseVector
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import utils._

import scala.collection.mutable.ListBuffer


class DREAM5TFReader(sc: SparkContext, fileName: String, fs: FileSystem, sample: Boolean = true) {
  // We hardcode this because these are properties of the MNIST dataset.
  val length = 40
  val nchan = 4 // A, T, G, C
  val labelSize = 1

  def getLength = length
  def getChannels = nchan

  assert(fs.exists(new Path(fileName)))

  val lines = sc.textFile(fileName)



  var X = lines.map(_.split("\t"))
  if (sample) {
    X = X.sample(false, 0.01)
    println("samples chosen: ", X.count)
  }
  var sequences =    new ListBuffer[Array[Double]]()
  var labels =    new ListBuffer[Double]()

  val tf: RDD[(Array[Double], Double)] = X.map(r => (ChannelConverter(r(2), None).toArray, r(3).toDouble))

  def getRDD(): RDD[(Array[Double], Double)] = tf

}


object DREAM5Loader {

  def apply(sc: SparkContext, fs: FileSystem, partitions: Int, dataset: String, filename: String, sample: Boolean = false): RDD[LabeledSequence] = {

    val fileLocation = s"SEQUENCE_INPUT/${filename}"


    val rdd: RDD[LabeledSequence] =
      if (fs.exists(new Path(fileLocation))) {
        // load from files
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
        val tfReader = new DREAM5TFReader(sc, s"${fName}",fs, sample)
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
    println(rdd.count)
    rdd
  }
}

case class SaveableArray(sequence: Array[Double], metadata: SequenceMetadata, label: Double) extends Serializable
case class SaveableVector(sequence: DenseVector[Double], label: Double) extends Serializable




