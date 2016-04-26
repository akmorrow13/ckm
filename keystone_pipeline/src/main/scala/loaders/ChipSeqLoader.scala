package loaders

import org.apache.hadoop.fs._
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import utils._

import scala.collection.mutable.ListBuffer


class ChipSeqReader(sc: SparkContext, fileName: String, fs: FileSystem, sample: Boolean = true) {
  // We hardcode this because these are properties of the MNIST dataset.
  val length = 1000
  val nchan = 4 // A, T, G, C
  val labelSize = 1

  def getLength = length
  def getChannels = nchan

  assert(fs.exists(new Path(fileName)))

  val lines = sc.textFile(fileName).filter(r => !r.contains("signal"))


  var X: RDD[Array[String]] = lines.map(_.split("\t"))
  if (sample) {
    X = X.sample(false, 0.01)
    println("samples chosen: ", X.count)
  }
  var sequences =    new ListBuffer[Array[Double]]()
  var labels =    new ListBuffer[Double]()
  val lengthOpt: Option[Int] = Option(length)
  val test: Array[String] = X.first
  val tf = X.map(r => (ChannelConverter(r(5), Some(1000)).toArray, r(4).toDouble))

  def getRDD(): RDD[(Array[Double], Double)] = tf

}

class ChipSeqDREAM5Reader(lines: RDD[String]) {
  // We hardcode this because these are properties of the MNIST dataset.
  val nchan = 4 // A, T, G, C
  val labelSize = 1

  def getChannels = nchan


  val X: RDD[Array[String]] = lines.map(_.split("\t"))
  var sequences =    new ListBuffer[Array[Double]]()
  var labels =    new ListBuffer[Double]()
  val tf = X.map(r => (ChannelConverter(r(0), None).toArray, r(1).toDouble))

  println(X.first.head)
  val length = X.first.head.length

  def getLength = length
  def getRDD(): RDD[(Array[Double], Double)] = tf

}


object ChipSeqLoader {

  def apply(sc: SparkContext, fs: FileSystem, partitions: Int, dataset: String, filename: String, sample: Boolean = false): RDD[LabeledSequence] = {

    val fileLocation = s"SEQUENCE_INPUT/${filename}"


    val rdd: RDD[LabeledSequence] =
      if (fs.exists(new Path(fileLocation))) {
        // load from files
        // TODO: Alyssa read all files in folder

        val data = sc.objectFile[SaveableArray](fileLocation)
        data.map(r => LabeledSequence(RowMajorArrayVectorizedSequence(r.sequence, r.metadata), r.label))

      } else {
        // compute files
        val fName =
          if (dataset == "train") {
            "train.tsv"
          } else if (dataset == "test") {
            "test.tsv"
          } else {
            assert(false, "Unknown dataset")
          }

        val tfReader = new ChipSeqReader(sc, s"${fName}", fs, sample)
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

  def apply(sc: SparkContext, partitions: Int, data: RDD[String]): RDD[LabeledSequence] = {

    val tfReader = new ChipSeqDREAM5Reader(data)
    val rdd = tfReader.getRDD
    rdd.persist
    val l = tfReader.getLength
    val chan = tfReader.getChannels
    rdd.map(r => LabeledSequence(RowMajorArrayVectorizedSequence(r._1, SequenceMetadata(l, chan)), r._2))
  }
}




