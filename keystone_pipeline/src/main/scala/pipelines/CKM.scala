package pipelines

import breeze.stats.distributions._
import breeze.linalg._
import breeze.numerics._
import evaluation.MulticlassClassifierEvaluator
import loaders.{CifarLoader, MnistLoader, SmallMnistLoader}
import nodes.images._
import workflow.Transformer
import nodes.learning.{BlockLeastSquaresEstimator, BlockWeightedLeastSquaresEstimator}
import nodes.stats.{StandardScaler, Sampler}
import nodes.util.{Identity, Cacher, ClassLabelIndicatorsFromIntLabels, TopKClassifier, MaxClassifier} 

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD
import pipelines.Logging
import scopt.OptionParser
import workflow.Pipeline
import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.commons.math3.random.MersenneTwister
import utils.{Image, MatrixUtils, Stats, ImageMetadata, LabeledImage, RowMajorArrayVectorizedImage, ChannelMajorArrayVectorizedImage}


import scala.reflect.BeanProperty
import org.yaml.snakeyaml.Yaml
import org.yaml.snakeyaml.constructor.Constructor

import java.io.{File, BufferedWriter, FileWriter}

object CKM extends Serializable with Logging {
  val appName = "CKM"

  def run(sc: SparkContext, conf: CKMConf) {
    val data: Dataset = loadData(sc, conf.dataset)

    val (xDim, yDim, numChannels) = getInfo(data)
    var currX = xDim
    var currY = yDim

    var convKernel: Pipeline[Image, Image] = new Identity()
    var numInputFeatures = numChannels

      implicit val randBasis: RandBasis = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(conf.seed)))
      val gaussian = new Gaussian(0, 1)
      val uniform = new Uniform(0, 1)

    for (i <- 0 until conf.layers) {

      var numOutputFeatures = conf.filters(i)
      val patchSize = math.pow(conf.patch_sizes(i), 2).toInt
      val W = DenseMatrix.rand(numOutputFeatures, numInputFeatures*patchSize, gaussian) :* conf.bandwidth(i)
      val b = DenseVector.rand(numOutputFeatures, uniform) :* (2*math.Pi)
      val ccap = new CCaP(W, b, currX, currY, numInputFeatures, 2, 2)
      convKernel = convKernel andThen ccap

      currX = ccap.outX
      currY = ccap.outY

      numInputFeatures = numOutputFeatures
    }

    val meta = data.train.take(1)(0).image.metadata
    val featurizer = ImageExtractor andThen ImageVectorizer  andThen Transformer[DenseVector[Double], Image](x => ChannelMajorArrayVectorizedImage(x.toArray, meta)) andThen  convKernel andThen ImageVectorizer andThen new Cacher[DenseVector[Double]]

    val XTrain = featurizer(data.train)
    val count = XTrain.count()
    val XTest = featurizer(data.test)
    XTest.count()
    val labelVectorizer = ClassLabelIndicatorsFromIntLabels(conf.numClasses)

    val yTrain = labelVectorizer(LabelExtractor(data.train))
    val yTest = labelVectorizer(LabelExtractor(data.test)).map(convert(_, Int).toArray)
    val numFeatures = XTrain.take(1)(0).size
    println(s"numFeatures: ${numFeatures}, count: ${count}")
    println(data.train.take(1)(0).label)
    val model = new BlockWeightedLeastSquaresEstimator(numFeatures, 1, conf.reg, 0.5).fit(XTrain, yTrain)
    val clf = model andThen MaxClassifier

    val yTrainPred = clf.apply(XTrain)
    val yTestPred =  clf.apply(XTest)

    val trainEval = MulticlassClassifierEvaluator(yTrainPred, LabelExtractor(data.train), 10)
    val testEval = MulticlassClassifierEvaluator(yTestPred, LabelExtractor(data.test), 10)
    println(s"total training accuracy ${1 - trainEval.totalError}")
    println(s"total testing accuracy ${1 - testEval.totalError}")

    val out_train = new BufferedWriter(new FileWriter("/tmp/ckm_train_results"))
    val out_test = new BufferedWriter(new FileWriter("/tmp/ckm_test_results"))

    val trainPredictions = model(XTrain)
    trainPredictions.zip(LabelExtractor(data.train)).map {
        case (weights, label) => s"$label," + weights.toArray.mkString(",")
      }.collect().foreach{x =>
        out_train.write(x)
        out_train.write("\n")
      }
      out_train.close()

    val testPredictions = model(XTest)
    testPredictions.zip(LabelExtractor(data.test)).map {
        case (weights, label) => s"$label," + weights.toArray.mkString(",")
      }.collect().foreach{x =>
        out_test.write(x)
        out_test.write("\n")
      }
      out_test.close()
  }

  def loadData(sc: SparkContext, dataset: String):Dataset = {
    val (train, test) =
    if (dataset == "cifar") {
      val train = CifarLoader(sc, "/home/eecs/vaishaal/ckm/mldata/cifar/cifar_train.bin")
      val test = CifarLoader(sc, "/home/eecs/vaishaal/ckm/mldata/cifar/cifar_test.bin").cache
      (train, test)
    } else if (dataset == "mnist") {
      val train = MnistLoader(sc, "/home/eecs/vaishaal/ckm/mldata/mnist", 10, "train").cache
      val test = MnistLoader(sc, "/home/eecs/vaishaal/ckm/mldata/mnist", 10, "test").cache
      (train, test)
    } else {
      val train = SmallMnistLoader(sc, "/home/eecs/vaishaal/ckm/mldata/mnist_small", 10, "train").cache
      val test = SmallMnistLoader(sc, "/home/eecs/vaishaal/ckm/mldata/mnist_small", 10, "test").cache
      (train, test)
    }
    return new Dataset(train, test)
  }

  def getInfo(data: Dataset): (Int, Int, Int) = {
    val image = data.train.take(1)(0).image
    (image.metadata.xDim, image.metadata.yDim, image.metadata.numChannels)
  }

  class CKMConf {
    @BeanProperty var  dataset: String = "mnist_small"
    @BeanProperty var  expid: String = "mnist_small_simple"
    @BeanProperty var  mode: String = "scala"
    @BeanProperty var  seed: Int = 0
    @BeanProperty var  layers: Int = 1
    @BeanProperty var  filters: Array[Int] = Array(1)
    @BeanProperty var  bandwidth : Array[Double] = Array(1.8)
    @BeanProperty var  patch_sizes: Array[Int] = Array(5)
    @BeanProperty var  loss: String = "softmax"
    @BeanProperty var  reg: Double = 0.001
    @BeanProperty var  numClasses: Int = 10
  }


  case class Dataset(
    val train: RDD[LabeledImage],
    val test: RDD[LabeledImage])


  /**
   * The actual driver receives its configuration parameters from spark-submit usually.
   * @param args
   */
  def main(args: Array[String]) = {

    if (args.size < 1) {
      println("Incorrect number of arguments...Exiting now.")
    } else {
      val configfile = scala.io.Source.fromFile(args(0))
      val configtext = try configfile.mkString finally configfile.close()
      println(configtext)
      val yaml = new Yaml(new Constructor(classOf[CKMConf]))
      val appConfig = yaml.load(configtext).asInstanceOf[CKMConf]

      val appName = s"CKM"
      val conf = new SparkConf().setAppName(appName)
      Logger.getLogger("org").setLevel(Level.WARN)
      Logger.getLogger("akka").setLevel(Level.WARN)
      conf.setIfMissing("spark.master", "local[16]")
      conf.set("spark.driver.maxResultSize", "0")
      val sc = new SparkContext(conf)
      run(sc, appConfig)
      sc.stop()
    }
  }
}
