package pipelines

import breeze.stats.distributions.Rand
import breeze.linalg._
import breeze.numerics._
import evaluation.MulticlassClassifierEvaluator
import loaders.{CifarLoader, MnistLoader, SmallMnistLoader}
import nodes.images._
import nodes.learning.{BlockLeastSquaresEstimator, ZCAWhitener, ZCAWhitenerEstimator}
import nodes.stats.{StandardScaler, Sampler}
import nodes.util.{Identity, Cacher, ClassLabelIndicatorsFromIntLabels, TopKClassifier}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD
import pipelines.Logging
import scopt.OptionParser
import workflow.Pipeline
import utils.{Image, MatrixUtils, Stats, ImageMetadata, LabeledImage}


import scala.reflect.BeanProperty
import org.yaml.snakeyaml.Yaml
import org.yaml.snakeyaml.constructor.Constructor

object CKM extends Serializable with Logging {
  val appName = "CKM"

  def run(sc: SparkContext, conf: CKMConf) {
    val data: Dataset = loadData(sc, conf.dataset)

    var (xDim, yDim, numChannels) = getInfo(data)

    var convKernel: Pipeline[Image, Image] = new Identity()
    var numInputFeatures = numChannels

    for (i <- 0 until conf.layers) {
      var numOutputFeatures = conf.filters(i)
      val patchSize = math.pow(conf.patch_sizes(i), 2).toInt
      val W = DenseMatrix.rand(numOutputFeatures, numInputFeatures*patchSize, Rand.gaussian) :* conf.bandwidth(i)
      println(s"Layer ${i} filter shape ${W.rows} ${W.cols}")
      val b = DenseVector.rand(numOutputFeatures, Rand.uniform) :* (2*math.Pi)
      convKernel = convKernel andThen new CCaP(W, b, xDim, yDim, numChannels, 2, 2)
      numInputFeatures = numOutputFeatures
    }

    val featurizer = ImageExtractor andThen convKernel andThen ImageVectorizer andThen new Cacher[DenseVector[Double]]

    val XTrain = featurizer(data.train)
    XTrain.count()
    val XTest = featurizer(data.test)
    XTest.count()
    val labelVectorizer = ClassLabelIndicatorsFromIntLabels(conf.numClasses)

    val yTrain = labelVectorizer(LabelExtractor(data.train))
    val yTest = labelVectorizer(LabelExtractor(data.test)).map(convert(_, Int).toArray)
    val numFeatures = XTrain.take(1)(0).size
    val clf = new BlockLeastSquaresEstimator(numFeatures, 1, conf.reg).fit(XTrain, yTrain) andThen TopKClassifier(1)

    val yTrainPred = clf.apply(XTrain)
    val yTestPred =  clf.apply(XTest)

    val trainAcc = 1 - Stats.getErrPercent(yTrainPred, yTrain.map(convert(_, Int).toArray),  yTrain.count())
    val testAcc = 1 - Stats.getErrPercent(yTestPred, yTest.map(convert(_, Int).toArray),  yTest.count())

    println(s"Train Accuracy is ${trainAcc}, Test Accuracy is ${testAcc}")
  }

  def loadData(sc: SparkContext, dataset: String):Dataset = {
    val (train, test) =
    if (dataset == "cifar") {
      val train = CifarLoader(sc, "../mldata/cifar").cache
      val test = CifarLoader(sc, "../mldata/cifar").cache
      (train, test)
    } else if (dataset == "mnist") {
      val train = MnistLoader(sc, "/work/vaishaal/ckm/mldata/mnist", 10, "train").cache
      val test = MnistLoader(sc, "/work/vaishaal/ckm/mldata/mnist", 10, "test").cache
      (train, test)
    } else {
      val train = SmallMnistLoader(sc, "/work/vaishaal/ckm/mldata/mnist_small", 10, "train").cache
      val test = SmallMnistLoader(sc, "/work/vaishaal/ckm/mldata/mnist_small", 10, "test").cache
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
      conf.setIfMissing("spark.master", "local[16]")
      val sc = new SparkContext(conf)
      run(sc, appConfig)
      sc.stop()
    }
  }
}
