package pipelines

import breeze.stats.distributions._
import breeze.linalg._
import breeze.numerics._
import breeze.stats.{mean, median}
import evaluation.MulticlassClassifierEvaluator
import loaders.{CifarLoader, CifarLoader2, MnistLoader, SmallMnistLoader}
import nodes.images._
import workflow.Transformer
import nodes.learning._
import nodes.stats.{StandardScaler, Sampler, SeededCosineRandomFeatures, BroadcastCosineRandomFeatures, CosineRandomFeatures}
import nodes.util.{Identity, Cacher, ClassLabelIndicatorsFromIntLabels, TopKClassifier, MaxClassifier, VectorCombiner}

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

object CKM2 extends Serializable with Logging {
  val appName = "CKM2"

  def pairwiseMedian(data: DenseMatrix[Double]): Double = {
      val x = data(0 until data.rows/2, *)
      val y = data(data.rows/2 to -1, *)
      val x_norm = norm(x :+ 1e-13, Axis._1)
      val y_norm = norm(y :+ 1e-13, Axis._1)
      val x_normalized = x / x_norm
      val y_normalized = y / y_norm
      val diff = (x_normalized - y_normalized)
      val diff_norm = norm(diff, Axis._1)
      val diff_norm_median = median(diff_norm)
      diff_norm_median
  }

  def samplePairwiseMedian(data: RDD[Image], patchSize: Int = 0): Double = {
      val baseFilters =
      if (patchSize == 0) {
        new Sampler(1000)(ImageVectorizer(data))
      } else {
        val patchExtractor = new Windower(1, patchSize)
                                              .andThen(ImageVectorizer.apply)
                                              .andThen(new Sampler(1000))
        patchExtractor(data)
      }

      val baseFilterMat = MatrixUtils.rowsToMatrix(baseFilters)
      pairwiseMedian(baseFilterMat)
  }

  def run(sc: SparkContext, conf: CKM2Conf) {
    val data: Dataset = loadData(sc, conf.dataset)
    val feature_id = conf.seed + "_" + conf.expid  + "_" + conf.layers + "_" + conf.patch_sizes.mkString("-") + "_" +
      conf.bandwidth.mkString("-") + "_" + conf.pool.mkString("-") + "_" + conf.poolStride.mkString("-") + conf.filters.mkString("-")


    val (xDim, yDim, numChannels) = getInfo(data)
    var currX = xDim
    var currY = yDim

    var convKernel: Pipeline[Image, Image] = new Identity()
    var numInputFeatures = numChannels
    println("RUNNING CKM TWOOOOO")
    implicit val randBasis: RandBasis = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(conf.seed)))
    val gaussian = new Gaussian(0, 1)
    val uniform = new Uniform(0, 1)
    var numOutputFeatures = 0
    val startLayer =
    if (conf.whiten) {
      // Whiten top level
      val patchExtractor = new Windower(1, conf.patch_sizes(0))
                                              .andThen(ImageVectorizer.apply)
                                              .andThen(new Sampler(100000))
      val baseFilters = patchExtractor(data.train.map(_.image))
      val baseFilterMat = MatrixUtils.rowsToMatrix(baseFilters)
      val whitener = new ZCAWhitenerEstimator(conf.whitenerValue).fitSingle(baseFilterMat)
      val whitenedBase = whitener(baseFilterMat)

      val rows = whitener.whitener.rows
      val cols = whitener.whitener.cols
      println(s"Whitener Rows :${rows}, Cols: ${cols}")

      numOutputFeatures = conf.filters(0)
      val patchSize = math.pow(conf.patch_sizes(0), 2).toInt
      val seed = conf.seed
      val ccap = new CC(numInputFeatures*patchSize, numOutputFeatures,  seed, conf.bandwidth(0), currX, currY, numInputFeatures, Some(whitener), conf.whitenerOffset, conf.pool(0), conf.insanity)
      if (conf.pool(0) > 1) {
        var pooler =  new Pooler(conf.poolStride(0), conf.pool(0), identity, (x:DenseVector[Double]) => mean(x))
        convKernel = convKernel andThen ccap andThen pooler
      } else {
        convKernel = convKernel andThen ccap
      }
      currX = math.ceil(((currX  - conf.patch_sizes(0) + 1) - conf.pool(0)/2)/conf.poolStride(0)).toInt
      currY = math.ceil(((currY  - conf.patch_sizes(0) + 1) - conf.pool(0)/2)/conf.poolStride(0)).toInt

      println(s"Layer 0 output, Width: ${currX}, Height: ${currY}")
      numInputFeatures = numOutputFeatures
      1
    } else {
      0
    }

    for (i <- startLayer until conf.layers) {
      numOutputFeatures = conf.filters(i)
      val patchSize = math.pow(conf.patch_sizes(i), 2).toInt
      val seed = conf.seed + i
      val ccap = new CC(numInputFeatures*patchSize, numOutputFeatures,  seed, conf.bandwidth(i), currX, currY, numInputFeatures, None, conf.whitenerOffset, conf.pool(i), conf.insanity)

      if (conf.pool(i) > 1) {
        var pooler =  new Pooler(conf.poolStride(i), conf.pool(i), identity, (x:DenseVector[Double]) => mean(x))
        convKernel = convKernel andThen ccap andThen pooler
      } else {
        convKernel = convKernel andThen ccap
      }
      // (8 - 3 + 1)
      currX = math.ceil(((currX  - conf.patch_sizes(i) + 1) - conf.pool(i)/2.0)/conf.poolStride(i)).toInt
      currY = math.ceil(((currY  - conf.patch_sizes(i) + 1) - conf.pool(i)/2.0)/conf.poolStride(i)).toInt
      println(s"Layer ${i} output, Width: ${currX}, Height: ${currY}")
      numInputFeatures = numOutputFeatures
    }
    val outFeatures = currX * currY * numOutputFeatures

    val meta = data.train.take(1)(0).image.metadata
    val first_pixel = data.train.take(1)(0).image.get(15,7,0)
    println(s"First Pixel: ${first_pixel}")
    val featurizer1 = ImageExtractor  andThen convKernel
    val featurizer2 = ImageVectorizer andThen new Cacher[DenseVector[Double]]


    //println(s"conv kernel output median: ${samplePairwiseMedian(featurizer1(data.train), conf.patch_sizes(1))}")

    var featurizer = featurizer1 andThen featurizer2
    if (conf.cosineSolver) {
      val randomFeatures = SeededCosineRandomFeatures(outFeatures, conf.cosineFeatures,  conf.cosineGamma, 24) andThen new Cacher[DenseVector[Double]]
      featurizer = featurizer andThen randomFeatures
    }

    var XTrain = featurizer(data.train)
    val count = XTrain.count()
    var XTest = featurizer(data.test)
    XTest.count()

    val numFeatures = XTrain.take(1)(0).size
    val blockSize = conf.blockSize
    println(s"numFeatures: ${numFeatures}, count: ${count}, blockSize: ${blockSize}")

    val labelVectorizer = ClassLabelIndicatorsFromIntLabels(conf.numClasses)


    val yTrain = labelVectorizer(LabelExtractor(data.train))
    val yTest = labelVectorizer(LabelExtractor(data.test)).map(convert(_, Int).toArray)
    yTrain.count()
    yTest.count()

    if (conf.saveFeatures) {
      println("Saving Features")
      XTrain.map(_.toArray.mkString(",")).zip(LabelExtractor(data.train)).saveAsTextFile(s"/ckn_${feature_id}_train_features")
      XTest.map(_.toArray.mkString(",")).zip(LabelExtractor(data.test)).saveAsTextFile(s"/ckn_${feature_id}_test_features")
    }

    val avgEigenValue = (XTrain.map((x:DenseVector[Double]) => mean(x :*  x)).sum()/(1.0*count))
    println(s"Average EigenValue : ${avgEigenValue}")
    if (conf.solve) {
      val model =
      if (conf.solver ==  "kernel" ) {
      val kernelGen = new GaussianKernelGenerator(conf.kernelGamma)
       new KernelRidgeRegression(kernelGen, Array(conf.reg), conf.blockSize, conf.numIters, Some(895423832L)).fit(XTrain, yTrain) andThen Transformer[Array[DenseVector[Double]], DenseVector[Double]](_(0))

     } else {
      new BlockWeightedLeastSquaresEstimator(blockSize, conf.numIters, conf.reg, conf.solverWeight).fit(XTrain, yTrain)
    }
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
  }

  def loadData(sc: SparkContext, dataset: String):Dataset = {
    val (train, test) =
    if (dataset == "cifar") {
      val train = CifarLoader2(sc, "/home/eecs/vaishaal/ckm/mldata/cifar/cifar_train.bin").cache
      val test = CifarLoader2(sc, "/home/eecs/vaishaal/ckm/mldata/cifar/cifar_test.bin").cache
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
    train.checkpoint()
    test.checkpoint()
    return new Dataset(train, test)
  }

  def getInfo(data: Dataset): (Int, Int, Int) = {
    val image = data.train.take(1)(0).image
    (image.metadata.xDim, image.metadata.yDim, image.metadata.numChannels)
  }

  class CKM2Conf {
    @BeanProperty var  dataset: String = "mnist_small"
    @BeanProperty var  expid: String = "mnist_small_simple"
    @BeanProperty var  mode: String = "scala"
    @BeanProperty var  seed: Int = 0
    @BeanProperty var  layers: Int = 1
    @BeanProperty var  filters: Array[Int] = Array(1)
    @BeanProperty var  bandwidth : Array[Double] = Array(1.8)
    @BeanProperty var  patch_sizes: Array[Int] = Array(5)
    @BeanProperty var  loss: String = "WeightedLeastSquares"
    @BeanProperty var  reg: Double = 0.001
    @BeanProperty var  numClasses: Int = 10
    @BeanProperty var  yarn: Boolean = true
    @BeanProperty var  solverWeight: Double = 0
    @BeanProperty var  cosineSolver: Boolean = false
    @BeanProperty var  cosineFeatures: Int = 40000
    @BeanProperty var  cosineGamma: Double = 1e-8
    @BeanProperty var  kernelGamma: Double = 5e-5
    @BeanProperty var  blockSize: Int = 4000
    @BeanProperty var  numBlocks: Int = 2
    @BeanProperty var  numIters: Int = 2
    @BeanProperty var  whiten: Boolean = false
    @BeanProperty var  whitenerValue: Double =  0.1
    @BeanProperty var  whitenerOffset: Double = 0.001
    @BeanProperty var  solve: Boolean = true
    @BeanProperty var  solver: String = "kernel"
    @BeanProperty var  insanity: Boolean = false
    @BeanProperty var  saveFeatures: Boolean = false
    @BeanProperty var  pool: Array[Int] = Array(2)
    @BeanProperty var  poolStride: Array[Int] = Array(2)
    @BeanProperty var  checkpointDir: String = "/tmp/spark-checkpoint"
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
      val yaml = new Yaml(new Constructor(classOf[CKM2Conf]))
      val appConfig = yaml.load(configtext).asInstanceOf[CKM2Conf]

      val appName = s"CKM2"
      val conf = new SparkConf().setAppName(appName)
      Logger.getLogger("org").setLevel(Level.WARN)
      Logger.getLogger("akka").setLevel(Level.WARN)
      conf.setIfMissing("spark.master", "local[16]")
      conf.set("spark.driver.maxResultSize", "0")
      conf.setAppName(appConfig.expid)
      val sc = new SparkContext(conf)
      sc.setCheckpointDir(appConfig.checkpointDir)
      run(sc, appConfig)
      sc.stop()
    }
  }
}
