package pipelines

import breeze.stats.distributions._
import breeze.linalg._
import breeze.numerics._
import breeze.stats.{mean, median}
import loaders.{CifarLoader, CifarLoader2, MnistLoader, SmallMnistLoader, ImageNetLoader}
import nodes.images._
import workflow.Transformer
import evaluation.MulticlassClassifierEvaluator

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

import scala.reflect.{BeanProperty, ClassTag}
import org.yaml.snakeyaml.Yaml
import org.yaml.snakeyaml.constructor.Constructor

import java.io.{File, BufferedWriter, FileWriter}

class LabelAugmenter[T: ClassTag](mult: Int) extends FunctionNode[RDD[T], RDD[T]] {
  def apply(in: RDD[T]) = in.flatMap(x => Seq.fill(mult)(x))
}

object CKM extends Serializable with Logging {
  val appName = "CKM"

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

  def run(sc: SparkContext, conf: CKMConf) {
    var data: Dataset = loadData(sc, conf.dataset)
    val feature_id = conf.seed + "_" + conf.expid  + "_" + conf.layers + "_" + conf.patch_sizes.mkString("-") + "_" +
      conf.bandwidth.mkString("-") + "_" + conf.pool.mkString("-") + "_" + conf.poolStride.mkString("-") + conf.filters.mkString("-")


    var convKernel: Pipeline[Image, Image] = new Identity()
    implicit val randBasis: RandBasis = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(conf.seed)))
    val gaussian = new Gaussian(0, 1)
    val uniform = new Uniform(0, 1)
    var numOutputFeatures = 0
    var trainIds = data.train.zipWithUniqueId.map(x => x._2.toInt)
    var testIds = data.test.zipWithUniqueId.map(x => x._2.toInt)

    val (xDim, yDim, numChannels) = getInfo(data)
    println(s"Info ${xDim}, ${yDim}, ${numChannels}")
    var numInputFeatures = numChannels
    var currX = xDim
    var currY = yDim


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
      val ccap = new CC(numInputFeatures*patchSize, numOutputFeatures,  seed, conf.bandwidth(0), currX, currY, numInputFeatures, Some(whitener), conf.whitenerOffset, conf.pool(0), conf.insanity, conf.fastfood)
      if (conf.pool(0) > 1) {
        var pooler =  new Pooler(conf.poolStride(0), conf.pool(0), identity, (x:DenseVector[Double]) => mean(x))
        convKernel = convKernel andThen ccap andThen pooler
      } else {
        convKernel = convKernel andThen ccap
      }
      currX = math.ceil(((currX  - conf.patch_sizes(0) + 1) - conf.pool(0)/2.0)/conf.poolStride(0)).toInt
      currY = math.ceil(((currY  - conf.patch_sizes(0) + 1) - conf.pool(0)/2.0)/conf.poolStride(0)).toInt

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
      val ccap = new CC(numInputFeatures*patchSize, numOutputFeatures,  seed, conf.bandwidth(i), currX, currY, numInputFeatures, None, conf.whitenerOffset, conf.pool(i), conf.insanity, conf.fastfood)

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
    val featurizer1 = ImageExtractor  andThen convKernel
    val featurizer2 = ImageVectorizer andThen new Cacher[DenseVector[Double]]

    println("OUT FEATURES " +  outFeatures)
    var featurizer = featurizer1 andThen featurizer2
    if (conf.cosineSolver) {
      val randomFeatures = SeededCosineRandomFeatures(outFeatures, conf.cosineFeatures,  conf.cosineGamma, 24) andThen new Cacher[DenseVector[Double]]
      featurizer = featurizer andThen randomFeatures
    }

    val dataLoadBegin = System.nanoTime
    data.train.count()
    data.test.count()
    val dataLoadTime = timeElapsed(dataLoadBegin)
    println(s"Loading data took ${dataLoadTime} secs")


    val convTrainBegin = System.nanoTime
    var XTrain = featurizer(data.train)
    val count = XTrain.count()
    val convTrainTime  = timeElapsed(convTrainBegin)
    println(s"Generating train features took ${convTrainTime} secs")

    val convTestBegin = System.nanoTime
    var XTest = featurizer(data.test)
    XTest.count()
    val convTestTime  = timeElapsed(convTestBegin)
    println(s"Generating test features took ${convTestTime} secs")

    val numFeatures = XTrain.take(1)(0).size
    val blockSize = conf.blockSize
    println(s"numFeatures: ${numFeatures}, count: ${count}, blockSize: ${blockSize}")

    val labelVectorizer = ClassLabelIndicatorsFromIntLabels(conf.numClasses)


    val yTrain = labelVectorizer(LabelExtractor(data.train))
    val yTest = labelVectorizer(LabelExtractor(data.test)).map(convert(_, Int).toArray)
    yTrain.count()
    yTest.count()
    println(s"${yTrain} train points")

    if (conf.saveFeatures) {
      println("Saving Features")
      XTrain.map(_.toArray.mkString(",")).zip(LabelExtractor(data.train)).saveAsTextFile(s"/ckn_${feature_id}_train_features")
      XTest.map(_.toArray.mkString(",")).zip(LabelExtractor(data.test)).saveAsTextFile(s"/ckn_${feature_id}_test_features")
    }

    val avgEigenValue = (XTrain.map((x:DenseVector[Double]) => mean(x :*  x)).sum()/(1.0*count))
    println(s"Average EigenValue : ${avgEigenValue}")
    if (conf.solve) {
        val model = new BlockWeightedLeastSquaresEstimator(blockSize, conf.numIters, conf.reg, conf.solverWeight).fit(XTrain, yTrain)
        val trainPredictions = model.apply(XTrain).cache()
        val testPredictions =  model.apply(XTest).cache()

        val yTrainPred = MaxClassifier.apply(trainPredictions)
        val yTestPred =  MaxClassifier.apply(testPredictions)

        val trainEval = MulticlassClassifierEvaluator(yTrainPred, LabelExtractor(data.train), conf.numClasses)
        val testEval = MulticlassClassifierEvaluator(yTestPred, LabelExtractor(data.test), conf.numClasses)

        if (conf.numClasses >= 5) {

          val testLabels = labelVectorizer(LabelExtractor(data.test))
          val top5Predicted = TopKClassifier(5)(testPredictions)
          val top1Actual = TopKClassifier(1)(testLabels)

           println("Top 5 test acc is " + (100 - Stats.getErrPercent(top5Predicted, top1Actual, testPredictions.count())) + "%")
        }

        println(s"total training accuracy ${1 - trainEval.totalError}")
        println(s"total testing accuracy ${1 - testEval.totalError}")

        val out_train = new BufferedWriter(new FileWriter("/tmp/ckm_train_results"))
        val out_test = new BufferedWriter(new FileWriter("/tmp/ckm_test_results"))

        trainPredictions.zip(LabelExtractor(data.train).zip(trainIds)).map {
            case (weights, (label, id)) => s"$id,$label," + weights.toArray.mkString(",")
          }.collect().foreach{x =>
            out_train.write(x)
            out_train.write("\n")
          }
          out_train.close()

        testPredictions.zip(LabelExtractor(data.test).zip(testIds)).map {
            case (weights, (label, id)) => s"$id,$label," + weights.toArray.mkString(",")
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
      } else if (dataset == "mnist_small") {
        val train = SmallMnistLoader(sc, "/home/eecs/vaishaal/ckm/mldata/mnist_small", 10, "train").cache
        val test = SmallMnistLoader(sc, "/home/eecs/vaishaal/ckm/mldata/mnist_small", 10, "test").cache
        (train, test)
      } else if (dataset == "imagenet") {
        val train = ImageNetLoader(sc, "/user/vaishaal/imagenet-train-brewed", 
          "/home/eecs/vaishaal/ckm/mldata/imagenet/imagenet-labels").cache
        val test = ImageNetLoader(sc, "/user/vaishaal/imagenet-validation-brewed", 
          "/home/eecs/vaishaal/ckm/mldata/imagenet/imagenet-labels").cache
        (train, test)
      } else if (dataset == "imagenet-small") {
        val train = ImageNetLoader(sc, "/user/vaishaal/imagenet-train-brewed-small", 
          "/home/eecs/vaishaal/ckm/mldata/imagenet-small/imagenet-small-labels").cache
        val test = ImageNetLoader(sc, "/user/vaishaal/imagenet-validation-brewed-small",
          "/home/eecs/vaishaal/ckm/mldata/imagenet-small/imagenet-small-labels").cache

        (train.repartition(200), test.repartition(200))
      } else {
        throw new IllegalArgumentException("Unknown Dataset")
      }
      train.checkpoint()
      test.checkpoint()
      return new Dataset(train, test)
  }
  def timeElapsed(ns: Long) : Double = (System.nanoTime - ns).toDouble / 1e9 

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
    @BeanProperty var  augment: Boolean = false
    @BeanProperty var  augmentPatchSize: Int = 24
    @BeanProperty var  augmentType: String = "random"
    @BeanProperty var  fastfood: Boolean = false
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
      val conf = new SparkConf().setAppName(appConfig.expid)
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
