package pipelines

import java.io.{BufferedWriter, FileWriter}
import java.util.Date

import breeze.linalg._
import breeze.stats.distributions._
import breeze.stats.mean
import evaluation.MulticlassClassifierEvaluator
import loaders._
import nodes.images.SequenceCC
import nodes.learning._
import nodes.stats.{SeededCosineRandomFeatures, Sampler}
import nodes.util._
import org.apache.commons.math3.random.MersenneTwister
import org.apache.log4j.{Level, Logger}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import utils._
import workflow.Pipeline
import java.nio.file.{Files, Paths}
import breeze.stats.{mean, median}

// TODO: Alyssa , ChannelMajorArrayVectorizedSequence

import org.yaml.snakeyaml.Yaml
import org.yaml.snakeyaml.constructor.Constructor

import scala.reflect.{BeanProperty, ClassTag}

object SequenceCKM extends Serializable {
  val appName = "SequenceCKM"

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

  def samplePairwiseMedian(data: RDD[Sequence], patchSize: Int = 0): Double = {
    val baseFilters =
      if (patchSize == 0) {
        new Sampler(1000)(SequenceVectorizer(data))
      } else {
        val patchExtractor = new SequenceWindower(1, patchSize)
          .andThen(SequenceVectorizer.apply)
          .andThen(new Sampler(1000))
        patchExtractor(data)
      }

    val baseFilterMat = MatrixUtils.rowsToMatrix(baseFilters)

    pairwiseMedian(baseFilterMat)
  }


  def run(sc: SparkContext, conf: CKMConf) {

    val feature_id = conf.seed + "_" + conf.expid  + "_" + conf.layers + "_" + conf.patch_sizes.mkString("-") + "_" + "_" + conf.pool.mkString("-") + "_" + conf.poolStride.mkString("-") + conf.filters.mkString("-")
    val data: SequenceDataset = loadData(sc, conf.dataset)
    // load in features if they were already saved
    val featureLocationPrefix =
      if (conf.cluster) {
        "/home/eecs/akmorrow/compbio294/ckm/keystone_pipeline/compbio/FEATURE_OUTPUT"
      } else {
        "/Users/akmorrow/Documents/COMPBIO294/Project/DREAM_data/FEATURE_OUTPUT/"
      }
    val featureLocation_train = featureLocationPrefix + s"ckn_${feature_id}_train_features"
    val featureLocation_test = featureLocationPrefix + s"ckn_${feature_id}_test_features"

    // Instantiate variables dependent on feature loading
    var XTrain: RDD[DenseVector[Double]] = null
    var XTest: RDD[DenseVector[Double]] = null
    var yTrain: RDD[DenseVector[Double]] = null
    var yTest: RDD[DenseVector[Double]] = null

    var trainIds: RDD[Long] = data.train.zipWithUniqueId.map(x => x._2)
    var testIds: RDD[Long] = data.test.zipWithUniqueId.map(x => x._2)
    val blockSize = conf.blockSize

    if (!Files.exists(Paths.get(featureLocation_train))) {

      // Compute bandwidth
      val median = SequenceCKM.samplePairwiseMedian(data.train.map(_.sequence), conf.patch_sizes(0))
      val bandwidth = 1 / (2 * Math.pow(median, 2))

      var convKernel: Pipeline[Sequence, Sequence] = new Identity()
      implicit val randBasis: RandBasis = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(conf.seed)))
      val gaussian = new Gaussian(0, 1)
      val uniform = new Uniform(0, 1)
      var numOutputFeatures = 0


      val (xDim, numChannels) = getInfo(data)
      println(s"Info ${xDim}, ${numChannels}")
      var numInputFeatures = numChannels
      var currX = xDim

      val startLayer =
        if (conf.whiten) {
          // Whiten top level
          val patchExtractor = new SequenceWindower(1, conf.patch_sizes(0))
            .andThen(SequenceVectorizer.apply)
            .andThen(new Sampler(100000))
          val baseFilters = patchExtractor(data.train.map(_.sequence))
          val baseFilterMat = MatrixUtils.rowsToMatrix(baseFilters)
          val whitener = new ZCAWhitenerEstimator(conf.whitenerValue).fitSingle(baseFilterMat)
          val whitenedBase = whitener(baseFilterMat)

          val rows = whitener.whitener.rows
          val cols = whitener.whitener.cols
          println(s"Whitener Rows :${rows}, Cols: ${cols}")

          numOutputFeatures = conf.filters(0)
          val patchSize = math.pow(conf.patch_sizes(0), 2).toInt
          val seed = conf.seed
          val dimensions = conf.dimensions
          val ccap: SequenceCC = new SequenceCC(numInputFeatures * patchSize,
            numOutputFeatures,
            seed,
            bandwidth,
            currX,
            numInputFeatures,
            Some(whitener),
            conf.whitenerOffset,
            conf.pool(0),
            conf.insanity,
            conf.fastfood)
          if (conf.pool(0) > 1) {
            var pooler = new SequencePooler(conf.poolStride(0), conf.pool(0), identity, (x: DenseVector[Double]) => mean(x))
            convKernel = convKernel andThen ccap andThen pooler
          } else {
            convKernel = convKernel andThen ccap
          }
          currX = math.ceil(((currX - conf.patch_sizes(0) + 1) - conf.pool(0) / 2.0) / conf.poolStride(0)).toInt

          println(s"Layer 0 output, Width: ${currX}")
          numInputFeatures = numOutputFeatures
          1
        } else {
          0
        }

      for (i <- startLayer until conf.layers) {
        numOutputFeatures = conf.filters(i)
        // val patchSize = math.pow(conf.patch_sizes(i), 2).toInt
        val patchSize = conf.patch_sizes(i).toInt
        val seed = conf.seed + i
        val ccap = new SequenceCC(numInputFeatures * patchSize,
          numOutputFeatures,
          seed, bandwidth, currX, numInputFeatures, None, conf.whitenerOffset, conf.pool(i), conf.insanity, conf.fastfood)

        if (conf.pool(i) > 1) {
          var pooler = new SequencePooler(conf.poolStride(i), conf.pool(i), identity, (x: DenseVector[Double]) => mean(x))
          convKernel = convKernel andThen ccap andThen pooler
        } else {
          convKernel = convKernel andThen ccap
        }

        currX = math.ceil(((currX - conf.patch_sizes(i) + 1) - conf.pool(i) / 2.0) / conf.poolStride(i)).toInt
        println(s"Layer ${i} output, Width: ${currX}")
        numInputFeatures = numOutputFeatures
      }
      val outFeatures = currX * numOutputFeatures

      val meta = data.train.take(1)(0).sequence.metadata
      val featurizer1 = SequenceExtractor andThen convKernel
      val featurizer2 = SequenceVectorizer andThen new Cacher[DenseVector[Double]]

      println("OUT FEATURES " + outFeatures)
      var featurizer = featurizer1 andThen featurizer2
      if (conf.cosineSolver) {
        val randomFeatures = SeededCosineRandomFeatures(outFeatures, conf.cosineFeatures, conf.cosineGamma, 24) andThen new Cacher[DenseVector[Double]]
        featurizer = featurizer andThen randomFeatures
      }

      val dataLoadBegin = System.nanoTime
      data.train.count()
      data.test.count()
      val dataLoadTime = timeElapsed(dataLoadBegin)
      println(s"Loading data took ${dataLoadTime} secs")


      val convTrainBegin = System.nanoTime
      XTrain = featurizer(data.train)
      val count = XTrain.count()
      val convTrainTime = timeElapsed(convTrainBegin)
      println(s"Generating train features took ${convTrainTime} secs")

      val convTestBegin = System.nanoTime
      XTest = featurizer(data.test)
      XTest.count()
      val convTestTime = timeElapsed(convTestBegin)
      println(s"Generating test features took ${convTestTime} secs")

      val numFeatures = XTrain.take(1)(0).size
      println(s"numFeatures: ${numFeatures}, count: ${count}, blockSize: ${blockSize}")


      if (conf.saveFeatures) {
        if (!Files.exists(Paths.get(featureLocation_train))) {
          println(s"Saving Features, ${feature_id}")
          val saveTrain: RDD[SaveableVector] = XTrain.zip(LabelExtractor(data.train)).map(r => SaveableVector(r._1, r._2))
          val saveTest: RDD[SaveableVector] = XTest.zip(LabelExtractor(data.test)).map(r => SaveableVector(r._1, r._2))
          saveTrain.saveAsObjectFile(featureLocationPrefix + s"ckn_${feature_id}_train_features")
          saveTest.saveAsObjectFile(featureLocationPrefix + s"ckn_${feature_id}_test_features")
        } else {
          println("feature files already saved")
        }

      }
      yTrain = data.train.map(r => DenseVector(r.label))
      yTest = data.test.map(r => DenseVector(r.label))

      val avgEigenValue = (XTrain.map((x: DenseVector[Double]) => mean(x :* x)).sum() / (1.0 * count))
      println(s"Average EigenValue : ${avgEigenValue}")
    } else { // end loading features
        XTrain = sc.objectFile[SaveableVector](featureLocation_train).map(_.sequence)
        XTest =  sc.objectFile[SaveableVector](featureLocation_test).map(_.sequence)
        yTrain = sc.objectFile[SaveableVector](featureLocation_train).map(r => DenseVector(r.label))
        yTest = sc.objectFile[SaveableVector](featureLocation_test).map(r => DenseVector(r.label))
    }



    println(yTrain.count, yTest.count)
    yTrain.count()
    yTest.count()
    println(s"${yTrain} train points")

    if (conf.solve) {
      println(XTrain.count, yTrain.count)
      val x = XTrain.first
      val y = yTrain.first

      val model = new BlockWeightedLeastSquaresEstimator(blockSize, conf.numIters, conf.reg, conf.solverWeight).fit(XTrain, yTrain)
      val trainPredictions: RDD[DenseVector[Double]] = model.apply(XTrain).cache()
      val testPredictions: RDD[DenseVector[Double]] =  model.apply(XTest).cache()

      assert(trainPredictions.first.length == 1)

      // assert data sizes are the same
      println(trainPredictions.count,data.train.count)
      println(testPredictions.count,data.test.count)
      assert(trainPredictions.count == data.train.count)
      assert(testPredictions.count  == data.test.count)

      val trainEval = trainPredictions.zip(yTrain).map(r => Math.pow((r._1(0) - r._2(0)),2))
      val testEval = testPredictions.zip(yTest).map(r =>  Math.pow((r._1(0) - r._2(0)),2))

      val testSSE = testEval.sum
      val trainSSE = trainEval.sum

      println(s"total training accuracy SSE ${trainSSE}")
      println(s"total testing accuracy ${testSSE}")

      val out_train = new BufferedWriter(new FileWriter(s"ckm_train_results_Filters:_${conf.filters(0)}"))
      val out_test = new BufferedWriter(new FileWriter(s"ckm_test_results_Filters:_${conf.filters(0)}"))

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



  def loadData(sc: SparkContext, dataset: String): SequenceDataset = {

    val trainfilename = dataset + "train"
    val testfilename = dataset + "test"
    val filePath =
    if (sc.isLocal)
      "/Users/akmorrow/Documents/COMPBIO294/Project/DREAM_data"
    else "compbio"

    val (train, test) =
      if (dataset == "sample_DREAM5") {
        val train: RDD[LabeledSequence] = DREAM5Loader(sc, filePath, 10, "train", trainfilename).cache
        val test: RDD[LabeledSequence] = DREAM5Loader(sc, filePath, 10, "test", testfilename).cache
        (train, test)
      } else if (dataset == "small_DREAM5") {
        val train: RDD[LabeledSequence] = DREAM5Loader(sc, filePath, 10, "train", trainfilename, sample = true).cache
        val test: RDD[LabeledSequence] = DREAM5Loader(sc, filePath, 10, "test", testfilename, sample = true).cache
        (train, test)
      }else {
        throw new IllegalArgumentException("Unknown Dataset")
      }
    println(s"training sample: ${train.first.sequence}, ${train.first.label}" )
    println(s"test sample:  ${test.first.sequence}, ${test.first.label}" )

    train.checkpoint()
    test.checkpoint()
    return new SequenceDataset(train, test)

  }

  def timeElapsed(ns: Long) : Double = (System.nanoTime - ns).toDouble / 1e9

  def getInfo(data: SequenceDataset): (Int, Int) = {
    val s = data.train.take(1)(0).sequence
    (s.metadata.dim, s.metadata.numChannels)
  }

  class CKMConf {
    @BeanProperty var  dataset: String = "mnist_small"
    @BeanProperty var  expid: String = "mnist_small_simple"
    @BeanProperty var  mode: String = "scala"
    @BeanProperty var  seed: Int = 0
    @BeanProperty var  layers: Int = 1
    @BeanProperty var  dimensions: Int = 2
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
    @BeanProperty var  cluster: Boolean = false
  }

  case class SequenceDataset(
                      val train: RDD[LabeledSequence],
                      val test: RDD[LabeledSequence])


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
