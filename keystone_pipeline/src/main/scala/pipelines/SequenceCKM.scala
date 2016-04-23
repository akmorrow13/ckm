package pipelines

import java.io.{File, BufferedWriter, FileWriter}
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
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.{LabeledPoint, LinearRegressionWithSGD}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import utils._
import workflow.Pipeline
import java.nio.file.{Files, Paths}
import breeze.stats.{mean, median}
import org.apache.spark.mllib.stat.Statistics
import org.apache.hadoop.conf.Configuration._

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


  def run(sc: SparkContext, conf: CKMConf, fs: FileSystem) {

    val feature_id = conf.seed + "_" + conf.expid  + "_" + conf.layers + "_" + conf.patch_sizes.mkString("-") + "_" + "_" + conf.pool.mkString("-") + "_" + conf.poolStride.mkString("-") + conf.filters.mkString("-")
    val data: SequenceDataset = loadData(sc, conf.dataset, fs, conf.sample)
    // load in features if they were already saved
    val featureLocationPrefix = "FEATURE_OUTPUT/"

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

    if (!fs.exists(new Path(featureLocation_train))) {

      // Compute bandwidth
//      val median = SequenceCKM.samplePairwiseMedian(data.train.map(_.sequence), conf.patch_sizes(0))
//      val bandwidth = 1 / (2 * Math.pow(median, 2))
      val bandwidth = 1
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
      val dataLoadTime = timeElapsed(dataLoadBegin)
      println(s"Loading data took ${dataLoadTime} secs")


      val convTrainBegin = System.nanoTime
      XTrain = featurizer(data.train)
      val convTrainTime = timeElapsed(convTrainBegin)
      println(s"Generating train features took ${convTrainTime} secs")

      val convTestBegin = System.nanoTime
      XTest = featurizer(data.test)
      val convTestTime = timeElapsed(convTestBegin)
      println(s"Generating test features took ${convTestTime} secs")

      val numFeatures = XTrain.take(1)(0).size
      println(s"numFeatures: ${numFeatures}, blockSize: ${blockSize}")


      if (conf.saveFeatures) {
        if (!Files.exists(Paths.get(featureLocation_train))) {
          println(s"Saving Features, ${feature_id}")
          val saveTrain: RDD[SaveableVector] = XTrain.zip(LabelExtractor(data.train)).map(r => SaveableVector(r._1, r._2))
          val saveTest: RDD[SaveableVector] = XTest.zip(LabelExtractor(data.test)).map(r => SaveableVector(r._1, r._2))
          saveTrain.saveAsObjectFile(featureLocation_train)
          saveTest.saveAsObjectFile(featureLocation_test)
        } else {
          println("feature files already saved")
        }

      }

      yTrain = data.train.map(r => DenseVector(r.label))
      yTest = data.test.map(r => DenseVector(r.label))

//      val avgEigenValue = (XTrain.map((x: DenseVector[Double]) => mean(x :* x)).sum() / (1.0 * count))
//      println(s"Average EigenValue : ${avgEigenValue}")
    } else { // end loading features
        XTrain = sc.objectFile[SaveableVector](featureLocation_train).map(_.sequence)
        XTest =  sc.objectFile[SaveableVector](featureLocation_test).map(_.sequence)
        yTrain = sc.objectFile[SaveableVector](featureLocation_train).map(r => DenseVector(r.label))
        yTest = sc.objectFile[SaveableVector](featureLocation_test).map(r => DenseVector(r.label))
      println(XTrain.count, XTest.count, yTrain.count, yTest.count)
    }

    val trainLabels = yTrain.map(r => r(0).toInt)
    val testLabels = yTest.map(r => r(0).toInt)

    println(s"${yTrain.count} train points, ${yTest.count} test points")

    var trainPredictions: RDD[DenseVector[Double]] = null
    var testPredictions: RDD[DenseVector[Double]] =  null

    if (conf.solve) {
      if (conf.leastSquaresSolver) {
        val model = new BlockLeastSquaresEstimator(blockSize, conf.numIters, conf.reg).fit(XTrain, yTrain)
        trainPredictions = model.apply(XTrain).cache()
        testPredictions =  model.apply(XTest).cache()
        println(trainPredictions.first)
      } else {
        val rdd = XTrain.zip(trainLabels).map(r => LabeledPoint(r._2.toDouble, Vectors.dense(r._1.toArray)))
        rdd.cache
        val numIterations = 100
        val stepSize = 0.0001
        val model = LinearRegressionWithSGD.train(rdd, numIterations, stepSize)
        trainPredictions = XTrain.map(point => DenseVector(model.predict(Vectors.dense(point.toArray))))
        testPredictions = XTest.map(point => DenseVector(model.predict(Vectors.dense(point.toArray))))
      }

      if (conf.numClasses == 1) {
        // compute train error
        computeCorrelation(trainPredictions.map(r => r(0)), yTrain.map(r => r(0)))

        // compute test error
        computeCorrelation(testPredictions.map(r => r(0)), yTest.map(r => r(0)))
      } else {
        val yTrainPred = MaxClassifier.apply(trainPredictions)
        val yTestPred =  MaxClassifier.apply(testPredictions)
        val trainEval = MulticlassClassifierEvaluator(yTrainPred, trainLabels, conf.numClasses)
        val testEval = MulticlassClassifierEvaluator(yTestPred, testLabels, conf.numClasses)

        // comput AUROC
        // TODO: what to put in here
        computeAUROC(trainPredictions.map(r => r(0)), yTrain.map(r => r(0)))
        computeAUROC(testPredictions.map(r => r(0)), yTest.map(r => r(0)))
      }


      val out_train = s"ckm_train_results_Filters_${conf.filters(0)}_${conf.dataset}"
      val out_test = s"ckm_test_results_Filters_${conf.filters(0)}_${conf.dataset}"

      trainPredictions.zip(trainLabels).map {
        case (weights, label) => s"$label," + weights.toArray.mkString(",")
      }.saveAsTextFile(out_train)

      testPredictions.zip(testLabels).map {
        case (weights, label) => s"$label," + weights.toArray.mkString(",")
      }.saveAsTextFile(out_test)

    }
  }

  def computeAUROC(x: RDD[Double], y: RDD[Double]) = {
    val zipped = x.zip(y)
    zipped.collect.foreach(println)
    val metrics = new BinaryClassificationMetrics(zipped)
    val auROC = metrics.areaUnderROC()

    println("Area under ROC = " + auROC)
  }

  def computeCorrelation(x: RDD[Double], y: RDD[Double]): Unit = {

    val zipped = x.zip(y)
    zipped.take(10).foreach(println)

    // Calculate MSE
    val mse: Double = zipped.map{ case (v1, v2) =>
      val err = (v1 - v2)
      err * err
    }.mean()

    // Calculate raw pearson correlation coefficient
    val rawCorrelation: Double = Statistics.corr(x,y, "pearson")

    // Calculate log pearson correlation coefficient
    val filteredLog = zipped.filter(r => (r._1 > 0 && r._2 > 0)).map(r => (Math.log(r._1), Math.log(r._2)))
    println(s"Negative values: ${zipped.count}, ${filteredLog}")

    val logCorrelation: Double = Statistics.corr(filteredLog.map(_._1), filteredLog.map(_._2), "pearson")

    // Calculate Spearman correlation coefficient
    val spearmanCorrelation: Double = Statistics.corr(x,y, "spearman")

    println(s"Correlations: raw correlation: ${rawCorrelation} " +
      s"\n log correlation: ${logCorrelation} " +
      s"\n Spearman Correlation: ${spearmanCorrelation} " +
      s"\n MSE: ${mse}")
  }



  def loadData(sc: SparkContext, dataset: String, fs: FileSystem, sample: Boolean): SequenceDataset = {

    val trainfilename = dataset + "train"
    val testfilename = dataset + "test"

    val (train, test) =
      if (dataset == "sample_DREAM5") {
        val train: RDD[LabeledSequence] = DREAM5Loader(sc, fs, 10, "train", trainfilename, sample).cache
        val test: RDD[LabeledSequence] = DREAM5Loader(sc, fs, 10, "test", testfilename, sample).cache
        (train, test)
      } else if (dataset == "sample_CHIPSEQ") {
        val train: RDD[LabeledSequence] = ChipSeqLoader(sc, fs, 10, "train", trainfilename, sample).cache
        val test: RDD[LabeledSequence] = ChipSeqLoader(sc, fs, 10, "test", testfilename, sample).cache
        println(train.first)
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
    @BeanProperty var  leastSquaresSolver: Boolean = true
    @BeanProperty var  sample: Boolean = false
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
      val cluster = false
      val url =
        if (cluster)
          "hdfs://amp-bdg-master.amplab.net:8020/user/akmorrow/"
        else
          "/Users/akmorrow/Documents/COMPBIO294/Project/ckm/"
      val path: Path = new Path(url)
      val fs: FileSystem = path.getFileSystem(new Configuration())
      val homedir = fs.getHomeDirectory.toString
      println(homedir)
      val configfile= fs.open(new Path(args(0)))

      val yaml = new Yaml(new Constructor(classOf[CKMConf]))
      val appConfig = yaml.load(configfile).asInstanceOf[CKMConf]
      val conf = new SparkConf().setAppName(appConfig.expid)
      Logger.getLogger("org").setLevel(Level.WARN)
      Logger.getLogger("akka").setLevel(Level.WARN)
      conf.setIfMissing("spark.master", "local[16]")
      conf.set("spark.driver.maxResultSize", "0")
      conf.set("spark.kryoserializer.buffer.max", "2G")
      conf.setAppName(appConfig.expid)
      val sc = new SparkContext(conf)
      sc.setCheckpointDir(appConfig.checkpointDir)
      run(sc, appConfig, fs)
      sc.stop()
    }
  }

  def getListOfFiles(dir: String):List[File] = {
    val d = new File(dir)
    if (d.exists && d.isDirectory) {
      d.listFiles.filter(_.isFile).toList
    } else {
      List[File]()
    }
  }
}
