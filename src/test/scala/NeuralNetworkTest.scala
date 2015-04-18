import breeze.numerics.sigmoid
import nn.DistributedNeuralNetwork
import nn.GenericNeuralNetwork
import org.scalatest.{FlatSpec, MustMatchers}
import breeze.linalg.{DenseMatrix, DenseVector}
import org.apache.spark.{SparkConf, SparkContext}
import nn.Helper._

class NeuralNetworkTest extends FlatSpec with MustMatchers {

  def sample = new {
    val w = Seq(
      DenseMatrix(
        (0.1, 0.3, 0.5),
        (0.2, 0.4, 0.6)
      ),
      DenseMatrix(
        (0.2, 0.4)
      )
    )

    val b = Seq(DenseVector(-0.1, -0.2), DenseVector(0.2))
    val input = DenseVector(3.0, 2.0, 1.0)
    val label = DenseVector(0.03)
  }

  def sc = {
    val conf = new SparkConf().setAppName("NN").setMaster("local[4]")
    new SparkContext(conf)
  }

  def testData(size: Int): MultiData = {
    val arr = new Array[SingleData](size)
    mnistTestData.copyToArray(arr, 0, size)
    collection.immutable.Seq[SingleData](arr:_*).par
  }

  def testLabel(size: Int): MultiData = {
    val arr = new Array[SingleData](size)
    mnistTestLabel.copyToArray(arr, 0, size)
    collection.immutable.Seq[SingleData](arr:_*).par
  }

  def trainData(size: Int): MultiData = {
    val arr = new Array[SingleData](size)
    mnistTrainData.copyToArray(arr, 0, size)
    collection.immutable.Seq[SingleData](arr:_*).par
  }

  def trainLabel(size: Int): MultiData = {
    val arr = new Array[SingleData](size)
    mnistTrainLabel.copyToArray(arr, 0, size)
    collection.immutable.Seq[SingleData](arr:_*).par
  }

  "Neural Network" must "able to be constructed by number of neurons" in {
    val sizes = Seq(3, 2, 3)
    GenericNeuralNetwork(sizes)
  }

  it must "able to compute activation output correctly for vector input" in {
    val nn = new GenericNeuralNetwork(sample.w, sample.b)
    val a = nn.activations(sample.input).last

    val s0 = sample.input     //activated input
    val wi1 = sample.w.head * s0 + sample.b.head  //compute weighted input of 2nd layer
    val s1 = sigmoid(wi1)       //compute activation of 2nd layer
    val wi2 = sample.w(1) * s1 + sample.b(1)  //weighted input of 3rd layer
    val s2 = sigmoid(wi2)       //final output

    assert(s2 == a)
  }

  it must "able to compute weighted input correctly" in {
    val nn = new GenericNeuralNetwork(sample.w, sample.b)
    val wi = nn.weightedInputs(sample.input).head

    val a = DenseVector(0.1, 0.3, 0.5).t * DenseVector(3.0, 2.0, 1.0) - 0.1
    val b = DenseVector(0.2, 0.4, 0.6).t * DenseVector(3.0, 2.0, 1.0) - 0.2

    assert((wi(0)*1000).toInt == (a*1000).toInt)    //rough estimation up to 3 decimal places
    assert((wi(1)*1000).toInt == (b*1000).toInt)
  }

  it must "able to compute error of different layer " in {
    val nn = new GenericNeuralNetwork(sample.w, sample.b)
    val f = nn.allErr(sample.input, sample.label)

    assert(f(0).length == 2)
    assert(f(1).length == 1)
  }

  ignore must "able to update itself using training data" in {
    {
      val nn = new GenericNeuralNetwork(sample.w, sample.b)
      val initialCost = nn.cost(sample.input, sample.label)

      val updated = nn.update(sample.input, sample.label, 0.5)
      val updatedCost = updated.cost(sample.input, sample.label)

      val diff = initialCost - updatedCost

      diff.foreach(x => assert(x > 0, "cost should decrease"))
    }

    {
      val nn = GenericNeuralNetwork(Seq(3,2,1))   //random
      val initialCost = nn.cost(sample.input, sample.label)

      val updated = nn.update(sample.input, sample.label, 0.5)
      val updatedCost = updated.cost(sample.input, sample.label)

      val diff = initialCost - updatedCost

      diff.foreach(x => assert(x > 0, "cost should decrease"))
    }
  }

  ignore must "converge after a batch update" in {
    val trainDataBatch = trainData(200)
    val trainLabelBatch = trainLabel(200)

    val nn = GenericNeuralNetwork(Seq(784, 30, 10))

    var trained = nn.batchUpdate(0.01)(trainDataBatch, trainLabelBatch)

    for (i <- 0 to 100){
      trained = trained.batchUpdate(0.01)(trainDataBatch, trainLabelBatch)
    }

    val oriCost = nn.cost(trainDataBatch, trainLabelBatch)
    val newCost = trained.cost(trainDataBatch, trainLabelBatch)

    val diff = oriCost - newCost
    diff.foreach(x => assert(x > 0, "cost should decrease"))

    val hit = trained.evaluate(trainDataBatch, trainLabelBatch.map(x => findMaxIndex(x)))
    print(hit)
  }

  ignore must "be able to be trained by thousands of record with many features" in {
    // get data and stored in iterator of Dense vector
    // the training should produce a new NeuralNetwork with adjusted weight and bias

    val trainData = mnistTrainData
    val trainLabel = mnistTrainLabel

    val nn = GenericNeuralNetwork(Seq(784, 30, 10))

    val trainedNN = nn.SGDByIterator(trainData, trainLabel, 4, 0.01, 10)

    val hit = trainedNN.evaluate(testData(500), testLabel(500).map(x => x(0).toInt))

    println(hit)
    assert(hit > 0.5)
  }

  ignore must "be able to be trained by spark core" in {
    val trainData = mnistTrainData
    val trainLabel = mnistTrainLabel

    val nn = DistributedNeuralNetwork(Seq(784, 60, 10))

    val trainedNN = nn.distributedSGDByIterator(trainData, trainLabel, 4, 0.01, 100, sc)

    val hit = trainedNN.evaluate(testData(500), testLabel(500).map(x => x(0).toInt))
    println(hit)
    assert(hit > 0.5)

  }
}
