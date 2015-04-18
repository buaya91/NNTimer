package exp

import nn.DistributedNeuralNetwork
import nn.GenericNeuralNetwork
import nn.Helper._
import org.apache.spark.{SparkContext, SparkConf}


case class RunSpec(dataSize: Int, networkSize: Seq[Int], cpuNo: Int, epochs: Int, useSpark: Boolean){
  def paramsList = List(dataSize, networkSize, cpuNo, epochs, useSpark)
}

case class RunResult(spec: RunSpec, accuracy: Double, timeInS: Long){
  def paramsList = spec.paramsList ::: List(accuracy, timeInS)
}

object ExpRunner {

  def time[A](block: => A): (Long, A) = {
    val t0 = System.currentTimeMillis()
    val result = block    //call by name
    val t1 = System.currentTimeMillis()
    ((t1 - t0)/1000, result)    //return time in seconds
  }

  def run(spec: RunSpec, url: String): RunResult = {
    import spec._
    require(dataSize <= 60000, "only have 60000 data")    //code defensively
    require(cpuNo <= 84, "only have 84 cores")

    //prepare data by iterator
    val (trainD,trainL) = trainData(dataSize)
    val (testD, testL) = testData(500)

    //time the processing
    val (elapsedTime, trainedNN) =
      if (useSpark)
        time {
          val sc = {
            val conf = new SparkConf().
              setAppName("Distributed Neural Network").
              setMaster(url).
              set("spark.cores.max", cpuNo.toString)

            new SparkContext(conf)
          }
          val network = DistributedNeuralNetwork(networkSize).distributedSGDByIterator(trainD, trainL, 10, 0.01, epochs, sc)
          sc.stop()
          network
        }
      else
        time {
          GenericNeuralNetwork(networkSize).SGDByIterator(trainD, trainL, 10, 0.01, epochs)
        }

    //produce result
    val acc = trainedNN.evaluate(testD, testL.map(x => x(0).toInt))
    RunResult(spec, acc, elapsedTime)
  }

  def trainData(size: Int): (Iterator[SingleData], Iterator[SingleData]) = {
    val dIter = mnistTrainData.slice(0, size)
    val lIter = mnistTrainLabel.slice(0, size)
    (dIter, lIter)
  }

  def testData(size: Int): (MultiData, MultiData) = {
    val arr1 = new Array[SingleData](size)
    val arr2 = new Array[SingleData](size)

    mnistTestData.copyToArray(arr1, 0, size)
    mnistTestLabel.copyToArray(arr2, 0, size)

    val d = collection.immutable.Seq[SingleData](arr1:_*).par
    val l = collection.immutable.Seq[SingleData](arr2:_*).par

    (d, l)
  }

}