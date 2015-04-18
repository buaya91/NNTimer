package nn

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics._

import scala.collection.parallel.immutable.ParSeq
import scala.io.Source


object Helper {
  type LayerWeights = DenseMatrix[Double] //w(j,k) refer to weights from k to j
  type LayerBias = DenseVector[Double]
  type LayerActivation = DenseVector[Double]

  type NetworkWeights = Seq[LayerWeights]
  type NetworkBias = Seq[LayerBias]
  type NetworkActivation = Seq[LayerActivation]

  type SingleData = DenseVector[Double]
  type MultiData = ParSeq[SingleData]

  def mnistTrainData: Iterator[SingleData] = {
    val path = "/home/qingwei/Documents/Dataset/training_data.csv"

    //split line and convert to double, then create dense vector using the array of double
    readFromCSV(path).map(line => {DenseVector(line.split(",").map(_.toDouble))})
  }

  def mnistTrainLabel: Iterator[SingleData] = {
    val path = "/home/qingwei/Documents/Dataset/training_label.csv"

    //split line and convert to double, then create dense vector using the array of double
    readFromCSV(path).map(line => {DenseVector(line.split(",").map(_.toDouble))})
  }

  def mnistTestData: Iterator[SingleData] = {
    val path = "/home/qingwei/Documents/Dataset/test_data.csv"

    //split line and convert to double, then create dense vector using the array of double
    readFromCSV(path).map(line => {DenseVector(line.split(",").map(_.toDouble))})
  }

  def mnistTestLabel: Iterator[SingleData] = {
    val path = "/home/qingwei/Documents/Dataset/test_label.csv"

    //split line and convert to double, then create dense vector using the array of double
    readFromCSV(path).map(line => {DenseVector(line.split(",").map(_.toDouble))})
  }

  def readFromCSV(path: String):Iterator[String] = {
    val src = Source.fromFile(path)
    src.getLines() // return line by line iterator
  }

  def sigmoidP(x: DenseVector[Double]): DenseVector[Double] = {
    sigmoid(x) :* (DenseVector.ones[Double](x.length).:-(sigmoid(x)))
  }

  def batching[A](input: ParSeq[A], batchSize: Int): Seq[ParSeq[A]] = {
    for (i <- 0 to input.length / batchSize)
      yield input.slice(i, i + batchSize)
  }

  def duplicateNIterator[A](oriItr: Iterator[A], i: Int): Seq[Iterator[A]] = {
    def duplicateIter(itr: Iterator[A], n: Int, acc: Seq[Iterator[A]]): Seq[Iterator[A]] = n match {
      case 0 => acc
      case _ => {
        val (itr1, itr2) = itr.duplicate
        duplicateIter(itr1, n - 1, acc :+ itr2)
      }
    }
    duplicateIter(oriItr, i, Seq())
  }

  def dataEqual(prediction: SingleData, actual: SingleData): Boolean = {
    val predictionMax = prediction.data.max
    val actualMax = actual.data.max

    val pMaxIndex = prediction.data.indexOf(predictionMax)
    val aMaxIndex = actual.data.indexOf(actualMax)

    if (pMaxIndex == aMaxIndex)
      true
    else
      false
  }

  def findMaxIndex(d: SingleData): Int = {
    val maxVal = d.data.max
    d.data.indexOf(maxVal)
  }
}
