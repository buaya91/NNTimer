package nn

import Helper._
import breeze.linalg.{DenseMatrix, DenseVector}
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast


case class DistributedNeuralNetwork(override val weights: NetworkWeights,
                                    override val bias: NetworkBias) extends GenericNeuralNetwork(weights, bias) with Serializable {


  def distributedSGDByIterator(input: Iterator[SingleData],
                               label: Iterator[SingleData],
                               batchSize: Int,
                               eta: Double,
                               epoch: Int,
                               sc: SparkContext): GenericNeuralNetwork = {

    //curried function: update network using multiple batches distributely
    val distributedMultipleBatchUpdate = (n: GenericNeuralNetwork,
                               i: MultiData,
                               l: MultiData) => {
      val d = new DistributedNeuralNetwork(n.weights, n.bias)   //change to DN
      d.distributedSGD(eta, epoch, batchSize, sc)(i, l)
    }

    // for every 2k record, perform the update
    processIteratorByChunks(input, label, 2000, distributedMultipleBatchUpdate)
  }

  //split data into batches and process them parallelly, each batch produce one result, which is
  //reduced to one network
  def distributedSGD(eta: Double, epoch: Int, batchSize: Int, sc: SparkContext)
                    (input: MultiData, label: MultiData): GenericNeuralNetwork = {
    def mergeNeuralNetwork(network1: GenericNeuralNetwork, network2: GenericNeuralNetwork): GenericNeuralNetwork = {
      val m1 = (network1.weights, network1.bias)
      val m2 = (network2.weights, network2.bias)
      val (combinedW, combinedB) = combine(m1, m2)
      val avgW = combinedW.map(w => w :/ 2.0)
      val avgB = combinedB.map(b => b :/ 2.0)

      new GenericNeuralNetwork(avgW, avgB)
    }

    val miniBatches = (batching[SingleData](input, batchSize), batching[SingleData](label, batchSize)).
      zipped.toIndexedSeq

    val miniBatchesRDD = sc.parallelize(miniBatches)

    var trainedNN: Broadcast[GenericNeuralNetwork] = sc.broadcast(this)

    for (i <- 0 to epoch) {
      trainedNN = sc.broadcast(miniBatchesRDD.
        map { case (inputBatch, labelBatch) => trainedNN.value.batchUpdate(eta)(inputBatch, labelBatch) }.
        reduce(mergeNeuralNetwork))
    }

    trainedNN.value
  }
}

object DistributedNeuralNetwork {

  def apply(sizes: Seq[Int]) = {
    val w: (Seq[LayerWeights], Seq[LayerBias]) = {
      for (i <- Range(1, sizes.length))
        yield (DenseMatrix.rand[Double](sizes(i), sizes(i - 1)), DenseVector.rand[Double](sizes(i)))
    }.unzip(pair => (pair._1, pair._2))

    val ws = w._1.map(l => l :- 0.5)
    val bs = w._2.map(l => l :- 0.5)

    new DistributedNeuralNetwork(ws, bs)
  }

}
