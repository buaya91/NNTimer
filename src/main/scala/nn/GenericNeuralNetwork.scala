package nn

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.sigmoid
import nn.Helper._

import scala.collection.parallel.immutable.ParSeq


class GenericNeuralNetwork(val weights: NetworkWeights, val bias: NetworkBias) extends Serializable{

  val noOfLayers = weights.length + 1

/*********************** Backpropagation related functions starts **********************/
  //activation output of a particular layer
  def activationOutput(layer: Int, input: SingleData): LayerActivation = layer match {
    case 0 => input     //layer 0 is input layer
    case _ => {
      val weightN = weights(layer - 1)    // both weights and bias is 1 size shorter than network,
      val biasN = bias(layer - 1)         // thus n-1 is current layer

      val aNMinus1 = activationOutput(layer - 1, input) //activation start from input layer, thus n-1 is previous layer
      sigmoid(weightN * aNMinus1 + biasN)
    }
  }

  def activations(input: SingleData): NetworkActivation = {
    def activationIter(n: Int): NetworkActivation = n match { //n range from 0 to no of layers -1
      case 0 => Seq(input)
      case _ => {
        val w = weights(n-1)    //2nd layer activation uses 1st weight
        val b = bias(n-1)

        val acts = activationIter(n-1)
        val act = sigmoid((w * acts.last) + b)

        acts :+ act
      }
    }

    require(input.length == this.weights.head.cols) //input should have same length with 1st layer

    activationIter(this.noOfLayers - 1)   //-1 becoz start from zero
  }

  //weighted input z of particular layer
  def weightedInput(layer: Int, input: SingleData): SingleData = layer match {
    case _ => {
      val a = activationOutput(layer-1, input)  //for activation, n-1 is previous layer
      weights(layer-1) * a + bias(layer-1)
    }
  }

  //weightedInputs is 1 size smaller than activations as 1st layer does not have weight
  def weightedInputs(input: SingleData): NetworkActivation = {
    def wiIter(n: Int): NetworkActivation = n match {  //n range from 0 to no of layers - 2
      case 0 => Seq((weights.head * activations(input)(0)) + bias(0))
      case _ => {
        val w = weights(n)
        val b = bias(n)
        val wis = wiIter(n-1)     //weighted inputs until n-1 layer
        val wi = (w * sigmoid(wis.last)) + b  //compute weighted input of layer n
        wis :+ wi
      }
    }
    require(input.length == this.weights.head.cols) //input should have same length with 1st layer
    wiIter(noOfLayers - 2)
  }

  //cost function of single data (DenseVector)
  def cost(trainingData: SingleData, trainingLabel: SingleData): SingleData = {
    val a = activations(trainingData).last
    val diff = a - trainingLabel
    (diff :* diff) :/ 2.0   //quadratic err = 1/2 * (y-a)^2
  }


  //eq 1 for backprop, compute err of output layer
  def outputError(input: SingleData, label: SingleData): LayerActivation = {
    val z = weightedInputs(input).last   //pick last of z
    val s = sigmoidP(z)
    val dCda = activations(input).last - label
    dCda :* s
  }

  //eq2 for backprop, a general formula to compute err of any layer, including output layer
  def layerNError(layer: Int, input: SingleData, label: SingleData): SingleData = layer match {
    case y if y == weights.size => outputError(input, label)    //output layer error

    //does not make sense to find error of 1st layer as it is input layer, without w and b
    case x if (x < 1) || (x > weights.size) => throw new NegativeArraySizeException("layer must be zero or more")
    case _ => {
      val wNPlusOne = weights(layer)    //weights(n+1)
      val oz = sigmoidP(weightedInput(layer, input)) //o'(z(n))
      (wNPlusOne.t * layerNError(layer+1, input, label)) :* oz
    }
  }

  def allErr(input: SingleData, label: SingleData): NetworkActivation = {
    def allErrIter(n: Int, acc: NetworkActivation): NetworkActivation = n match {
      case max if (max == this.noOfLayers - 2) => allErrIter(n-1,Seq(outputError(input, label)))
      case 0 => {
        val w = weights(n+1)
        val spv = sigmoidP(weightedInputs(input)(n))
        val err = (w.t * acc.last) :* spv
        err +: acc
      }
      case _ => {
        val w = weights(n+1)
        val spv = sigmoidP(weightedInputs(input)(n))
        val err = (w.t * acc.last) :* spv
        allErrIter(n-1, err +: acc)
      }
    }
    allErrIter(noOfLayers-2, Seq())
  }


  //eq3 for backprop, compute derivatives of cost to respect of bias, by layer
  def dCostdBias(layer: Int, input: SingleData, label: SingleData): LayerBias = {
    this.layerNError(layer, input, label)  //exactly same with error on layer n
  }


  //eq4 for backprop, compute derivatives of cost to respect of weights, by layer
  def dCostdWeight(layer: Int, input: SingleData, label: SingleData): LayerWeights = {
    val a = this.activationOutput(layer - 1, input)
    val err = this.layerNError(layer, input, label)
    err * a.t
  }

  def dCdWs(input: SingleData, label: SingleData): NetworkWeights = {
    val allActs = this.activations(input)
    val errs = allErr(input, label)

    for (i <- 0 to this.noOfLayers - 2) yield {
      errs(i) * allActs(i).t
    }
  }

  //NetworkWeights is a sequence of LayerWeights, same rule applies to NetworkBias
  def backProp(input: SingleData, label: SingleData): (NetworkWeights, NetworkBias) = {


    val gradB: Seq[LayerBias] = allErr(input, label)
    val gradW: Seq[LayerWeights] = dCdWs(input, label)
    (gradW, gradB)
  }

  /*********************** Backpropagation related functions ends **********************/



  /************************Network Update related functions starts**********************************/
  //return a new Network, input is single batch
  def batchUpdate(eta: Double)(batchInput: MultiData, batchLabel: MultiData): GenericNeuralNetwork = {
    val (deltaW, deltaB) = {
      val batches = (batchInput zip batchLabel)
      val gradients = batches.map(il => this.backProp(il._1, il._2))   //gradients for each data
      val avgGradient = gradients.reduce(combine)     //combine delta(s)
      avgGradient
    }

    //apply delta to corresponding weights using formula of W = W - (a * dW)
    val newWeights = (weights, deltaW).zipped.map { case (a, b) => a - (b :* eta) }
    val newBias = (bias, deltaB).zipped.map { case (a,b) => a - (b :* eta) }

    new GenericNeuralNetwork(newWeights, newBias)
  }


  def SGD(eta: Double, epoch: Int, batchSize: Int)(input: MultiData, label: MultiData): GenericNeuralNetwork = {

    //create batches of data and label separately
    val miniBatches = (batching[SingleData](input, batchSize), batching[SingleData](label, batchSize)).
      zipped.toIndexedSeq

    //mutable state : networkModel have to be updated iteratively
    var networkModel = this
    for (i <- 0 to epoch)
      miniBatches.foreach(batch => { networkModel = networkModel.batchUpdate(eta)(batch._1, batch._2) })

    networkModel
  }



  def processIteratorByChunks(input: Iterator[SingleData],
                              label: Iterator[SingleData],
                              chunkSize: Int,
                              op: (GenericNeuralNetwork, MultiData, MultiData) => GenericNeuralNetwork): GenericNeuralNetwork = {
    //buffer to hold chunks of data from iterators
    val inputBuffer = new Array[SingleData](chunkSize)
    val labelBuffer = new Array[SingleData](chunkSize)

    var networkModel: GenericNeuralNetwork = this   //mutable as we need to update it for every chunk

    while (input.hasNext && label.hasNext) {
      //consume chunks of data from iterator and store to buffer for later consumption
      input.copyToArray(inputBuffer, 0, chunkSize)
      label.copyToArray(labelBuffer, 0, chunkSize)

      //convert to immutableSeq as needed
      val inputImmu = collection.immutable.Seq[SingleData](inputBuffer.toSeq:_*)
      val labelImmu = collection.immutable.Seq[SingleData](labelBuffer.toSeq:_*)

      //operation to apply changes on Networkmodel
      networkModel = op(networkModel, inputImmu.par, labelImmu.par)
    }

    networkModel
  }

  def processBatches(input: MultiData,
                     label: MultiData,
                     batchSize: Int,
                     op: (GenericNeuralNetwork, MultiData, MultiData) => GenericNeuralNetwork): GenericNeuralNetwork = {
    val miniBatches = (batching[SingleData](input, batchSize), batching[SingleData](label, batchSize)).
      zipped.toIndexedSeq

    //mutable state : networkModel have to be updated iteratively
    var networkModel = this

    //side effect: change networkModel using batch
    miniBatches.foreach(batch => networkModel = op(networkModel, batch._1, batch._2))

    networkModel
  }

  def SGDByIterator(input: Iterator[SingleData],
                      label: Iterator[SingleData],
                      batchSize: Int,
                      eta: Double,
                      epoch: Int): GenericNeuralNetwork = {

    //curried function: process chunk of data in multiple batches and update network
    val multipleBatchUpdate = (n: GenericNeuralNetwork,
                               i: MultiData,
                               l: MultiData) => n.SGD(eta, epoch, batchSize)(i, l)

    processIteratorByChunks(input, label, 2000, multipleBatchUpdate)
  }


  /************************Network Update related functions starts**********************************/


  /******************** Helper functions starts ********************************/
  def combine(model1: (NetworkWeights, NetworkBias), model2: (NetworkWeights, NetworkBias)): (NetworkWeights, NetworkBias) = {
    val newW = (model1._1, model2._1).zipped.map { case (a,b) => (a + b) }
    val newB = (model1._2, model2._2).zipped.map { case (a,b) => (a + b) }
    (newW, newB)
  }

  def classify(data: SingleData): Int = {
    val output: LayerActivation = activations(data).last
    findMaxIndex(output)
  }

  def evaluate(data: MultiData, testLabel: ParSeq[Int]): Double = {
    //val predictions = data.map(x => this.classify(x))
    //val testResult = testLabel.map { x => x(0).toInt}

    val predictions = for (d <- data) yield {
      classify(d)
    }

    val correctCount: Int = (predictions zip testLabel).map(x => x._1 == x._2).count(tf => tf)
    correctCount.toDouble / testLabel.length.toDouble
  }

  /******************** Helper functions ends ********************************/


  /******************* Test purpose ***************************************/
  //an update which is not used but stays for verification purpose
  def update(input: SingleData, label: SingleData, eta: Double): GenericNeuralNetwork = {

    val (dCdWs, dCdBs) = backProp(input, label)

    val newWeights = (weights, dCdWs).zipped.map { case (a, b) => a - (b :* eta) }
    val newBias = (bias, dCdBs).zipped.map { case (a,b) => a - (b :* eta) }

    new GenericNeuralNetwork(newWeights, newBias)
  }

  def cost(input: MultiData, label: MultiData): SingleData = {
    val costVec = (input zip label).map { x => cost(x._1, x._2) }.reduce((a,b) => (a :+ b):/2.0)
    costVec
  }
}

object GenericNeuralNetwork {

  def apply(sizes: Seq[Int]) = {
    val w: (Seq[LayerWeights], Seq[LayerBias]) = {
      for (i <- Range(1, sizes.length))
        yield (DenseMatrix.rand[Double](sizes(i), sizes(i - 1)), DenseVector.rand[Double](sizes(i)))
    }.unzip(pair => (pair._1, pair._2))

    val ws = w._1.map(l => l :- 0.5)
    val bs = w._2.map(l => l :- 0.5)

    new GenericNeuralNetwork(ws, bs)
  }

}

