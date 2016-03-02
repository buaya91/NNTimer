# NNTimer
application to collect time used for neural network training using MNIST data

This project is a final year research project which aims to study how to create a distributed & parallel neural network, and the impact on both speed and accuracy of Neural Network.

The neural network is implemented in Scala so that it could interface easily with Spark cluster, which is written in Scala as well.

I achieved parallelism by processing data independently, and merge the final results.

The findings include
1. Training of Neural Network get slower when there are more nodes in cluster, which implies overhead of network communication outweight gains of computing power
2. Accurancy are not affected significantly by parallelism.

