package de.hpi.anlp.mlp

/**
 * Configuration of MLP. If params are out of range an exception is thrown
 * @param momentum  Momentum parameter used to adjust the value of the gradient of the weights
 *                  with previous value (smoothing)
 * @param learningRate   Learning rate ]0, 1] used in the computation of the gradient of the weights
 *                       during training
 * @param hidLayers  Sequence of number of neurons for the hidden layers
 * @param numEpochs  Number of epochs or iterations allowed to train the weights/model
 * @param eps  Convergence criteria used as exit condition of the convergence toward optimum 
 *             weights that minimize the sum of squared error		 
 * @param activation Activation function (sigmoid or tanh) that computes the output of hidden 
 *                   layers during forward propagation
 *
 */
case class MLPConfig(
                      momentum: Double,
                      learningRate: Double,
                      hidLayers: Array[Int],
                      numEpochs: Int,
                      eps: Double = 1e-17,
                      activation: Double => Double) {

  /**
   * Id of output layer
   */
  final def outLayerId: Int =
    if (hidLayers.isEmpty)
      1
    else
      hidLayers.size + 1

  /**
   * # hidden layers in network
   */
  def nHiddens =
    if (hidLayers.isEmpty)
      0
    else
      hidLayers.size
}
