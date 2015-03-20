package de.hpi.anlp.mlp

import de.hpi.anlp.utils.ScalaMLTypes.MLPTask

import scala.util.Random

/**
 * Class that defines the connection between two consecutive (or sequential layers)
 * in a Multi-layer Perceptron. The connections is composed of all the synapses between
 * any neuron or variable of each layer.The Synapse is defined as a nested tuple(Double, Double)
 * tuple (weights, deltaWeights)
 */
class MLPConnection(
                     config: MLPConfig,
                     src: MLPLayer,
                     dst: MLPLayer)
                   (implicit mlpObjective: MLPTask) {

  private val BETA = 0.01

  /**
   * Synapse defined as a tuple of [weight, gradient(weights)]
   */
  type MLPSynapse = (Double, Double)

  /*
   * Initialize the matrix (Array of Array) of Synapse by generating
   * a random value between 0 and BETA
   */
  private[this] val synapses: Array[Array[MLPSynapse]] = Array.tabulate(dst.len)(n =>
    if (n > 0) 
      Array.fill(src.len)((Random.nextDouble * BETA, 0.0))
    else 
      Array.fill(src.len)((1.0, 0.0)))

  /**
   * Implement the forward propagation of input value. The output
   * value depends on the conversion selected for the output. If the output or destination
   * layer is a hidden layer, then the activation function is applied to the dot product of
   * weights and values. If the destination is the output layer, the output value is just
   * the dot product weights and values
   */
  def connectionForwardPropagation: Unit = {
    // Iterates over all the synapsed except the first or bian selement
    val _output = synapses.drop(1).map(x => {
      // Compute the dot product
      val sum = x.zip(src.output).foldLeft(0.0)((s, xy) => s + xy._1._1 * xy._2)

      // Applies the activation function if this is a hidden layer (not output)
      if (!isOutLayer) config.activation(sum) else sum
    })

    // Apply the objective function (SoftMax,...) to the output layer
    val out = if (isOutLayer) mlpObjective(_output) else _output
    out.copyToArray(dst.output, 1)
  }

  /**
   * Access the identifier for the source and destination layers
   */
  @inline
  final def getLayerIds: (Int, Int) = (src.id, dst.id)

  @inline
  final def getSynapses: Array[Array[MLPSynapse]] = synapses

  /**
   * Implement the back propagation of output error (target - output). The method uses
   * the derivative of the logistic function to compute the delta value for the output of
   * the source layer
   */
  def connectionBackpropagation: Unit =
    Range(1, src.len).foreach(i => {
      val err = Range(1, dst.len).foldLeft(0.0)((s, j) =>
        s + synapses(j)(i)._1 * dst.delta(j))

      // The delta value is computed as the derivative of the
      // output value adjusted for the back-propagated error, err
      src.delta(i) = src.output(i) * (1.0 - src.output(i)) * err
    })


  /**
   * Implement the update of the synapse (weight, grad weight) following the
   * back propagation of output error. This method is called during training.
   */
  def connectionUpdate: Unit =
  // Iterates through all element of the destination layer except the bias element
    Range(1, dst.len).foreach(i => {
      val delta = dst.delta(i)

      // Compute all the synapses (weight, gradient weight) between
      // the destination elements (index i) and the source elements (index j)
      Range(0, src.len).foreach(j => {
        val _output = src.output(j)
        val oldSynapse = synapses(i)(j)
        // Compute the gradient with the delta
        val grad = config.learningRate * delta * _output
        // Apply the gradient adjustment formula
        val deltaWeight = grad + config.momentum * oldSynapse._2
        // Update the synapse
        synapses(i)(j) = (oldSynapse._1 + deltaWeight, grad)
      })
    })

  /**
   * Convenient method to update the values of a synapse while
   * maintaining immutability
   */
  private def update(i: Int, j: Int, x: Double, dx: Double): Unit = {
    val old = synapses(i)(j)
    synapses(i)(j) = (old._1 + x, dx)
  }

  private def isOutLayer: Boolean = dst.id == config.outLayerId
}
