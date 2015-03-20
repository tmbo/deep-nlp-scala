package de.hpi.anlp.mlp

import de.hpi.anlp.utils.Model
import de.hpi.anlp.utils.ScalaMLTypes.{DblVector, MLPTask}

/**
 * MLP model represents a MLP configuration and instance. A MLP model consists of MLPLayer s (layer of the MLP model), 
 * MLPSynapse s (connection between two elements) and MLPConnections (container of synapses of a layer)
 */
class MLPModel(
                config: MLPConfig,
                nInputs: Int,
                nOutputs: Int)(
                implicit mlpObjective: MLPTask) extends Model {
  
  val topology = 
    if (config.nHiddens == 0) 
      Array[Int](nInputs, nOutputs)  // if no hidden layer is set, there is only an output layer
    else 
      Array[Int](nInputs) ++ config.hidLayers ++ Array[Int](nOutputs)

  /*
   * Aarrays of layers for the topology
   */
  val layers: Array[MLPLayer] = topology.zipWithIndex
    .map {
    case (t, idx) =>
      new MLPLayer(idx, t + 1)
  }

  /*
   * Create a array of connection between layer. A connection is
   * made of multiple synapses.
   */
  val connections = Range(0, layers.size - 1).map(n =>
    new MLPConnection(config, layers(n), layers(n + 1))(mlpObjective)).toArray

  /**
   * Alias for the input or first layer in the network
   */
  @inline
  def inLayer: MLPLayer = layers.head

  /**
   * Alias for the last layer (output layer) in the network
   */
  @inline
  def outLayer: MLPLayer = layers.last

  /**
   * Training cycle: Forward propagation of input, back propagation of error and the re-computation of the weight and 
   * gradient of the elements.
   */
  def trainEpoch(x: DblVector, y: DblVector): Double = {
    // Initialize the input layer
    inLayer.set(x)
    // Apply the forward progapation of input to all the connections
    // starting with the input layer
    connections.foreach(_.connectionForwardPropagation)

    // Compute the sum of squared errors
    val _sse = sse(y)

    // Create a back iterator
    val bckIterator = connections.reverseIterator

    // Apply the error back propagation to all the connections
    // starting with the output lauer
    bckIterator.foreach(_.connectionBackpropagation)

    // Finally update the connections (weigths and grad weights) of synapses
    connections.foreach(_.connectionUpdate)
    _sse
  }


  /**
   * Compute the mean squares error for the network as the sum
   * of the mean squares error for each output value.
   */
  @inline
  final def sse(label: DblVector): Double =
    outLayer.sse(label)

  /**
   * Compute the output values for the network using the forward propagation
   */
  def getOutput(x: DblVector): DblVector = {
    inLayer.set(x)

    connections.foreach(_.connectionForwardPropagation)

    outLayer.output
  }

  /**
   * Write the content of this model (weights) into a file
   */
  override def saveToFile: Boolean = {
    val content = new StringBuilder(s"$nInputs,")
    if (config.nHiddens != 0)
      content.append(config.hidLayers.mkString(","))

    content.append(s"$nOutputs\n")
    connections.foreach(c => {
      content.append(s"${c.getLayerIds._1},${c.getLayerIds._2}:")
      content.append(c.getSynapses.map(s => s"${s.mkString(",")}\n"))
    })
    write(content.toString)
  }

  /**
   * Textual description of the model for Multi-layer Perceptron. The representation
   * include the description of the connections and layers.
   */
  override def toString: String = {
    val buf = new StringBuilder
    connections.foreach(buf.append(_))
    layers.foreach(buf.append(_))
    buf.toString
  }
}
