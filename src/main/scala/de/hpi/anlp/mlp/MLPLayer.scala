package de.hpi.anlp.mlp

import de.hpi.anlp.utils.ScalaMLTypes.DblVector

/**
 * A MLP layer is built using the input vector and add an extra element to account for the bias w0
 */
class MLPLayer(val id: Int, val len: Int) {

  /**
   * Values of the output vector
   */
  val output = new DblVector(len)

  /**
   * Difference for the propagated error on the source or upstream
   */
  val delta = new DblVector(len)
  output.update(0, 1.0)

  /**
   * Initialize the value of the input for this MLP layer
   */
  def set(_x: DblVector): Unit = {
    _x.copyToArray(output, 1)
  }

  /**
   * Compute the sum of squared error of the elements of this MLP layer
   */
  final def sse(labels: DblVector): Double = {
    var _sse = 0.0
    output.drop(1).zipWithIndex.foreach {
      case (on, idx) => {
        val err = labels(idx) - on
        delta.update(idx + 1, on * (1.0 - on) * err)
        _sse += err * err
      }
    }
    _sse * 0.5 // normalized C
  }

  /**
   * Is this layer the output layer
   */
  final def isOutput(lastId: Int): Boolean = id == lastId
}
