package de.hpi.anlp.mlp

import de.hpi.anlp.utils.ScalaMLTypes._

/**
 * Class for the Regression objective for the MLP. This implementation uses softmax
 */
object MLPTasks {
  def MLPMultiClassifier(y: DblVector): DblVector = {
    val softmaxValues = new DblVector(y.size)
    val expY = y.map(Math.exp(_))
    val expYSum = expY.sum
    expY.map(_ / expYSum).copyToArray(softmaxValues, 1)
    softmaxValues
  }
}