package de.hpi.anlp.mlp

import scala.util.{Try, Success, Failure}
import org.apache.log4j.Logger
import de.hpi.anlp.utils.ScalaMLTypes._

class MLP(
           config: MLPConfig,
           xt: Array[Array[Double]],
           labels: DblMatrix)
         (implicit mlpObjective: MLPTask) {

  private val logger = Logger.getLogger("MLP")

  // Flag that indicates that the training converged toward a definite model
  private[this] var converged = false

  /**
   * Model for the Multi-layer Perceptron of type MLPModel
   */
  val model: Option[MLPModel] = train match {
    case Success(_model) =>
      Some(_model)
    case Failure(e) =>
      logger.error("MLP.model ", e)
      None
  }

  /**
   * Test whether the model has converged
   */
  final def hasConverged: Boolean = converged

  /**
   * Define the predictive function of the classifier or regression
   */
  def output: PartialFunction[Array[Double], DblVector] = {
    case x: Array[Double] if (!x.isEmpty && model != None && x.size == xt(0).size) => {

      Try(model.get.getOutput(x)) match {
        case Success(y) => y
        case Failure(e) => {
          logger.error("MLP ", e)
          Array.empty[Double]
        }
      }
    }
  }


  /**
   * Computes the accuracy of the training session. The accuracy is estimated
   * as the percentage of the training data points for which the square root of
   * the sum of squares error, normalized by the size of the  training set exceed a
   * predefined threshold
   */
  final def accuracy(threshold: Double): Option[Double] = model.map(m => {

    // counts the number of data points for were correctly classified
    val nCorrects = xt.zip(labels)
      .foldLeft(0)((s, xtl) => {

      // Get the output layer for this input xt.
      val output = model.get.getOutput(xtl._1)

      // Compute the sum of squared error while excluding bias element
      val _sse = xtl._2.zip(output.drop(1))
        .foldLeft(0.0)((err, tp) => {
        val diff = tp._1 - tp._2
        err + diff * diff
      }) * 0.5

      // Compute the least square error and adjusts it for the number of output variables.
      val error = Math.sqrt(_sse) / (output.size - 1)
      if (error < threshold) s + 1 else s
    })

    // returns the percentage of observations correctly classified
    nCorrects.toDouble / xt.size
  })

  /**
   * Training method for the Multi-layer perceptron
   */
  private def train: Try[MLPModel] = {
    Try {
      val _model = new MLPModel(config, xt(0).size, labels(0).size)(mlpObjective)

      // Scaling or normalization factor for the sum of the squared error
      val errScale = 1.0 / (labels(0).size * xt.size)

      // Apply the exit condition for this online training strategy
      // The convergence criteria selected is the reconstruction error
      // generated during an epoch adjusted to the scaling factor and compare
      // to the predefined criteria config.eps
      converged = Range(0, config.numEpochs).find(epoch => {
        val e = xt.toArray.zip(labels).foldLeft(0.0)((s, xtlbl) =>
          s + _model.trainEpoch(xtlbl._1, xtlbl._2)
        ) * errScale
        if (epoch % 10 == 0)
          println("SSE: " + e)
        e < config.eps
      }) != None
      _model
    }
  }
}
