package de.hpi.anlp.ann

import scala.util.{Try, Success, Failure}
import org.apache.log4j.Logger

import org.scalaml.core.Types.ScalaMl._
import org.scalaml.core.XTSeries
import org.scalaml.util.DisplayUtils
import XTSeries._

class MLP[T <% Double](
                                        config: MLPConfig,
                                        xt: XTSeries[Array[T]],
                                        labels: DblMatrix)
                                      (implicit mlpObjective: MLP.MLPObjective){

  import MLP._

  check(xt, labels)
  private val logger = Logger.getLogger("MLP")

  // Flag that indicates that the training converged toward a definite modle
  private[this] var converged = false

  /**
   * Model for the Multi-layer Perceptron of type MLPModel. This implementation
   * allows the model to be created even in the training does not converged towards
   * a stable network of synapse weights. The client code is responsible for
   * evaluating the value of the state variable converge and perform a validation run
   */
  val model: Option[MLPModel] = train match {
    case Success(_model) => Some(_model)
    case Failure(e) => DisplayUtils.none("MLP.model ", logger, e)
  }


  /**
   * Test whether the model has converged. In some cases, a MLP model
   * may be created although the training has not converged.
   * @return true if the training execution converges, false otherwise
   */
  @inline
  final def hasConverged: Boolean = converged

  /**
   * <p>Define the predictive function of the classifier or regression as a data
   * transformation by overriding the pipe operator |>.</p>
   * @throws MatchError if the model is undefined or the input string has an incorrect size
   * @return PartialFunction of features vector of type Array[T] as input and
   *         the predicted vector values as output
   */
  def |> : PartialFunction[Array[T], DblVector] = {
    case x: Array[T] if (!x.isEmpty && model != None && x.size == dimension(xt)) => {

      Try(model.get.getOutput(x)) match {
        case Success(y) => y
        case Failure(e) => {
          DisplayUtils.error("MLP.|> ", logger, e)
          Array.empty[Double]
        }
      }
    }
  }


  /**
   * <p>Computes the accuracy of the training session. The accuracy is estimated
   * as the percentage of the training data points for which the square root of
   * the sum of squares error, normalized by the size of the  training set exceed a
   * predefined threshold.</p>
   * @param threshold threshold applied to the square root of the sum of squares error to
   *                  validate the data point
   * @return accuracy value [0, 1] if model exits, None otherwise
   */
  final def accuracy(threshold: Double): Option[Double] = model.map(m => {

    // counts the number of data points for were correctly classified
    val nCorrects = xt.toArray.zip(labels)
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

  /*
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
      converged = Range(0, config.numEpochs).find(_ => {
        xt.toArray.zip(labels).foldLeft(0.0)((s, xtlbl) =>
          s + _model.trainEpoch(xtlbl._1, xtlbl._2)
        ) * errScale < config.eps
      }) != None
      _model
    }
  }
}


/**
 * <p>Companion object for the Multi-layer Perceptron. The singleton is used to:<br>
 * Define several variants of the constructor<br>
 * Define the class/trait hierarchy for the objective of the MLP {classification, regression}</p>
 *
 * @author Patrick Nicolas
 * @since May 8, 2014
 * @note Scala for Machine Learning Chapter 9 Artificial Neural Network /
 *       Multilayer perceptron/Training cycle/epoch
 */
object MLP {
  private val EPS = 1e-5

  /**
   * <p>Trait that defined the signature of the objective function.<br>
   * += for updating parameters if needed<br>
   * normalize to normalize the output.</p>
   */
  trait MLPObjective {
    /**
     * <p>Normalize the output vector to match the objective of the MLP. The
     * output vector is the output layers minus the bias, output(0).</p>
     * @param output raw output vector
     * @return normalized output vector
     */
    def apply(output: DblVector): DblVector
  }

  /**
   * Class for the Regression objective for the MLP. This implementation uses softmax
   */
  class MLPMultiClassifier extends MLPObjective {

    /**
     * <p>Normalize the output vector to match the objective of the MLP. The
     * output vector is the output layers minus the bias, output(0).</p>
     * @param output raw output vector
     * @return normalized output vector
     */
    override def apply(output: DblVector): DblVector = softmax(output.drop(1))

    private def softmax(y: DblVector): DblVector = {
      val softmaxValues = new DblVector(y.size)
      val expY = y.map(Math.exp(_))
      val expYSum = expY.sum
      expY.map(_ / expYSum).copyToArray(softmaxValues, 1)
      softmaxValues
    }
  }

  /**
   * Default constructor for the Multi-layer perceptron (type MLP)
   * @param config  Configuration parameters class for the MLP
   * @param xt Time series of features in the training set
   * @param labels  Labeled or target observations used for training
   * @param mlpObjective Objective of the model (classification or regression)
   */
  def apply[T <% Double](
                          config: MLPConfig,
                          xt: XTSeries[Array[T]],
                          labels: DblMatrix)
                        (implicit mlpObjective: MLP.MLPObjective): MLP[T] =
    new MLP[T](config, xt, labels)

  /**
   * Constructor for the Multi-layer perceptron (type MLP) that takes an array of
   * observation as argument
   * @param config  Configuration parameters class for the MLP
   * @param obs Array of observations used in the training set
   * @param labels  Labeled or target observations used for training
   * @param mlpObjective Objective of the model (classification or regression)
   */
  def apply[T <% Double](
                          config: MLPConfig,
                          obs: Array[Array[T]],
                          labels: DblMatrix)
                        (implicit mlpObjective: MLP.MLPObjective): MLP[T] =
    new MLP[T](config, XTSeries[Array[T]](obs), labels)

  /**
   * Constructor for the Multi-layer perceptron (type MLP) that takes an array of observation as
   * argument and a one dimension label data
   * @param config  Configuration parameters class for the MLP
   * @param obs Array of observations used in the training set
   * @param labels  Array of One variable labels
   * @param mlpObjective Objective of the model (classification or regression)
   */
  def apply[T <% Double](
                          config: MLPConfig,
                          obs: Array[Array[T]],
                          labels: DblVector)
                        (implicit mlpObjective: MLP.MLPObjective): MLP[T] =

    new MLP[T](config, XTSeries[Array[T]](obs), labels.map(Array[Double](_)))


  private def check[T](xt: XTSeries[Array[T]], labels: DblMatrix): Unit = {
    require(!xt.isEmpty,
      "Features for the MLP are undefined")
    require(!labels.isEmpty,
      "Labeled observations for the MLP are undefined")
    require(xt.size == labels.size,
      s"Number of features for MLP ${xt.size} is different from number of labels ${labels.size}")
  }
}

// ----------------------------------------------  EOF ------------------------------------