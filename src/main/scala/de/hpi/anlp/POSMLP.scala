package de.hpi.anlp

import de.hpi.anlp.mlp.MLPConfig
import de.hpi.anlp.conll.AnnotatedToken
import de.hpi.anlp.nnpos.POSMLPModel

/**
 * Configure, train and evaluate a MLP model 
 */
object POSMLP {

  /**
   * Train a new MLP model using the given training data and states. The configuration can be adjusted in this function.
   */
  def train(trainDocuments: Iterable[List[AnnotatedToken]], states: List[String]) = {
    val NUM_EPOCHS = 1000
    val EPS = 0.00001
    val learningRate = 0.01
    val hiddenLayers = Array[Int]()
    val momentum = 0.1
    val activationF = (x: Double) => 1.0 / (1.0 + Math.exp(-0.8 * x))

    val config = MLPConfig(momentum, learningRate, hiddenLayers, NUM_EPOCHS, EPS, activationF)

    POSMLPModel.fit(config, states, trainDocuments, preW = 2, postW = 2)
  }

  /**
   * Evaluate a given MLP model on the test data set and its gold standart
   */
  def evaluate(mlp: POSMLPModel, testDocuments: Iterable[List[AnnotatedToken]], states: List[String]) = {
    val evaluator = new POSEvaluator(states)

    testDocuments.foreach { sentence =>
      val unannotated = sentence.map(_.token)
      val tags = mlp.output(unannotated)
      evaluator.add(tagged = tags, gold = sentence.map(_.tag))
    }
    evaluator
  }

}
