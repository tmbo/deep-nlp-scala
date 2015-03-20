package de.hpi.anlp

import de.hpi.anlp.conll.AnnotatedToken
import de.hpi.anlp.hmm.{TrainedHMM, ConstantSmoothedHMM, HMM}

/**
 * Configure, train and evaluate a HMM based model* 
 */
object POSHMM {

  /**
   * Train a new HMM model. The model can either use smoothing or not
   */
  def train(trainDocuments: Iterable[List[AnnotatedToken]], states: List[String], smoothed: Boolean) = {
    val hmm =
      if (smoothed)
        new ConstantSmoothedHMM(states, smoothingConstant = 1)
      else
        new HMM(states)

    hmm.train(trainDocuments)
  }

  /**
   * Evaluate a given HMM on a test data set and its gold standart
   */
  def evaluate(hmm: TrainedHMM, testDocuments: Iterable[List[AnnotatedToken]], states: List[String]) = {
    val evaluator = new POSEvaluator(states)

    testDocuments.foreach { sentence =>
      val unannotated = sentence.map(_.token)
      val (prob, tags) = hmm.mostProbablePath(unannotated)
      evaluator.add(tagged = tags, gold = sentence.map(_.tag))
    }

    evaluator
  }

}
