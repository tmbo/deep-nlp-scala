package de.hpi.anlp.hmm

import scala.collection.breakOut

/**
 * A trained HMM which can be used to predict tags on a given sentence. Its a result of the training of a configured HMM
 */
case class TrainedHMM(states: List[String],
                      n: Int,
                      underlyingEmissionProbs: Array[Map[String, Double]],
                      underlyingTransitionProbs: Array[Double],
                      underlyingStartProbs: Array[Double]) {

  def incommingProbabilities(trellis: Vector[Array[Double]], sidx: Int, trellisLevel: Int, emissionPs: Array[Double]): Map[Int, Double] = {
    (0 until states.size).map { prevStateIdx =>
      val probability = trellis(trellisLevel - 1)(prevStateIdx) +
        math.log(transitionProbability(prevStateIdx, sidx)) +
        math.log(emissionPs(sidx))

      prevStateIdx -> probability
    }(breakOut)
  }

  /**
   * Viterbi implementation for graph traversal. Used to find most probable hidden states for the observed outputs.
   */
  def viterbi(observations: List[String]) = {
    observations match {
      case Nil =>
        0.0 -> Vector.empty
      case firstObservation :: remainingObservations =>
        var paths = states.toArray.map(state => Vector(state))
        val initialNodeLevel = states.zipWithIndex.toArray.map {
          case (state, idx) =>
            math.log(startProbability(idx)) + math.log(emissionProbability(idx, firstObservation))
        }

        var trellis = Vector(initialNodeLevel)

        remainingObservations.zipWithIndex.map {
          case (observation, trellisLevel) =>
            val nextNodeLevel = new Array[Double](states.size)
            val updatedPaths = new Array[Vector[String]](states.size)
            var emissionPs: Array[Double] = (0 until states.size).map { idx =>
              emissionProbability(idx, observation)
            }(breakOut)

            if (emissionPs.forall(_ == 0))
              emissionPs = Array.fill(states.size)(1.0)

            states.zipWithIndex.map {
              case (state, sidx) =>
                val (bestState, bestProb) = incommingProbabilities(trellis, sidx, trellisLevel + 1, emissionPs).maxBy(_._2)
                nextNodeLevel.update(sidx, bestProb)
                updatedPaths.update(sidx, paths(bestState) :+ state)
            }

            trellis :+= nextNodeLevel
            paths = updatedPaths
        }

        val (bestProb, bestState) = trellis.last.zipWithIndex.maxBy(_._1)
        bestProb -> paths(bestState)
    }
  }

  def emissionProbability(sidx: Int, observation: String): Double =
    underlyingEmissionProbs(sidx)(observation)

  def transitionProbability(from: Int, to: Int): Double =
    underlyingTransitionProbs(from * states.size + to)

  def startProbability(sidx: Int): Double =
    underlyingStartProbs(sidx)

  def mostProbablePath(observations: List[String]) = viterbi(observations)
}