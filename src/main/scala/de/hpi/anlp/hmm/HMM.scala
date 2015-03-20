package de.hpi.anlp.hmm

import de.hpi.anlp.conll.AnnotatedToken
import scala.collection.mutable

/**
 * HMM to be configured. After train is called a new trainedHMM instance is created
 */
class HMM(states: List[String], n: Int = 2) {
  val stateIdx = states.zipWithIndex.toMap

  /**
   * Train traverses the input data, collects statistics and calculates probabilities for hidden states and outputs. 
   * Those can then be used in the HMM to predict tags for unseen sentences 
   */
  def train(annotatedData: Iterable[List[AnnotatedToken]]) = {
    val transitions = new Array[Double](math.pow(states.size, n).toInt)

    val emissions = Array.fill(states.size)(mutable.HashMap.empty[String, Int].withDefaultValue(0))

    val starts = new Array[Double](states.size)

    annotatedData.foreach { annotated =>
      annotated.headOption.map { first =>
        val sIdx = stateIdx(first.tag)
        starts(sIdx) += 1
      }

      val idxs = annotated.map {
        case AnnotatedToken(token, tag) =>
          val idx = stateIdx(tag)
          emissions(idx).update(token, emissions(idx)(token) + 1)
          idx
      }

      idxs.sliding(n, 1).foreach { window =>
        if (window.size == n) {
          val idx = window.foldLeft(0)((p, s) => p * states.size + s)
          transitions(idx) = transitions(idx) + 1
        }
      }
    }

    val emissionProbs = calculateEmissionProbabilities(emissions)
    val transitionProbs = calculateTransitionProbabilities(transitions)
    val startProbs = calculateStartProbabilities(starts)
    new TrainedHMM(states, n, emissionProbs, transitionProbs, startProbs)
  }

  /**
   * Given the number of seen starts, calculate the start probabilities
   */
  def calculateStartProbabilities(starts: Array[Double]) = {
    val sum = starts.sum
    starts.map(_ / sum)
  }

  /**
   * Given the transition statistics, calculate transition probabilities
   */
  def calculateTransitionProbabilities(transitions: Array[Double]) = {
    val sum = transitions.sum
    transitions.map(_ / sum)
  }

  /**
   * Given emission statistics calculate emission probabilities for each hidden state
   */
  def calculateEmissionProbabilities(emissions: Array[mutable.Map[String, Int]]) = {
    emissions.map { emissionsForTag =>
      val sum = emissionsForTag.values.sum
      emissionsForTag.mapValues { tokenFreq =>
        tokenFreq.toDouble / sum
      }.toMap.withDefaultValue(0.0)
    }
  }
}




