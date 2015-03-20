package de.hpi.anlp.hmm

import scala.collection.mutable

/**
 * Extension of the base HMM implementation to use constant smoothing. This is especially useful for words and tag word
 * combinations that were not seen during training
 */
class ConstantSmoothedHMM(states: List[String], n: Int = 2, smoothingConstant: Int = 1) extends HMM(states, n) {
  override def calculateStartProbabilities(starts: Array[Double]) = {
    val sum = starts.sum  + states.size * smoothingConstant
    starts.map(startCount => (startCount + smoothingConstant)/ sum)
  }

  override def calculateTransitionProbabilities(transitions: Array[Double]) = {
    val sum = transitions.sum + states.size * smoothingConstant
    transitions.map(transitionCount => (transitionCount + smoothingConstant) / sum)
  }

  override def calculateEmissionProbabilities(emissions: Array[mutable.Map[String, Int]]) = {
    emissions.map { emissionsForTag =>
      val sum = emissionsForTag.values.sum + states.size * smoothingConstant
      emissionsForTag.mapValues { tokenFreq =>
        (tokenFreq + smoothingConstant).toDouble / sum
      }.toMap.withDefaultValue(smoothingConstant.toDouble / sum)
    }
  }
}
