/*
 * Copyright (C) 20011-2014 Scalable minds UG (haftungsbeschränkt) & Co. KG. <http://scm.io>
 */
package de.hpi.anlp.hmm

import de.hpi.anlp.AnnotatedToken
import scala.collection.mutable
import scala.collection.breakOut


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

class HMM(states: List[String], n: Int = 2) {
  val stateIdx = states.zipWithIndex.toMap

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

  def calculateStartProbabilities(starts: Array[Double]) = {
    val sum = starts.sum
    starts.map(_ / sum)
  }

  def calculateTransitionProbabilities(transitions: Array[Double]) = {
    val sum = transitions.sum
    transitions.map(_ / sum)
  }

  def calculateEmissionProbabilities(emissions: Array[mutable.Map[String, Int]]) = {
    emissions.map { emissionsForTag =>
      val sum = emissionsForTag.values.sum
      emissionsForTag.mapValues { tokenFreq =>
        tokenFreq.toDouble / sum
      }.toMap.withDefaultValue(0.0)
    }
  }
}




