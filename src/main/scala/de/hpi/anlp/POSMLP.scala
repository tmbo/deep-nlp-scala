/*
 * Copyright (C) 20011-2014 Scalable minds UG (haftungsbeschränkt) & Co. KG. <http://scm.io>
 */
package de.hpi.anlp

import de.hpi.anlp.ann.MLP.MLPMultiClassifier
import de.hpi.anlp.ann.{MLP, MLPConfig}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

case class WordDictionary(underlying: scala.collection.Map[String, Array[Double]], numStates: Int) {
  val nullVec = Array.fill(numStates)(0.0)

  val uniformVec = Array.fill(numStates)(1.0 / numStates)

  def word2vec(word: String): Array[Double] = underlying.get(word) match {
    case Some(vec) => vec
    case _ if word == POSMLP.BORDER => nullVec
    case _ => uniformVec
  }

  def words2vec(words: List[String]) =
    Array.concat(words.map(word2vec): _*)
}

case class TagDictionary(states: List[String]) {
  val stateIdx = states.zipWithIndex.toMap

  val revIdx = states.zipWithIndex.map {
    case (el, i) => i -> el
  }.toMap

  val size = states.size

  val tag2vec: Map[String, Array[Double]] = states.zipWithIndex.map {
    case (state, idx) =>
      val a = Array.fill(size)(0.0)
      a(idx) = 1
      state -> a
  }.toMap.withDefaultValue(Array.fill(size)(0.0))
}

case class POSMLP(mlp: MLP[Double], tags: TagDictionary, dict: WordDictionary, preW: Int, postW: Int) {
  def output(sentence: List[String]): Seq[String] = {
    POSMLP.slidingWindow(sentence, preW, postW).map { window =>
      val features = dict.words2vec(window)
      labelForArray(mlp |> features)
    }.toSeq
  }

  private def imax(a: Array[Double]): Int = {
    var mv: Option[Double] = None
    var mi: Option[Int] = None
    var i = 0
    while (i < a.size) {
      if (mv.isEmpty || a(i) > mv.get) {
        mi = Some(i)
        mv = Some(a(i))
      }
      i += 1
    }
    mi getOrElse -1
  }

  private def labelForArray(a: Array[Double]): String = {
    tags.revIdx(imax(a))
  }
}

object POSMLP {
  val BORDER = "<=BORDER=>"

  def slidingWindow(sentence: Seq[String], preW: Int, postW: Int) = {
    val pre = List.fill(preW)(BORDER)

    val post = List.fill(postW)(BORDER)

    (pre ++ sentence ++ post).sliding(preW + 1 + postW)
  }


  private def caluculateWordDictionary(tags: TagDictionary, annotatedData: Iterable[List[AnnotatedToken]]): WordDictionary = {

    val emissions = mutable.HashMap.empty[String, Array[Double]]
    val counter   = mutable.HashMap.empty[String, Int].withDefaultValue(0)

    annotatedData.foreach { annotated =>
      annotated.foreach {
        case AnnotatedToken(token, tag) =>
          val idx = tags.stateIdx(tag)
          counter.update(token, counter(token) + 1)
          emissions.get(token) match {
            case Some(a) =>
              a(idx) += 1
            case _ =>
              val a = Array.fill(tags.size)(0.0)
              a(idx) = 1
              emissions += (token -> a)
          }
      }
    }
    
    emissions.foreach{
      case (token, freqs) =>
        (0 until freqs.length).foreach{i =>
          freqs.update(i, freqs(i) / counter(token))
        }
    }

    WordDictionary(emissions, tags.size)
  }

  def calculateXY(annotatedData: Iterable[List[AnnotatedToken]], preW: Int, postW: Int, dict: WordDictionary, tags: TagDictionary) = {
    val X = new ArrayBuffer[Array[Double]]()
    val y = new ArrayBuffer[Array[Double]]()

    annotatedData.foreach { annotated =>
      slidingWindow(annotated.view.map(_.token), preW, postW).foreach { window =>
        if (window.size == preW + 1 + postW) {
          X.append(dict.words2vec(window))
        }
      }

      annotated.foreach {
        case AnnotatedToken(_, tag) =>
          y.append(tags.tag2vec(tag))
      }
    }
    (X.toArray, y.toArray)
  }

  def train(mlpCfg: MLPConfig, states: List[String], annotatedData: Iterable[List[AnnotatedToken]], preW: Int = 2, postW: Int = 2) = {

    val tags = TagDictionary(states)
    val dict = caluculateWordDictionary(tags, annotatedData)
    println("Finished creation word dictionary. Size: " + dict.underlying.size)
    val (features, labels) = calculateXY(annotatedData, preW, postW, dict, tags)
    println("Finished calculating features. Size: " + features.size)
    
    implicit val mlpObjective = new MLPMultiClassifier
    val mlp = MLP.apply[Double](mlpCfg, features.take(10000), labels.take(10000))
    
    POSMLP(mlp, tags, dict, preW, postW)
  }
}
