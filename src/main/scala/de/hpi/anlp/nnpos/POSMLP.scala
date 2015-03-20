package de.hpi.anlp.nnpos

import de.hpi.anlp.mlp.{MLP, MLPConfig, MLPTasks}
import de.hpi.anlp.conll.AnnotatedToken
import de.hpi.anlp.utils.{SentenceUtils, WordDictionary, TagDictionary}
import scala.collection.mutable.ArrayBuffer

case class POSMLPModel(mlp: MLP, tags: TagDictionary, dict: WordDictionary, preW: Int, postW: Int) {
  def output(sentence: List[String]): Seq[String] = {
    SentenceUtils.slidingWindow(sentence, preW, postW).map { window =>
      val features = dict.words2vec(window)
      labelForArray(mlp.output(features))
    }.toSeq
  }

  private def imax(a: Array[Double]): Int = {
    var mv: Option[Double] = None
    var mi: Option[Int] = None
    var i = 1
    while (i < a.size) {
      if (mv.isEmpty || a(i) > mv.get) {
        mi = Some(i)
        mv = Some(a(i))
      }
      i += 1
    }
    mi getOrElse 0
  }

  private def labelForArray(a: Array[Double]): String = {
    tags.revIdx(imax(a) - 1)
  }
}

object POSMLPModel {

  private def calculateXY(annotatedData: Iterable[List[AnnotatedToken]], preW: Int, postW: Int, dict: WordDictionary, tags: TagDictionary) = {
    val X = new ArrayBuffer[Array[Double]]()
    val y = new ArrayBuffer[Array[Double]]()

    annotatedData.foreach { annotated =>
      SentenceUtils.slidingWindow(annotated.view.map(_.token), preW, postW).foreach { window =>
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

  def fit(mlpCfg: MLPConfig, states: List[String], annotatedData: Iterable[List[AnnotatedToken]], preW: Int = 2, postW: Int = 2) = {

    val tags = TagDictionary(states)
    val dict = WordDictionary.build(tags, annotatedData)
    println("Finished creation word dictionary. Size: " + dict.underlying.size)
    val (features, labels) = calculateXY(annotatedData, preW, postW, dict, tags)
    println("Finished calculating features. Size: " + features.size)

    val mlp = new MLP(mlpCfg, features, labels)(MLPTasks.MLPMultiClassifier _)

    POSMLPModel(mlp, tags, dict, preW, postW)
  }
}
