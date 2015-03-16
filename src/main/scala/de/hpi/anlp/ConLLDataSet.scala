/*
 * Copyright (C) 20011-2014 Scalable minds UG (haftungsbeschränkt) & Co. KG. <http://scm.io>
 */
package de.hpi.anlp

import org.deeplearning4j.datasets.fetchers.BaseDataFetcher
import org.deeplearning4j.models.word2vec.Word2Vec
import org.deeplearning4j.text.movingwindow.{Window, Windows, WindowConverter}
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.util.FeatureUtil
import scala.collection.JavaConversions._

class ConLLWordVectorDataFetcher(val vec: Word2Vec, val labels: List[String], val conLLFile: ConLLFileReader) extends BaseDataFetcher {

  val iter = conLLFile.iterator
  
  val labelIdx = labels.zipWithIndex.toMap
  
  val factory = new DefaultTokenizerFactory()
  
  var leftOver = Vector.empty[DataSet]
  
  override def fetch(numExamples: Int): Unit = {

    if(leftOver.size >= numExamples) {
      curr = DataSet.merge(leftOver.take(numExamples))
      leftOver = leftOver.drop(numExamples)
      cursor += curr.numExamples()
    } else if(!iter.hasNext) {
      if(!leftOver.isEmpty) {
        curr = DataSet.merge(leftOver)
        leftOver = Vector.empty
        cursor += curr.numExamples()
      }
    } else {
      val list = iter.take(numExamples).flatMap { example =>
        val words = example.map(_.token)
        val labels = example.map(_.tag)
        Windows.windows(words, vec.getWindow()).zip(labels).map {
          case (window, label) =>
            val wordVector = WindowConverter.asExampleArray(window, vec, false)
            val labelVector = FeatureUtil.toOutcomeVector(labelIdx(label), labels.size)
            new DataSet(wordVector, labelVector)
        }
      }
      
      val merge = (list ++ leftOver).take(numExamples).toList

      curr = DataSet.merge(merge)
      cursor += curr.numExamples()

      if(list.hasNext)
        leftOver ++= list
    }
  }
  
  override def inputColumns() =
    vec.lookupTable().layerSize() * vec.getWindow()

  override def totalOutcomes() =
    labels.size

  override def hasMore() =
    iter.hasNext || leftOver.size > 0
}

