package de.hpi.anlp.conll

import org.deeplearning4j.datasets.fetchers.BaseDataFetcher
import org.deeplearning4j.models.word2vec.Word2Vec
import org.deeplearning4j.text.movingwindow.{WindowConverter, Windows}
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.util.FeatureUtil

import scala.collection.JavaConversions._

/**
 * Data fetcher to enumerate word vectors from ConLL files
 */
class ConLLWordVectorDataFetcher(val vec: Word2Vec, val labels: List[String], val conLLFile: ConLLFileReader) extends BaseDataFetcher {

  // Iterator over the files contents
  val iter = conLLFile.iterator

  // Label index
  val labelIdx = labels.zipWithIndex.toMap

  // Tokenizer to improve tokens 
  val factory = new DefaultTokenizerFactory()

  // If the requested number of examples doesn't align with the tokens in a sentence we need to save left over tokens
  var leftOver = Vector.empty[DataSet]

  /**
   * Fetch the next numExample tokens
   * @param numExamples Number of tokens
   */
  override def fetch(numExamples: Int): Unit = {

    if (leftOver.size >= numExamples) {
      curr = DataSet.merge(leftOver.take(numExamples))
      leftOver = leftOver.drop(numExamples)
      cursor += curr.numExamples()
    } else if (!iter.hasNext) {
      if (!leftOver.isEmpty) {
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

      if (list.hasNext)
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

