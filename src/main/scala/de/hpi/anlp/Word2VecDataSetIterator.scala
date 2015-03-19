/*
 * Copyright (C) 20011-2014 Scalable minds UG (haftungsbeschränkt) & Co. KG. <http://scm.io>
 */
package de.hpi.anlp

import de.hpi.WindowConverter
import org.deeplearning4j.datasets.iterator.{DataSetPreProcessor, DataSetIterator}
import org.deeplearning4j.models.word2vec.Word2Vec
import org.deeplearning4j.text.inputsanitation.InputHomogenization
import org.deeplearning4j.text.movingwindow.{Windows, Window}
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.util.FeatureUtil

/**
 * Allows for customization of all of the params of the iterator
 * @param vec the word2vec model to use
 * @param sentenceIter the sentence iterator to use
 * @param labels the possible labels
 * @param batch the batch size
 */  
class Word2VecDataSetIterator(vec: Word2Vec, sentenceIter: Iterable[List[AnnotatedToken]], labels: List[String], val batch: Int = 10) extends DataSetIterator {
  
  var iter = windowIter()
  
  val labelIdx = labels.zipWithIndex.toMap
  
  var preProcessor: Option[DataSetPreProcessor] = None

  def windowIter() = {
    var counter = 0
    sentenceIter.flatMap{ sentence =>
      import scala.collection.JavaConversions._
      val words = sentence.map(s => new InputHomogenization(s.token).transform())
      val wordLabels = sentence.map(_.tag)
      counter += 1
      if(counter % 3500 == 0)
        println("Processing sentence " + counter)
      Windows.windows(words, vec.getWindow()).zip(wordLabels).map {
        case (window, label) =>
          window.setLabel(label)
          window
      }
    }.toList
  }

  /**
   * Like the standard next method but allows a
   * customizable number of examples returned
   *
   * @param num the number of examples
   * @return the next data applyTransformToDestination
   */
  override def next(num: Int): DataSet = {
    synchronized {
      try {
        val windows = iter.take(num).toList
        
        iter = iter.drop(num)
        
        if (windows.isEmpty)
          null
        else {
          val inputs = Nd4j.create(windows.size, inputColumns())
          val labelOutput = Nd4j.create(windows.size, labels.size)

          windows.zipWithIndex.foreach {
            case (window, row) =>
              inputs.putRow(row, WindowConverter.asExampleMatrix(window, vec))
              labelOutput.putRow(row, FeatureUtil.toOutcomeVector(labelIdx(window.getLabel), labels.size))
          }

          val ds = new DataSet(inputs, labelOutput)

          preProcessor.foreach { pp =>
            pp.preProcess(ds)
          }

          ds
        }
      } catch {
        case e: Exception =>
          println("Exception raised: " + e.getMessage)
          e.printStackTrace()
          throw e
      }
    }
  }

  override def totalExamples(): Int = {
    throw new UnsupportedOperationException()
  }
  
  override def inputColumns(): Int= {
    vec.lookupTable().layerSize() * vec.getWindow()
  }
  
  override def totalOutcomes(): Int= {
    labels.size
  }

  override def reset() = {
    iter = windowIter()
  }

  override def cursor(): Int = {
    0
  }

  @Override
  override def numExamples(): Int = {
    0
  }

  /**
   * Returns {true} if the iteration has more elements.
   * (In other words, returns {true} if {#next} would
   * return an element rather than throwing an exception.)
   *
   * @return {true} if the iteration has more elements
   */
  override def hasNext(): Boolean =  {
    !iter.isEmpty
  }

  /**
   * Returns the next element in the iteration.
   *
   * @return the next element in the iteration
   */
  override def next(): DataSet = {
    next(batch)
  }

  /**
   * Removes from the underlying collection the last element returned
   * by this iterator (optional operation).  This method can be called
   * only once per call to {@link #next}.  The behavior of an iterator
   * is unspecified if the underlying collection is modified while the
   * iteration is in progress in any way other than by calling this
   * method.
   *
   * @throws UnsupportedOperationException if the {remove}
   *                                       operation is not supported by this iterator
   * @throws IllegalStateException         if the {next} method has not
   *                                       yet been called, or the {remove} method has already
   *                                       been called after the last call to the {next}
   *                                       method
   */
  override def remove(): Unit = {
    throw new UnsupportedOperationException()
  }

  override def setPreProcessor(dataSetPreprocessor: DataSetPreProcessor): Unit = {
    preProcessor =  Some(dataSetPreprocessor)
  }
}
