package de.hpi.anlp.utils

import de.hpi.anlp.conll.AnnotatedToken
import scala.collection.mutable

/**
 * A word dictionary is a lookup table for seen words. There is a fallback for unseen words
 */
case class WordDictionary(underlying: scala.collection.Map[String, Array[Double]], numStates: Int) {
  /**
   * Value that gets returned if a sentence border is reached 
   */
  val nullVec = Array.fill(numStates)(0.0)

  /**
   * Value that gets returned if a word hasn't been seen during training 
   */
  val uniformVec = Array.fill(numStates)(1.0 / numStates)

  /**
   * Retrieve the vector representation of a word
   */
  def word2vec(word: String): Array[Double] = underlying.get(word) match {
    case Some(vec) => vec
    case _ if word == SentenceUtils.SENTENCE_BORDER => nullVec
    case _ => uniformVec
  }

  /**
   * Retrieve the vector representation of a word
   */
  def words2vec(words: List[String]) =
    Array.concat(words.map(word2vec): _*)
}

/**
 * Helper to construct a word dictionary given an input data set 
 */
object WordDictionary {

  /**
   * Use the given annotated data to build up a word dictionary containing the probabilities of a word occouring with 
   * each tag. Implements the Tag-Prob word representation 
   */
  def build(tags: TagDictionary, annotatedData: Iterable[List[AnnotatedToken]]): WordDictionary = {

    val emissions = mutable.HashMap.empty[String, Array[Double]]
    val counter = mutable.HashMap.empty[String, Int].withDefaultValue(0)

    // Iterate over the data set to count tag <-> token occurrences
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

    // Calculate the emission probabilities for each word and tag combination
    emissions.foreach {
      case (token, freqs) =>
        (0 until freqs.length).foreach { i =>
          freqs.update(i, (freqs(i) + 1) / (counter(token) + freqs.length))
        }
    }

    WordDictionary(emissions, tags.size)
  }
}