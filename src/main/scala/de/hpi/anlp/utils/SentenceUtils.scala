package de.hpi.anlp.utils

/**
 * Helper class to construct windows over a given sentence. Windows at the border of the sentence are filled up with 
 * a border tag
 */
object SentenceUtils {
  val SENTENCE_BORDER = "<=BORDER=>"


  def slidingWindow(sentence: Seq[String], preW: Int, postW: Int) = {
    val pre = List.fill(preW)(SENTENCE_BORDER)

    val post = List.fill(postW)(SENTENCE_BORDER)

    (pre ++ sentence ++ post).sliding(preW + 1 + postW)
  }
}
