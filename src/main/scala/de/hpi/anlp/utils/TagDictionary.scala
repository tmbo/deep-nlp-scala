package de.hpi.anlp.utils

/**
 * A wraper around helpers with tags. Creates a reverse tag index and can be used to convert a tag to a one-hot vector
 */
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
