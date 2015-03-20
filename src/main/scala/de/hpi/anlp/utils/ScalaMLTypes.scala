package de.hpi.anlp.utils

/**
 * Types and conversion between ML types and native Scala types
 */
object ScalaMLTypes {
  type DblMatrix = Array[Array[Double]]
  type DblVector = Array[Double]
  type MLPTask = DblVector => DblVector
}
