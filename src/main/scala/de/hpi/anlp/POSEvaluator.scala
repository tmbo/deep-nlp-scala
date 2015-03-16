/*
 * Copyright (C) 20011-2014 Scalable minds UG (haftungsbeschränkt) & Co. KG. <http://scm.io>
 */
package de.hpi.anlp

import scala.collection.breakOut
import scala.collection.mutable

class POSEvaluator(tags: List[String]) {
  val tagCounter: Map[String, mutable.Map[String, Int]] = tags.map { tag =>
    tag -> mutable.HashMap("system" -> 0, "gold" -> 0, "both" -> 0)
  }(breakOut)
  
  def add(tagged: Seq[String], gold: Seq[String]) : Unit = {
    tagged.zip(gold).map{
      case (systemTag, goldTag) =>
        tagCounter(systemTag)("system") += 1
        tagCounter(goldTag)("gold") += 1
        if(systemTag == goldTag)
          tagCounter(goldTag)("both") += 1
    }
  }
  
  def printEvaluation(): Unit = {
    val overall = tagCounter.values.map(_("system")).sum
    val correct = tagCounter.values.map(_("both")).sum

    println("%5s %6s %6s %6s".format("", "Prec", "Rec","F1"))

    tagCounter.foreach{
      case (tag, counts) =>
        val p = precision(counts)
        val r = recall(counts)
        val f1Score = f1(p, r)
        println("%5s %.4f %.4f %.4f".format(tag, p, r, f1Score))
    }

    println("\nAccuracy: %.4f".format(correct.toDouble / overall))
  }
  
  def precision(counts: mutable.Map[String, Int]) = {
    if(counts("system") == 0)
      Double.NaN
    else 
      counts("both").toDouble / counts("system")
  }

  def recall(counts: mutable.Map[String, Int]) = {
    if(counts("gold") == 0)
      Double.NaN
    else
      counts("both").toDouble / counts("gold")
  }

  def f1(precision: Double, recall: Double) = {
    if(precision + recall == 0 || precision + recall == Double.NaN)
      Double.NaN
    else
      2 * precision * recall / (precision + recall)
  }
}
