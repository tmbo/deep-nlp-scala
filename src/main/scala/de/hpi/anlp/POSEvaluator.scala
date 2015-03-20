package de.hpi.anlp

import java.util.Locale
import scala.collection.breakOut
import scala.collection.mutable

/**
 * Given the guessed tags of a trained model and a data sets gold standart this class calculates tag based accuracy
 * F1 score and overall accuracx 
 */
class POSEvaluator(tags: List[String]) {
  /**
   * Underlying counter for occurrences 
   */
  val tagCounter: Map[String, mutable.Map[String, Int]] = tags.map { tag =>
    tag -> mutable.HashMap("system" -> 0, "gold" -> 0, "both" -> 0)
  }(breakOut)

  /**
   * Add another guessed, gold annotation to the evaluator 
   */
  def add(tagged: String, gold: String): Unit = {
    tagCounter(tagged)("system") += 1
    tagCounter(gold)("gold") += 1
    if (tagged == gold)
      tagCounter(gold)("both") += 1
  }

  def add(tagged: Seq[String], gold: Seq[String]): Unit = {
    tagged.zip(gold).map {
      case (systemTag, goldTag) =>
        add(systemTag, goldTag)
    }
  }

  /**
   * Prints the evalution to the standart out
   */
  def printEvaluation(): Unit = {
    val overall = tagCounter.values.map(_("system")).sum
    val correct = tagCounter.values.map(_("both")).sum

    println("%5s, %6s, %6s, %6s".format("", "Prec", "Rec", "F1"))

    tagCounter.foreach {
      case (tag, counts) =>
        val p = precision(counts)
        val r = recall(counts)
        val f1Score = f1(p, r)
        println("%5s, %.4f, %.4f, %.4f".formatLocal(Locale.ENGLISH, tag, p, r, f1Score))
    }

    println("\nAccuracy: %.4f".format(correct.toDouble / overall))
  }

  def precision(counts: mutable.Map[String, Int]) = {
    if (counts("system") == 0)
      Double.NaN
    else
      counts("both").toDouble / counts("system")
  }

  def recall(counts: mutable.Map[String, Int]) = {
    if (counts("gold") == 0)
      Double.NaN
    else
      counts("both").toDouble / counts("gold")
  }

  def f1(precision: Double, recall: Double) = {
    if (precision + recall == 0 || precision + recall == Double.NaN)
      Double.NaN
    else
      2 * precision * recall / (precision + recall)
  }
}
