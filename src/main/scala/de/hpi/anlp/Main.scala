package de.hpi.anlp

import de.hpi.anlp.conll.ConLLFileReader

/**
 * Main console application 
 */
object Main extends App {

  // Tags to use for POS tagging
  val states = List("NOUN", "ADV", "PRT", ".", "ADP", "DET", "PRON", "VERB", "X", "NUM", "CONJ", "ADJ")

  // Training documents
  val trainDocuments = new ConLLFileReader("assets/de-train.tt")

  // Test documents
  val testDocuments = new ConLLFileReader("assets/de-eval.tt")

  // Lets have a look what arguments where passed in
  println("ARGS: " + args.mkString(" , "))

  // Call the appropriate training according to the passed argument
  args.headOption match {
    case Some("mlp") =>
      trainMLP()
    case Some("rbm") =>
      trainRBM()
    case Some("hmm-s") =>
      trainHMM(smoothed = true)
    case Some("hmm") =>
      trainHMM(smoothed = false)
    case _ =>
      throw new Exception("You need to specify the model to train. One of 'mlp', 'rbm', 'hmm-s', 'hmm'")
  }

  def trainHMM(smoothed: Boolean): Unit = {
    val hmm = POSHMM.train(trainDocuments, states, smoothed)

    println("\n--- Evaluation of: HMM.smoothed=" + smoothed)
    POSHMM.evaluate(hmm, testDocuments, states).printEvaluation()
    println("---")
  }

  def trainMLP() = {
    val mlp = POSMLP.train(trainDocuments, states)

    println("\n--- Evaluation of MLP ...")
    POSMLP.evaluate(mlp, testDocuments, states).printEvaluation()
    println("---")
  }

  def trainRBM() = {
    val (network, vec) = POSRBM.train(trainDocuments, states)

    println("\n--- Evaluation of RBM ...")
    POSRBM.evaluate(network, testDocuments, vec, states).printEvaluation()
    println("---")
  }
}