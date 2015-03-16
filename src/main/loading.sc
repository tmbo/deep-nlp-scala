import java.io.FileReader

import edu.stanford.nlp.ling.{CoreAnnotations, CoreLabel}
import edu.stanford.nlp.sequences.{SeqClassifierFlags, CoNLLDocumentReaderAndWriter}

val conllreader = {
  val r = new CoNLLDocumentReaderAndWriter()
  r.init(new SeqClassifierFlags())
  r
}
val it = conllreader.getIterator(new FileReader("/Users/tombocklisch/Documents/Studium/ANLP/deep-nlp-scala/assets/de-train.tt"))
var numDocs = 0
var numTokens = 0
var lastAnsBase = ""
var numEntities = 0
while (it.hasNext) {
  val doc = it.next()
  numDocs += 1
  import scala.collection.JavaConversions._
  for (fl <- doc) {
    if (fl.word != "XX") {
      val ans: String = fl.get(classOf[CoreAnnotations.AnswerAnnotation])
      var ansBase: String = null
      var ansPrefix: String = null
      val bits: Array[String] = ans.split("-")
      if (bits.length == 1) {
        ansBase = bits(0)
        ansPrefix = ""
      }
      else {
        ansBase = bits(1)
        ansPrefix = bits(0)
      }
      numTokens += 1
      if (!(ansBase == "O")) {
        if (ansBase == lastAnsBase) {
          if (ansPrefix == "B") {
            numEntities += 1
          }
        }
        else {
          numEntities += 1
        }
      }
    }
  }
}