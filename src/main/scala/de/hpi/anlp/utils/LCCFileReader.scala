package de.hpi.anlp.utils

import scala.io.Source

/**
 * File reader to process data sets from the NLP institute of the university of Leipzig
 */
class LCCFileReader(fileName: String) extends Iterable[String] {
  /**
   * Every sentence is preceeded by its id. We are going to strip the id since we are only interested in the sentence 
   */
  val sentenceFileRx = "(?s)^[0-9]+\\s(.*)$" r

  override def iterator = new Iterator[String] {
    val lineIt = Source.fromFile(fileName).getLines
    var nextVal = readNextFromInput()
    var readLines = 0

    override def hasNext: Boolean = nextVal.isDefined

    override def next(): String = {
      nextVal match {
        case Some(value) =>
          nextVal = readNextFromInput()
          value
        case _ =>
          throw new NoSuchElementException("next on empty iterator")
      }
    }

    private def readNextFromInput(): Option[String] = {
      if (lineIt.hasNext) {
        // Read next sentence and strip the id from it
        val current = lineIt.next()
        current match {
          case sentenceFileRx(sentence) =>
            readLines += 1
            Some(sentence)
          case _ =>
            throw new Exception(s"Invalid line #$readLines in LCC file. Line content: '$current'")
        }
      }
      else
        None
    }
  }
}
