/*
 * Copyright (C) 20011-2014 Scalable minds UG (haftungsbeschränkt) & Co. KG. <http://scm.io>
 */
package de.hpi.anlp

import scala.io.Source

class LCCFileReader(fileName: String) extends Iterable[String] {
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
