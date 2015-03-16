/*
 * Copyright (C) 20011-2014 Scalable minds UG (haftungsbeschränkt) & Co. KG. <http://scm.io>
 */
package de.hpi.anlp

import java.io.{PrintWriter, File}

import scala.io.Source

case class AnnotatedToken(token: String, tag: String)

class ConLLFileWriter(fileName: String) {
  var openedWriter: Option[PrintWriter] = Some(new PrintWriter(new File(fileName)))

  def write(sentence: Seq[String], annotations: Seq[String]): Boolean = {
    openedWriter.map { writer =>
      sentence.zip(annotations).map {
        case (word, annotation) =>
          writer.println(word + "\t" + annotation)
      }
      writer.println("") // add an empty line to complete the sentence

      true
    } getOrElse false
  }

  def close() = {
    openedWriter.map { writer =>
      writer.flush()
      writer.close()
    }
    openedWriter = None
  }
}

class ConLLFileReader(fileName: String) extends Iterable[List[AnnotatedToken]] {
  override def iterator = new Iterator[List[AnnotatedToken]] {
    val lineIt = Source.fromFile(fileName).getLines
    var nextVal = readNextFromInput()
    var readLines = 0

    override def hasNext: Boolean = nextVal.isDefined

    override def next(): List[AnnotatedToken] = {
      nextVal match {
        case Some(value) =>
          nextVal = readNextFromInput()
          value
        case _ =>
          throw new NoSuchElementException("next on empty iterator")
      }
    }

    private def readNextFromInput(): Option[List[AnnotatedToken]] = {
      val currentTokens = lineIt.takeWhile(_.trim != "")
      if (currentTokens.isEmpty)
        None
      else {
        val cs = currentTokens.toList
        val annotated = cs.map { current =>
          readLines += 1
          current.split('\t') match {
            case Array(token, label) =>
              AnnotatedToken(token, label)
            case _ =>
              throw new Exception(s"Invalid line #$readLines in ConLL file. Line content: '$current'")
          }
        }
        readLines += 1
        Some(annotated)
      }
    }
  }
}
