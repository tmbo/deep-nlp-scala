package de.hpi.anlp.utils

import org.apache.log4j.Logger
import scala.io.Source._
import scala.util.{Failure, Success, Try}

/**
 * Read and write content from and to a file
 */
object FileUtils {
  private val logger = Logger.getLogger("FileUtils")

  /**
   * Read the content of a file as a String
   */
  def read(toFile: String, className: String): Option[String] =
    Try(fromFile(toFile).mkString) match {
      case Success(content) =>
        Some(content)
      case Failure(e) =>
        logger.error(s"Reading $className failed. File $toFile", e)
        None
    }

  /**
   * Write the content into a file. The content is defined as a string.
   */
  def write(content: String, pathName: String, className: String): Boolean = {
    import java.io.PrintWriter

    var printWriter: Option[PrintWriter] = None
    var status = false
    Try {
      printWriter = Some(new PrintWriter(pathName))
      printWriter.map(_.write(content))
      status = true
    }
    match {
      // Catch and display exception description and return false
      case Failure(e) => {
        logger.error(s"$className.write failed for $pathName", e)

        if (printWriter != None) {
          Try(printWriter.map(_.close)) match {
            case Success(res) => res
            case Failure(e) =>
              logger.error(s"$className.write Failed for $pathName", e)
          }
        }
      }
      case Success(s) => {}
    }
    status
  }
}
