package de.hpi.anlp.utils

trait Model {
  /**
   * Write the model parameters associated to this object into a file
   */
  protected def write(content: String): Boolean =
    FileUtils.write(content, Model.RELATIVE_PATH, getClass.getSimpleName)

  /**
   * This operation or method has to be overwritten for a model to be saved into a file
   */
  def saveToFile: Boolean =
    false
}

object Model {
  private val RELATIVE_PATH = "models/"

  /**
   * Read this model parameters from a file defined as in file `classname`
   */
  def read(className: String): Option[String] =
    FileUtils.read(RELATIVE_PATH, className)
}