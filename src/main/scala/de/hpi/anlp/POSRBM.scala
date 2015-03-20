package de.hpi.anlp

import java.io.File
import de.hpi.anlp.conll.AnnotatedToken
import de.hpi.anlp.utils.Word2VecDataSetIterator
import org.apache.commons.math3.random.MersenneTwister
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.models.featuredetectors.rbm.RBM
import org.deeplearning4j.models.word2vec.Word2Vec
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.layers.factory.LayerFactories
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.text.inputsanitation.InputHomogenization
import org.deeplearning4j.text.sentenceiterator.{SentencePreProcessor, FileSentenceIterator}
import org.deeplearning4j.text.tokenization.tokenizerfactory.UimaTokenizerFactory
import org.deeplearning4j.util.SerializationUtils
import org.nd4j.linalg.api.activation.Activations
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.nd4j.linalg.netlib.SimpleNetlibBlas

/**
 * Configure, train and evaluate RBM based POS taggers 
 */
object POSRBM {
  // Size of the unsupervised news corpus to use. Should be either 1M, 10K or 300K
  val vecTrainSize = "1M"

  // Window size to train on
  val windowSize = 5

  // Word vector size used during the word2vec training
  val wordVecLayers = 50

  /**
   * Load a word2vec model from disc
   */
  def loadWordVectorModel() = {
    SerializationUtils.readObject(new File(s"output/word2vec_$vecTrainSize.model")).asInstanceOf[Word2Vec]
  }

  /**
   * Load a neural network from disc
   */
  def loadNeuralNetwork(fileName: String) = {
    SerializationUtils.readObject(new File(fileName)).asInstanceOf[MultiLayerNetwork]
  }

  /**
   * Store a word2vec instance to disc for later retrieval
   */
  def storeWordVectorModel(model: Word2Vec) = {
    SerializationUtils.saveObject(model, new File(s"output/word2vec_$vecTrainSize.model"));
  }

  /**
   * Instanciate a default sentence preprocessor and apply standart input homogenization
   */
  private def sentencePreprocessor = new SentencePreProcessor() {
    val sentenceFileRx = "(?s)^[0-9]+\\s(.*)$" r

    override def preProcess(sentenceLine: String): String = {
      sentenceLine match {
        case sentenceFileRx(sentence) =>
          new InputHomogenization(sentence).transform()
        case _ =>
          throw new Exception("Invalid input line.")
      }
    }
  }

  /**
   * Train a new word2vec model on the given news corpus
   */
  private def trainWordVectorModel() = {
    val file = new File(s"assets/deu_news_2010_$vecTrainSize-text/deu_news_2010_$vecTrainSize-sentences.txt")

    val sentenceIterator = new FileSentenceIterator(sentencePreprocessor, file)

    val t = new UimaTokenizerFactory()
    val vec = new Word2Vec.Builder()
      .minWordFrequency(5)
      .windowSize(windowSize)
      .layerSize(wordVecLayers)
      .iterate(sentenceIterator)
      .tokenizerFactory(t)
      .build()

    vec.fit()
    vec
  }

  /**
   * Train a RBM on the training data set either using an existing word2vec model or creating a new one. THis is the 
   * place to configure the RBM 
   */
  def train(trainDocuments: Iterable[List[AnnotatedToken]], states: List[String]) = {
    val vec = trainWordVectorModel()
    storeWordVectorModel(vec)
    //    val vec = loadWordVectorModel()

    println("Finished Word2Vec!")

    printf("Sim('fernsehen', 'familie') = %f\n", vec.similarity("fernsehen", "familie"));

    val fetcher = new Word2VecDataSetIterator(vec, trainDocuments, states, batch = 10)
    val gen = new MersenneTwister(123);

    val layerFactory = LayerFactories.getFactory(classOf[RBM])
    val conf = new NeuralNetConfiguration.Builder()
      .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
      .iterations(100)
      .rng(gen)
      .weightInit(WeightInit.NORMALIZED)
      .learningRate(0.001f)
      .nIn(wordVecLayers * windowSize)
      .nOut(states.size)
      .lossFunction(LossFunctions.LossFunction.MCXENT)
      .visibleUnit(RBM.VisibleUnit.SOFTMAX)
      .hiddenUnit(RBM.HiddenUnit.RECTIFIED)
      .layerFactory(layerFactory)
      .list(2)
      .`override`(new NeuralNetConfiguration.ConfOverride() {
      override def `override`(i: Int, builder: NeuralNetConfiguration.Builder) {
        if (i == 1) {
          builder.weightInit(WeightInit.ZERO);
          builder.activationFunction(Activations.softMaxRows());
        }
      }
    })
      .hiddenLayerSizes(50)
      .build()

    val network = new MultiLayerNetwork(conf)

    println("Started fitting network...")

    network.fit(fetcher)

    println("Finished fitting Network!")

    SerializationUtils.saveObject(network, new File(s"output/network_$vecTrainSize.model6"))

    (network, vec)
  }


  private def labelForArray(a: INDArray, statesIndex: Map[Int, String]) = {
    val m = SimpleNetlibBlas.iamax(a)
    statesIndex(m)
  }

  /**
   * Evaluate a given RBM model on the test data set and its gold standard. Beside the implemented evaluation of
   * POSEvaluator this will also execute the model specific evaluation implemented in the dl4j library.
   */
  def evaluate(network: MultiLayerNetwork, testDocuments: Iterable[List[AnnotatedToken]], vec: Word2Vec, states: List[String]) = {

    println("Started evaluating Network!")

    val testData = new Word2VecDataSetIterator(vec, testDocuments, states, batch = 20000).next()
    val predicted = network.output(testData.getFeatureMatrix)

    val statesIndex = states.zipWithIndex.map {
      case (el, i) => i -> el
    }.toMap

    val evaluator = new POSEvaluator(states)
    val buildInEval = new Evaluation()

    val predictedLabels: Seq[String] = (0 until predicted.length()).map { i =>
      val guessRow: INDArray = predicted.getRow(i)
      labelForArray(guessRow, statesIndex)
    }

    val goldLabels = (0 until testData.numExamples()).map { i =>
      val currRow: INDArray = testData.getLabels.getRow(i)
      labelForArray(currRow, statesIndex)
    }

    evaluator.add(predictedLabels, goldLabels)
    buildInEval.eval(testData.getLabels, predicted)

    System.out.println(buildInEval.stats())

    evaluator
  }

}
