package de.hpi.anlp

import java.io.File
import java.util
import de.hpi.anlp.hmm.{ConstantSmoothedHMM, HMM}
import edu.stanford.nlp.tagger.maxent.MaxentTagger
import org.apache.commons.math3.random.MersenneTwister
import org.deeplearning4j.datasets.test.TestDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.models.featuredetectors.rbm.RBM
import org.deeplearning4j.models.rntn.RNTN
import org.deeplearning4j.models.rntn.RNTN.Builder
import org.deeplearning4j.models.word2vec.Word2Vec
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.Configuration.IntegerRanges
import org.deeplearning4j.nn.conf.{DeepLearningConfigurable, Configuration, NeuralNetConfiguration}
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ConfOverride
import org.deeplearning4j.nn.layers.factory.LayerFactories
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.stepfunctions.GradientStepFunction
import org.deeplearning4j.plot.Tsne
import org.deeplearning4j.scaleout.actor.core.DefaultModelSaver
import org.deeplearning4j.scaleout.actor.runner.DeepLearning4jDistributed
import org.deeplearning4j.scaleout.api.statetracker.StateTracker
import org.deeplearning4j.scaleout.job.{DataSetIteratorJobIterator, JobIterator, JobIteratorFactory}
import org.deeplearning4j.scaleout.perform._
import org.deeplearning4j.scaleout.perform.models.word2vec.Word2VecPerformer
import org.deeplearning4j.scaleout.statetracker.hazelcast.HazelCastStateTracker
import org.deeplearning4j.text.corpora.treeparser.{TreeParser, TreeVectorizer}
import org.deeplearning4j.text.inputsanitation.InputHomogenization
import org.deeplearning4j.text.sentenceiterator.{SentencePreProcessor, FileSentenceIterator}
import org.deeplearning4j.text.tokenization.tokenizerfactory.UimaTokenizerFactory
import org.deeplearning4j.util.SerializationUtils
import org.nd4j.linalg.api.activation.Activations
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.nd4j.linalg.netlib.SimpleNetlibBlas

object Main extends App {
  
  val vecTrainSize = "1M"

//  val vecTrainSize = "10K"
  
  val windowSize = 5
  
  val wordVecLayers = 50

  val states = List("NOUN", "ADV", "PRT", ".", "ADP", "DET", "PRON", "VERB", "X", "NUM", "CONJ", "ADJ")

  val trainDocuments = new ConLLFileReader("assets/de-train.tt")

  val testDocuments = new ConLLFileReader("assets/de-eval.tt")
  
  def evaluateHMM(hmm: HMM): Unit = {
    val trainedHMM = hmm.train(trainDocuments)

    val evaluator = new POSEvaluator(states)

    testDocuments.foreach{ sentence =>
      val unannotated = sentence.map(_.token)
      val (prob, tags) = trainedHMM.mostProbablePath(unannotated)
      //      fileWriter.write(unannotated, tags)
      evaluator.add(tagged = tags, gold = sentence.map(_.tag))
    }

    println("\n--- Evaluation of: " + hmm.getClass.getCanonicalName)

    evaluator.printEvaluation()

    println("---")
  }

  def evaluateAllHMMs(): Unit = {
    evaluateHMM(new ConstantSmoothedHMM(states, smoothingConstant = 1))
    
    evaluateHMM(new HMM(states))
  }
  
  def trainWordVectorModel() = {
    val file = new File(s"assets/deu_news_2010_$vecTrainSize-text/deu_news_2010_$vecTrainSize-sentences.txt")

    val sentenceIterator = new FileSentenceIterator(new SentencePreProcessor() {
      val sentenceFileRx = "(?s)^[0-9]+\\s(.*)$" r
      override def preProcess(sentenceLine: String): String = {
        sentenceLine match{
          case sentenceFileRx(sentence) =>
            new InputHomogenization(sentence).transform()
          case _ =>
            throw new Exception("Invalid input line.")
        }
      }
    }, file)


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

  def loadWordVectorModel() = {
    SerializationUtils.readObject(new File(s"output/word2vec_$vecTrainSize.model")).asInstanceOf[Word2Vec]
  }

  def loadNeuralNetwork(fileName: String) = {
    SerializationUtils.readObject(new File(fileName)).asInstanceOf[MultiLayerNetwork]
  }
  
  def storeWordVectorModel(model: Word2Vec) = {
    SerializationUtils.saveObject(model, new File(s"output/word2vec_$vecTrainSize.model"));
  }
  
  def LCCReader() = {
//    val vec = trainWordVectorModel()
//    storeWordVectorModel(vec)
    val vec = loadWordVectorModel()
    
    println("Finished Word2Vec!")

    printf("Sim('fernsehen', 'familie') = %f\n", vec.similarity("fernsehen","familie"));

    val fetcher = new Word2VecDataSetIterator(vec, trainDocuments, states, batch = 10)
    val gen = new MersenneTwister(123);
    
    val layerFactory = LayerFactories.getFactory(classOf[RBM])
    val conf = new NeuralNetConfiguration.Builder()
      .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
      .iterations(10)
      .rng(gen)
      .weightInit(WeightInit.DISTRIBUTION)
      .stepFunction(new GradientStepFunction())
      .learningRate(0.01f)
      .nIn(wordVecLayers * windowSize)
      .nOut(states.size)
      .visibleUnit(RBM.VisibleUnit.SOFTMAX)
      .hiddenUnit(RBM.HiddenUnit.RECTIFIED)
      .layerFactory(layerFactory)
      .list(2)
      .hiddenLayerSizes(50)
      .build()

//    val json = conf.toJson()
    
//    val stateTracker = new HazelCastStateTracker

//    val distributedConf = new Configuration()
//    distributedConf.set(DeepLearningConfigurable.MULTI_LAYER_CONF,json)
//    distributedConf.set(DeepLearningConfigurable.CLASS,classOf[RBM].getName());
//    distributedConf.set(WorkerPerformerFactory.WORKER_PERFORMER,classOf[MultilayerNetworkPerformerFactory].getName);
//    val iter: JobIterator = new DataSetIteratorJobIterator(fetcher)

    //run the master
//    val runner = new DeepLearning4jDistributed(iter, stateTracker)
//    runner.setup(distributedConf)
//    val modelSaver = new DefaultModelSaver(new File("output/nn-model.bin"))
//    runner.setModelSaver(modelSaver)
//    runner.train()

    val network = new MultiLayerNetwork(conf)
    

    println("Started fitting network...")
    
    network.fit(fetcher)
    
    println("Finished fitting Network!")
    
//    val network = modelSaver.load(classOf[MultiLayerNetwork])
//
    SerializationUtils.saveObject(network, new File(s"output/network_$vecTrainSize.model6"))

    val testData = new Word2VecDataSetIterator(vec, testDocuments, states, batch = 20000).next()

//    val actualLabels = 

    val predicted = network.output(testData.getFeatureMatrix)

    val eval = new Evaluation()

    println("Started evaluating Network!")

    eval.eval(testData.getLabels,predicted)

    System.out.println(eval.stats());

    println("Finished evaluating Network!")
  }
  
  def evaluateNetwork(fileName: String) = {
    val network = loadNeuralNetwork(fileName)
    val vec = loadWordVectorModel()
    val testData = new Word2VecDataSetIterator(vec, testDocuments, states, batch = 20000).next()
    val predicted = network.output(testData.getFeatureMatrix)
    val evaluator = new POSEvaluator(states)
    
    val labelIndex = states.zipWithIndex.map{
      case (el, i) => i -> el
    }.toMap
    
    val predictedLabels: Seq[String] = (0 until predicted.length()).map{ i =>
      val guessRow: INDArray = predicted.getRow(i)
      val guessMax = SimpleNetlibBlas.iamax(guessRow)
      labelIndex(guessMax)
    }
    
    val goldLabels = (0 until testData.numExamples()).map{ i =>
      val currRow: INDArray = testData.getLabels.getRow(i)
      val currMax = SimpleNetlibBlas.iamax(currRow)
      labelIndex(currMax)
    }

    evaluator.add(predictedLabels, goldLabels)

    println("\n--- Evaluation of: " + fileName)

    evaluator.printEvaluation()

    println("---")
  }

//  evaluateNetwork("output/network_10K.model1")
  LCCReader()
  
}

class NeuralNetworkPerformerFactory extends BaseWorkPerformerFactory {

  def instantiate: WorkerPerformer = {
    new NeuralNetWorkPerformer()
  }
}

class MultilayerNetworkPerformerFactory extends BaseWorkPerformerFactory{
  def instantiate: BaseMultiLayerNetworkWorkPerformer = {
    new BaseMultiLayerNetworkWorkPerformer
  }
  
}
