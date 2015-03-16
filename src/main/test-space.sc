import java.io.File
import java.net.{URL, URLClassLoader}

import org.apache.commons.io.IOUtils
import org.springframework.core.io.ClassPathResource

def addPath(s: String){
  val f = new File(s)     
  println(f.exists())
  val u = f.toURI()
  val urlClassLoader = ClassLoader.getSystemClassLoader().asInstanceOf[URLClassLoader]
  val urlClass = classOf[URLClassLoader]
  val method = urlClass.getDeclaredMethod("addURL", classOf[URL])
  method.setAccessible(true)
  method.invoke(urlClassLoader, u.toURL())
}

addPath("/Users/tombocklisch/Documents/Studium/ANLP/deep-nlp-scala/src/main/resources")

import java.io.File

import edu.stanford.nlp.tagger.maxent.MaxentTagger
import org.apache.commons.math3.random.MersenneTwister
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator
import org.deeplearning4j.distributions.Distributions
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.models.featuredetectors.rbm.RBM
import org.deeplearning4j.models.word2vec.Word2Vec
import org.deeplearning4j.nn.conf.{MultiLayerConfiguration, NeuralNetConfiguration}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.text.inputsanitation.InputHomogenization
import org.deeplearning4j.text.sentenceiterator.{SentencePreProcessor, FileSentenceIterator}
import org.deeplearning4j.text.tokenization.tokenizerfactory.UimaTokenizerFactory
import org.nd4j.linalg.api.activation.Activations
import org.nd4j.linalg.lossfunctions.LossFunctions

println("running")
val sample = "This is a sample text."
val tagged = new MaxentTagger("edu/stanford/nlp/models/pos-tagger/english-left3words/english-left3words-distsim.tagger")
def reuters = {
  val reutersFile = "assets/reuters21578/"
  val file = new File(reutersFile)

  new FileSentenceIterator(new SentencePreProcessor() {

    override def preProcess(sentence: String): String =
      new InputHomogenization(sentence).transform()

  },file)

}

val iter = reuters
val t = new UimaTokenizerFactory()
val vec = new Word2Vec.Builder()
  .windowSize(5)
  .layerSize(300)
  .iterate(iter)
  .tokenizerFactory(t)
  .build()

vec.fit()

val oil = "oil"

printf("%f\n", vec.similarity(oil, oil))

printf("%f\n", vec.similarity(oil, "fish"));


def deep() = {
  val gen = new MersenneTwister(123);
  val conf = new NeuralNetConfiguration.Builder()
    .hiddenUnit(RBM.HiddenUnit.RECTIFIED)
    .momentum(5e-1f) //this expresses decimals as floats. Remember e?
    .visibleUnit(RBM.VisibleUnit.GAUSSIAN)
    .regularization(true)
    .dist(Distributions.uniform(gen))
    .activationFunction(Activations.tanh())
    .iterations(10000)
    .weightInit(WeightInit.DISTRIBUTION)
    .lossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY)
    .rng(gen)
    .learningRate(1e-3f)
    .nIn(4)
    .nOut(3)
    .build()
  val d = new MultiLayerNetwork(conf.asInstanceOf[MultiLayerConfiguration])
  val iter = new IrisDataSetIterator(150, 150);
  val next = iter.next();
  next.normalizeZeroMeanZeroUnitVariance();
  next.shuffle();
  val testAndTrain = next.splitTestAndTrain(110);
  val train = testAndTrain.getTrain();
  d.fit(train);
  val test = testAndTrain.getTest();

  val eval = new Evaluation();
  val output = d.output(test.getFeatureMatrix());
  eval.eval(test.getLabels(),output);
  println("Score " + eval.stats());
}