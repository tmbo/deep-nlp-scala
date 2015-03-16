package de.hpi;
        
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.movingwindow.Window;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;

public class WindowConverter {
    public static double[] asExample(Window window,Word2Vec vec) {
        int length = vec.lookupTable().layerSize();
        List<String> words = window.getWords();
        int windowSize = window.getWindowSize();

        double[] example = new double[ length * windowSize];
        int count = 0;
        for(int i = 0; i < words.size(); i++) {
            String word = words.get(i);
            INDArray n = vec.getWordVectorMatrixNormalized(word);
            INDArray vec2 = n == null ? vec.getWordVectorMatrix(Word2Vec.UNK) : vec.getWordVectorMatrix(word);
            if(vec2 == null)
                vec2 = vec.getWordVectorMatrix(Word2Vec.UNK);
            for(int j = 0; j < vec2.length(); j++) {
                example[count++] = vec2.getDouble(j);
            }


        }

        return example;
    }

    public static INDArray asExampleMatrix(Window window,Word2Vec vec) {
        return Nd4j.create(asExample(window, vec));
    }

}
