package Ariketa2;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class StratiffiedHO {
    public static void main(String[] args) throws Exception {
        String dataTest = args[0];
        String dataTrain = args[1];

        DataSource sourceTrain = new DataSource(dataTrain);
        Instances train = sourceTrain.getDataSet();
        if (train.classIndex() == -1)
            train.setClassIndex(train.numAttributes() - 1);

        System.out.println(train.numInstances());

        DataSource sourceTest = new DataSource(dataTest);
        Instances test = sourceTest.getDataSet();
        if (test.classIndex() == -1)
            test.setClassIndex(train.numAttributes() - 1);

        System.out.println(test.numInstances());

        NaiveBayes nb = new NaiveBayes();
        nb.buildClassifier(train);

        Evaluation eval = new Evaluation(train);
        eval.evaluateModel(nb, test);

        System.out.println(eval.toMatrixString());
    }
}
