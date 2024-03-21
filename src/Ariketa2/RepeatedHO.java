package Ariketa2;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;

import java.io.FileWriter;
import java.util.Random;

public class RepeatedHO {
    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            System.out.println("Repeated Hold-Out erabilera: <errepikapen kop> <data.arff>");
        }

        int kop = Integer.parseInt(args[0]);
        String dataPath = args[1];

        DataSource source = new DataSource(dataPath);
        Instances data = source.getDataSet();

        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }

        Evaluation cumulativeEvaluation = new Evaluation(data);

        for (int i = 1; i <= kop; i++) {
            System.out.println(i + ". buelta");
            data.randomize(new Random(i));

            RemovePercentage filter = new RemovePercentage();
            filter.setPercentage(33);
            filter.setInvertSelection(false);
            filter.setInputFormat(data);
            Instances train = Filter.useFilter(data, filter);
            System.out.println("Train %66: " + train.numInstances());

            filter.setInvertSelection(true);
            filter.setInputFormat(data);
            Instances test = Filter.useFilter(data, filter);
            System.out.println("Test %33: " + test.numInstances());

            NaiveBayes nb = new NaiveBayes();
            nb.buildClassifier(train);

            cumulativeEvaluation.evaluateModel(nb, test);
        }
        // Calcular y mostrar el promedio de las métricas de evaluación
        System.out.println("Resultado promedio después de " + kop + " repeticiones:");
        System.out.println(cumulativeEvaluation.toSummaryString());

        // Mostrar métricas detalladas como precisión, recall, F-measure, etc.
        System.out.println(cumulativeEvaluation.toClassDetailsString());
        System.out.println(cumulativeEvaluation.toMatrixString());
    }
}
