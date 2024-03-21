package Ariketa3;

import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;

import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileWriter;

public class Iragarpenak {
    public static  void main(String[] args) throws Exception {
        if (args.length != 3){
            System.err.println("Erabilera: java Iragarpenak <model.model> <test.arff> <output.txt>");
            System.exit(1);
        }
        String modelPath = args[0];
        String testPath = args[1];
        String outputPath = args[2];

        NaiveBayes nb = (NaiveBayes) SerializationHelper.read(new FileInputStream(modelPath));

        DataSource source = new DataSource(testPath);
        Instances data = source.getDataSet();
        if (data.classIndex()==-1){
            data.setClassIndex(data.numAttributes()-1);
        }

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(outputPath))) {
            for (int i = 0; i < data.numInstances(); i++) {
                Instance instance = data.instance(i);
                double label = nb.classifyInstance(instance);
                String prediction = data.classAttribute().value((int) label);
                writer.write("Instancia " + (i + 1) + ": Iragarpena = " + prediction + "\n");
            }
        }
    }
}
