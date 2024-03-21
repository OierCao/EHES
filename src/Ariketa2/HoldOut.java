package Ariketa2;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.Evaluation;
import weka.filters.unsupervised.instance.RemovePercentage;
import weka.filters.Filter;

import java.util.Random;
import java.io.FileWriter;
import java.text.SimpleDateFormat;
import java.util.Date;

public class HoldOut {
    public static void main(String[] args) throws Exception {
        // Zure datuen eta irteeraren path-a hartu
        String dataPath = "1.PraktikaDatuak-20240121/heart-c.arff";
        String outputPath = "hola.txt";

        // Datuak kargatu
        DataSource source = new DataSource(dataPath);
        Instances data = source.getDataSet();
        if (data.classIndex() == -1)
            data.setClassIndex(data.numAttributes() - 1);

        data.randomize(new Random(1));

        // Datuak banatu: %66 entrenamendurako eta gainontzekoa test-rako
        // Entrenamendu multzoa sortu (66%)
        RemovePercentage filter = new RemovePercentage();
        filter.setPercentage(66);
        filter.setInvertSelection(true);
        filter.setInputFormat(data);
        Instances train = Filter.useFilter(data, filter);

        // Test multzoa sortu (gainontzeko %34)
        filter.setInvertSelection(false);
        filter.setInputFormat(data);
        Instances test = Filter.useFilter(data, filter);

        // Naive Bayes eredua sortu eta entrenatu
        NaiveBayes nb = new NaiveBayes();
        nb.buildClassifier(train);

        System.out.println("OK");

        // Ebaluazioa
        Evaluation eval = new Evaluation(train);
        eval.evaluateModel(nb, test);

        // Emaitzak gordetzeko
        try (FileWriter writer = new FileWriter(outputPath)) {
            writer.write("Exekuzio data: " + new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new Date()) + "\n");
            writer.write("Argumentuak: Data path=" + dataPath + ", Output path=" + outputPath + "\n\n");
            writer.write("Nahasmen matrizea:\n" + eval.toMatrixString() + "\n");
        }
    }
}
