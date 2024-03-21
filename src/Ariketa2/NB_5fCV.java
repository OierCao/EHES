package Ariketa2;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Random;

public class NB_5fCV {
    public static void main(String[] args) throws Exception{
        if (args.length != 2){
            System.err.println("Erabilera: java NaiveBayesEvaluator <input.arff> <output.txt>");
            System.exit(1);
        }
        String inputPath = args[0];
        String outputPath = args[1];

        DataSource source = new DataSource(inputPath);
        Instances data = source.getDataSet();
        if (data.classIndex()==-1){
            data.setClassIndex(data.numAttributes()-1);
        }
        int k = 5;

        ArffSaver arffSaver = new ArffSaver();
        arffSaver.setInstances(data);
        arffSaver.setFile(new File(""));

        NaiveBayes nb = new NaiveBayes();
        Evaluation evaluation = new Evaluation(data);

        evaluation.crossValidateModel(nb, data, k, new Random(1));
        nb.buildClassifier(data);

        try(BufferedWriter writer = new BufferedWriter(new FileWriter(outputPath))) {
            SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
            String date = dateFormat.format(new Date());

            String info = "Exekuzio data: " + date +
                    "\nHasierako path-a:" + inputPath +
                    "\nAmaierako path-a: " + outputPath +
                    "\nEbaluazio emaitzak: " + getEbaluazioEmaitzak(evaluation) + evaluation.toClassDetailsString();

            writer.write(info);

            System.out.println("Emaitzak gorde dira");
        } catch (IOException e){e.printStackTrace();}
    }

    public static String getEbaluazioEmaitzak(Evaluation evaluation) throws Exception {
        try {
            System.out.println(evaluation.toSummaryString());
            System.out.println(evaluation.toMatrixString());
            return evaluation.toSummaryString() + "\n" + evaluation.toMatrixString();
        }
        catch (Exception e){
            e.printStackTrace();
            return e.toString();
        }
    }
}
