package Ariketa3;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Random;

public class Eredua {
    public static  void main(String[] args) throws Exception {
        if (args.length != 3){
            System.err.println("Erabilera: java Eredua <data.arff> <model.model> <output.txt>");
            System.exit(1);
        }
        String dataPath = args[0];
        String modelPath = args[1];
        String outputPath = args[2];

        DataSource source = new DataSource(dataPath);
        Instances data = source.getDataSet();
        if (data.classIndex()==-1){
            data.setClassIndex(data.numAttributes()-1);
        }

        NaiveBayes nb = new NaiveBayes();
        nb.buildClassifier(data);

        //Modeloa gorde
        SerializationHelper.write(modelPath, nb);

        //CrossValidation
        Evaluation evaluationCV = new Evaluation(data);
        evaluationCV.crossValidateModel(nb, data, 5, new Random(1));

        //HoldOut
        Instances dataRandom = new Instances(data);
        dataRandom.randomize(new Random(1)); //Data randomizatu

        int trainSize = (int)Math.round(data.numInstances()*0.7); //%70 izateko behar diren datu kantitatea
        int testSize = data.numInstances() - trainSize;
        Instances train = new Instances(dataRandom, 0, trainSize);
        Instances test = new Instances(dataRandom, trainSize, testSize);

        Evaluation evaluationHO = new Evaluation(train);
        evaluationHO.evaluateModel(nb, test);

        //Idatzi dokumentua
        BufferedWriter writer = new BufferedWriter(new FileWriter(outputPath));
        SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
        String date = dateFormat.format(new Date());

        writer.write("Egikaritze data: " + date + "\n");
        writer.write("Argumentuak: Data path: " + dataPath + ", Model path: " + modelPath + ", Evaluation path: " + outputPath + "\n\n");
        writer.write("K-fold cross-validation nahasmen matrizea:\n" + evaluationCV.toMatrixString() + "\n");
        writer.write("Hold-out nahasmen matrizea (%70 train, %30 test):\n" + evaluationHO.toMatrixString());
        writer.close();
    }
}
