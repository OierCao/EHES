package Ariketa6;

import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.FixedDictionaryStringToWordVector;
import weka.filters.unsupervised.instance.SparseToNonSparse;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;

public class Iragarpenak {
    public static void main(String[] args) throws Exception {

        String trainBoWPath = args[0];
        String testPath = args[1];
        String dictionaryPath = args[2];

        DataSource trainSource = new DataSource(trainBoWPath);
        Instances trainData = trainSource.getDataSet();
        if (trainData.classIndex() == -1) {
            trainData.setClassIndex(trainData.numAttributes()-1);
        }

        DataSource testSource = new DataSource(testPath);
        Instances testData = testSource.getDataSet();
        if (testData.classIndex() == -1) {
            testData.setClassIndex(testData.numAttributes()-1);
        }

        FixedDictionaryStringToWordVector fixedFilter = new FixedDictionaryStringToWordVector();
        fixedFilter.setDictionaryFile(new File(dictionaryPath));
        fixedFilter.setInputFormat(testData);

        Instances filterTestData = Filter.useFilter(testData, fixedFilter);

        SparseToNonSparse nonSparse = new SparseToNonSparse();
        nonSparse.setInputFormat(filterTestData);
        Instances testNonSparse = Filter.useFilter(filterTestData, nonSparse);


        FilteredClassifier fc = new FilteredClassifier();
        fc.buildClassifier(trainData);

        /*for (Instance ins:testNonSparse){
            ins.setClassValue(fc.classifyInstance(ins));
        }*/

        System.out.println("IRAGARPENAK" + "\n");

        BufferedWriter writer = new BufferedWriter(new FileWriter(new File("klaseerreala.txt")));
        int count_ondo=0;
        int count_txarto=0;
        int instantzia_Kop=testNonSparse.numInstances();
        for (int i = 0; i < testNonSparse.numInstances(); i++) {
            double clsLabel = fc.classifyInstance(testNonSparse.instance(i));
            double actualClassValue = testNonSparse.instance(i).classValue();
            if (clsLabel == actualClassValue){
                count_ondo++;
                System.out.println("Ondo"+count_ondo);
            }
            else{
                count_txarto++;
                System.out.println("Txarto"+count_txarto);
            }
            writer.write(testNonSparse.instance(i).toString() + ",' " + testNonSparse.classAttribute().value((int) clsLabel)+"'");
            writer.newLine();
        }

        writer.close();

        System.out.println("Ondo predikatutakoak: " + count_ondo);
        System.out.println("Txarto predikatutakoak: " + count_txarto);
        System.out.println("Guztira instantziak: " + instantzia_Kop);

        // Cast the numerator or the denominator to double before division
        double percentage = ((double) count_ondo / instantzia_Kop) * 100;

        System.out.println("Zehaztasuna (%): " + String.valueOf(percentage));


    }

}