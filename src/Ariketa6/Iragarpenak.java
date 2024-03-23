package Ariketa6;

import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.FixedDictionaryStringToWordVector;
import weka.filters.unsupervised.instance.SparseToNonSparse;

import java.io.File;

public class Iragarpenak {
    public static void main(String[] args) throws Exception {

        String trainBoWPath = args[0];
        String testPath = args[1];
        String dictionaryPath = args[2];

        DataSource trainSource = new DataSource(trainBoWPath);
        Instances trainData = trainSource.getDataSet();
        if (trainData.classIndex() == -1) {
            trainData.setClassIndex(0);
        }

        DataSource testSource = new DataSource(testPath);
        Instances testData = testSource.getDataSet();
        if (testData.classIndex() == -1) {
            testData.setClassIndex(0);
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

        for (Instance ins:testNonSparse){
            ins.setClassValue(fc.classifyInstance(ins));
        }

        System.out.println("IRAGARPENAK" + "\n");
        System.out.println(testNonSparse.toString());
    }

}
