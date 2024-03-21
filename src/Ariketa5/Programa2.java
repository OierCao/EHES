package Ariketa5;

import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;

public class Programa2 {
    public static void main(String[] args) throws Exception {

        String inputPath = args[0];
        String modeloa = args[1];

        DataSource source = new DataSource(inputPath);
        Instances data = source.getDataSet();

        if (data.classIndex() == -1){
            data.setClassIndex(data.numAttributes() - 1);
        }

        AttributeSelection selector = new AttributeSelection();
        selector.setEvaluator(new CfsSubsetEval());
        selector.setSearch(new BestFirst());
        selector.setInputFormat(data);
        Instances newData = Filter.useFilter(data, selector);

        NaiveBayes nb = new NaiveBayes();
        nb.buildClassifier(newData);

        SerializationHelper.write(modeloa, nb);
    }
}
