package Ariketa6;

import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.tokenizers.WordTokenizer;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.filters.unsupervised.instance.SparseToNonSparse;

import java.io.File;

public class BoW {

    public static void main(String[] args) throws Exception {

        String data = args[0];
        String gordePath = args[1];
        String dictionaryPath = args[2];

        // Kargatu entrenamendu datuak
        DataSource trainSource = new DataSource(data);
        Instances trainData = trainSource.getDataSet();

        if (trainData.classIndex() == -1)
            trainData.setClassIndex(trainData.numAttributes()-1);

        // Sortu eta konfiguratu StringToWordVector filtroa
        StringToWordVector filter = new StringToWordVector();
        filter.setOutputWordCounts(false);
        filter.setWordsToKeep(1000);
        filter.setLowerCaseTokens(true);
        filter.setDictionaryFileToSaveTo(new File(dictionaryPath));

        //Tokenizer
        WordTokenizer tokenizer = new WordTokenizer();
        tokenizer.setDelimiters(".,;:'\"()?!\n -");
        filter.setTokenizer(tokenizer);

        filter.setInputFormat(trainData);

        Instances train = Filter.useFilter(trainData, filter);

        //Sparsetik Non Sparsera bihurtu
        SparseToNonSparse nonSparse = new SparseToNonSparse();
        nonSparse.setInputFormat(train);

        Instances dataNonSparse = Filter.useFilter(train, nonSparse);

        //Attribute Selection
        AttributeSelection selector = new AttributeSelection();
        selector.setEvaluator(new CfsSubsetEval());
        selector.setSearch(new BestFirst());
        selector.setInputFormat(dataNonSparse);
        Instances newData = Filter.useFilter(dataNonSparse, selector);

        //Gorde .arff -a
        ArffSaver saver = new ArffSaver();
        saver.setFile(new File(gordePath));
        saver.setInstances(newData);
        saver.writeBatch();
    }
}