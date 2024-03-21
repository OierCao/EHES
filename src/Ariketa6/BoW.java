package Ariketa6;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
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
            trainData.setClassIndex(0);

        // Sortu eta konfiguratu StringToWordVector filtroa
        StringToWordVector filter = new StringToWordVector();
        filter.setOutputWordCounts(true);
        filter.setWordsToKeep(1000);
        filter.setLowerCaseTokens(true);
        filter.setDictionaryFileToSaveTo(new File(dictionaryPath));
        filter.setInputFormat(trainData);

        Instances train = Filter.useFilter(trainData, filter);

        //Sparsetik Non Sparsera bihurtu
        SparseToNonSparse nonSparse = new SparseToNonSparse();
        nonSparse.setInputFormat(train);

        Instances dataNonSparse = Filter.useFilter(train, nonSparse);

        //Gorde .arff -a
        ArffSaver saver = new ArffSaver();
        saver.setFile(new File(gordePath));
        saver.setInstances(dataNonSparse);
        saver.writeBatch();
    }
}
