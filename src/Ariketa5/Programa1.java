package Ariketa5;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.instance.StratifiedRemoveFolds;

import java.io.FileWriter;

public class Programa1 {
    public static  void main(String[] args) throws Exception {

        if (args.length < 3) {
            System.out.println("Uso: java StratifiedSplit <ruta_data.arff> <ruta_train.arff> <ruta_test_blind.arff>");
            return;
        }

        String dataPath = args[0];
        String dataTrain = args[1];
        String dataTest = args[2];

        // Datuak kargatu
        DataSource source = new DataSource(dataPath);
        Instances data = source.getDataSet();

        // Klase atributua ezarri, normalean azkena izaten da
        if (data.classIndex() == -1)
            data.setClassIndex(data.numAttributes() - 1);

        // StratifiedRemoveFolds filtroa konfiguratu
        StratifiedRemoveFolds filter = new StratifiedRemoveFolds();

        // Entrenamendu multzoa sortu
        filter.setNumFolds(10);
        filter.setFold(7);
        filter.setInvertSelection(true);
        filter.setInputFormat(data);

        Instances train = Filter.useFilter(data, filter);

        // Test multzoa sortu
        filter.setInvertSelection(false); // Inbertitu aukeraketa, test multzorako zatia lortzeko
        filter.setInputFormat(data);

        Instances test = Filter.useFilter(data, filter);

        //Gorde artxiboak (falta)
        FileWriter fwTest = new FileWriter(dataTest);
        fwTest.write(test.toString());
        fwTest.close();

        FileWriter fwTrain = new FileWriter(dataTrain);
        fwTrain.write(train.toString());
        fwTrain.close();
    }
}
