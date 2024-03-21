package Ariketa2;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.instance.StratifiedRemoveFolds;
import java.io.FileWriter;

public class StratiffiedBanaketa {
    public static  void main(String[] args) throws Exception {
        String dataPath = args[0];
        String dataTrain = args[1];
        String dataTest = args[2];

        // Datuak kargatu
        DataSource source = new DataSource(dataPath);
        Instances data = source.getDataSet();

        // Klase atributua ezarri, normalean azkena izaten da
        if (data.classIndex() == -1)
            data.setClassIndex(data.numAttributes() - 1);

        System.out.println("data: "+ data.numInstances());

        // StratifiedRemoveFolds filtroa konfiguratu
        StratifiedRemoveFolds filter = new StratifiedRemoveFolds();

        // Entrenamendu multzoa sortu
        filter.setNumFolds(10); // 3 zatitan banatu, %66 entrenamendurako
        filter.setFold(7); // Lehenengo zatia hartu entrenamendurako
        filter.setInvertSelection(true); // Inbertitu aukeraketa, entrenamendurako zatia lortzeko
        filter.setInputFormat(data);

        Instances train = Filter.useFilter(data, filter);
        System.out.println("Train: "+ train.numInstances());

        // Test multzoa sortu

        filter.setInvertSelection(false); // Inbertitu aukeraketa, test multzorako zatia lortzeko
        filter.setInputFormat(data);
        Instances test = Filter.useFilter(data, filter);
        System.out.println("Test: "+ test.numInstances());

        //Gorde artxiboak (falta)
        FileWriter fwTest = new FileWriter(dataTest);
        fwTest.write(test.toString());
        fwTest.close();

        FileWriter fwTrain = new FileWriter(dataTrain);
        fwTrain.write(train.toString());
        fwTrain.close();
    }
}
