package AriketaGehigarria;

import weka.classifiers.trees.RandomTree;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;

import java.io.FileWriter;

public class makePredictions {
    public static void main(String[] args) throws Exception {
        //-----------------------------[HASIERAKETAK]-----------------------------
        // - - - - - (Paths) - - - - -
        //in
        String modelPath = args[0]; //RT.model
        String testPath = args[1]; //test.arff
        //out
        String testIragarpenak = args[2]; //test_predictions.arff

        //- - - - - (Model) - - - - -
        RandomTree rtModel = (RandomTree) SerializationHelper.read(modelPath);

        //- - - - - (Test) - - - - -
        DataSource testSource= new DataSource(testPath);
        Instances testData = testSource.getDataSet();
        testData.setClassIndex(testData.numAttributes()-1);

        FileWriter fw = new FileWriter(testIragarpenak);

        //---------------------------------[    ]--------------------------------



        //-----------------------------[IRAGARPENAK]-----------------------------
        System.out.println("HASIERAKO DATUAK" + "\n");
        System.out.println(testData.toString());

        //predictions
        for (Instance ins:testData){
            ins.setClassValue(rtModel.classifyInstance(ins));
        }

        System.out.println("IRAGARPENAK" + "\n");
        System.out.println(testData.toString());
        //gorde
        fw.write(testData.toString());
        fw.close();
        //---------------------------------[    ]--------------------------------
    }



}
