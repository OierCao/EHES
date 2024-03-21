package AriketaGehigarria;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomTree;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;

import java.io.FileWriter;
import java.util.Random;

public class getModel_and_Assesment {
    public static void main(String[] args) throws Exception {
        //-----------------------------[HASIERAKETAK]-----------------------------
        // - - - - - (Paths) - - - - -
        //in
        String testPath = args[0]; //data.arff
        //out
        String assesmentPath = args[1]; //assesment.txt
        String modelPath = args[2]; //RT.model

        //- - - - - (Data) - - - - -
        DataSource testSource= new DataSource(testPath);
        Instances data = testSource.getDataSet();
        data.setClassIndex(data.numAttributes()-1);

        //- - - - - (Out) - - - - -
        FileWriter fwKalitate = new FileWriter(assesmentPath);

        //---------------------------------[    ]--------------------------------




        //--------------------------------[GARAPENA]-------------------------------

        //--------------<DATU BANAKETA>--------------
        // - - - - - (Stratified Hold-Out) - - - - -
        //settings
        Resample rs = new Resample();
        rs.setSampleSizePercent(70);
        rs.setNoReplacement(true);
        //train
        rs.setInputFormat(data);
        Instances trainData = Filter.useFilter(data,rs);
        //dev
        rs.setInvertSelection(true);
        rs.setInputFormat(data);
        Instances devData = Filter.useFilter(data,rs);


        //--------------<DATUEI BURUZKO INFORMAZIOA>--------------
        // - - - - - (MaxClassIndex) - - - - -
        int maxMaiztasuna = Integer.MIN_VALUE;
        int maxClassIndex = 0;
        for (int x=0; x < data.classAttribute().numValues(); x++){
            int maiztasuna = data.attributeStats(data.classIndex()).nominalCounts[x];
            if (maiztasuna > maxMaiztasuna){
                maxMaiztasuna = maiztasuna;
                maxClassIndex = x;
            }
        }
        //-------------------------<   >--------------------------


        //--------------<PARAMETRO EKORKETA>--------------
        // - - - - - (RandomTree) - - - - -
        int kOpt = 1;
        double bestAccuracy = Double.MIN_VALUE;
        int jumpKop = 5;
        int kJump = data.numAttributes()/jumpKop;
        for (int k = 1; k <= data.numAttributes(); k++) {
            kJump = k*kJump;
            RandomTree randomTree = new RandomTree();
            randomTree.setKValue(kJump);
            randomTree.buildClassifier(trainData);

            //eval
            Evaluation eval = new Evaluation(devData);
            eval.evaluateModel(randomTree,devData);
            double accuracy = eval.precision(maxClassIndex);

            //balio optimoa
            if (accuracy > bestAccuracy) {
                bestAccuracy = accuracy;
                kOpt = kJump;
            }
        }
        System.out.println("kOpt:" + kOpt);
        //---------------------<   >-----------------------


        //--------------<BEZEROARENTZAKO ENTREGA>--------------
        // - - - - - (Kalitatea) - - - - -
        //modeloa eraiki
        RandomTree rtModel = new RandomTree();
        rtModel.setKValue(kOpt);
        rtModel.buildClassifier(data);

        //ebaluazio ez-zintzoa
        Evaluation evalEzZintzo = new Evaluation(data);
        evalEzZintzo.evaluateModel(rtModel,data);

        //repeated 3-fold cross validation
        int iterKop = 5;
        double kFoldPrecisionTot = 0;
        double KFoldRecallTot = 0;
        double KFoldFmeasureTot = 0;

        double[] precisionValues = new double[5];
        double[] recallValues = new double[5];
        double[] fMeasureValues = new double[5];
        for (int i = 0; i < iterKop; i++) {
            Evaluation evalKfold = new Evaluation(devData);
            evalKfold.crossValidateModel(rtModel,devData,3,new Random(i+1));
            //avg
            kFoldPrecisionTot += evalKfold.weightedPrecision();
            KFoldRecallTot += evalKfold.weightedRecall();
            KFoldFmeasureTot += evalKfold.weightedFMeasure();
            //desb
            precisionValues[i] = evalKfold.weightedPrecision();
            recallValues[i] = evalKfold.weightedRecall();
            fMeasureValues[i] = evalKfold.weightedFMeasure();
        }
        //avg//
        double kFoldPrecisionAvg = kFoldPrecisionTot/iterKop;
        double KFoldRecallAvg = KFoldRecallTot/iterKop;
        double KFoldFmeasureAvg = KFoldFmeasureTot/iterKop;
        //desb//
        double precisionVar = 0;
        double recallVar = 0;
        double fMeasureVar = 0;
        for (int i = 0; i < iterKop; i++) {
            precisionVar += Math.pow(precisionValues[i] - kFoldPrecisionAvg, 2);
            recallVar += Math.pow(recallValues[i] - KFoldRecallAvg, 2);
            fMeasureVar += Math.pow(fMeasureValues[i] - KFoldFmeasureAvg, 2);
        }
        double kFoldPrecisionDesb = Math.sqrt(precisionVar / iterKop);;
        double KFoldRecallDesb = Math.sqrt(recallVar / iterKop);;
        double KFoldFmeasureDesb = Math.sqrt(fMeasureVar / iterKop);


        //emaitzak gorde
        //kFold//
        fwKalitate.write("K-FOLD" + "\n");
        fwKalitate.write("PrecisionAvg: " + kFoldPrecisionAvg + "\n");
        fwKalitate.write("RecallAvg: " + KFoldRecallAvg + "\n");
        fwKalitate.write("F-measureAvg: " + KFoldFmeasureAvg + "\n");
        fwKalitate.write("PrecisionDesb: " + kFoldPrecisionDesb + "\n");
        fwKalitate.write("RecallDesb: " + KFoldRecallDesb + "\n");
        fwKalitate.write("F-measureDesb: " + KFoldFmeasureDesb + "\n");
        //ezZintzo//
        fwKalitate.write("\n" + "Ez ZINTZO" + "\n");
        fwKalitate.write("Precision: " + evalEzZintzo.weightedPrecision() + "\n");
        fwKalitate.write("Recall: " + evalEzZintzo.weightedRecall() + "\n");
        fwKalitate.write("F-measure: " + evalEzZintzo.weightedFMeasure() + "\n");


        // - - - - - (Modelo Entrega) - - - - -
        //modeloa gorde
        SerializationHelper.write(modelPath,rtModel);
        //---------------------<   >-----------------------

        //--------------------------------[ ]--------------------------------



        //fw itxi
        fwKalitate.close();
    }



}
