package AriketaGehigarria;//[WEKA]
import weka.filters.supervised.attribute.AttributeSelection;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.supervised.instance.Resample;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.BestFirst;
import weka.classifiers.functions.SMO;
import weka.classifiers.Evaluation;
import weka.filters.Filter;
import weka.core.*;

//[JAVA]
import java.io.FileWriter;
import java.util.Random;

public class Extra {

    public static void main(String[] args) throws Exception {
        //-----------------------------[HASIERAKETAK]-----------------------------
        // - - - - - (Paths) - - - - -
        //in
        String dataPath = args[0];
        String testPath = args[1];
        //out
        String modelPath = args[2];
        //txt
        String dataInfoPath = args[3];
        String kalitatePath = args[4];

        //- - - - - (Data) - - - - -
        DataSource source= new DataSource(dataPath);
        Instances data = source.getDataSet();
        data.setClassIndex(data.numAttributes()-1);
        //- - - - - (Test) - - - - -
        DataSource testSource= new DataSource(testPath);
        Instances testData = testSource.getDataSet();
        testData.setClassIndex(testData.numAttributes()-1);

        //- - - - - (Out) - - - - -
        FileWriter fwInfo = new FileWriter(dataInfoPath);
        FileWriter fwKalitate = new FileWriter(kalitatePath);
        //---------------------------------[    ]--------------------------------



        //--------------------------------[GARAPENA]-------------------------------

        //¿¿¿¿¿egin dezakegu zuzenean atributu filtraketa datu guztiei?????'

        //--------------<DATU FILTRAKETA>--------------
        // - - - - - (Attribute Selection) - - - - -
        //settings
        AttributeSelection as = new AttributeSelection();
        as.setEvaluator(new CfsSubsetEval());
        as.setSearch(new BestFirst());
        //filter
        as.setInputFormat(data);
        Instances filteredData = Filter.useFilter(data,as);
        //ikusi aldaketak
        for (int a=0; a < data.numAttributes(); a++){
            fwInfo.write("Attribute: " + data.attribute(a).name());
        }
        for (int a=0; a < filteredData.numAttributes(); a++){
            fwInfo.write("Attribute: " + filteredData.attribute(a).name());
        }
        //-------------------<   >-------------------


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

        // - - - - - (DevData-ren klasea = ?) - - - - -
        for (int i=0; i<devData.size();i++){
            devData.instance(i).setMissing(data.numAttributes() - 1);
        }
        //-------------------<   >-------------------


        //--------------<DATUEI BURUZKO INFORMAZIOA>--------------
        // - - - - - (MinClassIndex) - - - - -
        int minMaiztasuna = Integer.MAX_VALUE;
        int minClassIndex = 0;
        for (int x=0; x < data.classAttribute().numValues(); x++){
            int maiztasuna = data.attributeStats(data.classIndex()).nominalCounts[x];
            if (maiztasuna < minMaiztasuna){
                minMaiztasuna = maiztasuna;
                minClassIndex = x;
            }
        }
        //-------------------------<   >--------------------------


        //--------------<PARAMETRO EKORKETA>--------------
        // - - - - - (SMO) - - - - -
        double fMeasureMaxKernel = Double.MIN_VALUE;
        int bestEx = 1;
        for (int ex=1; ex < 4; ex++){
            //smo eraiki
            PolyKernel kernel = new PolyKernel();
            kernel.setExponent(ex);
            SMO smo = new SMO();
            smo.setKernel(kernel);
            smo.buildClassifier(trainData);

            //eval
            Evaluation eval = new Evaluation(devData);
            eval.evaluateModel(smo,devData,3,new Random(1));
            double fMeasureKernel = eval.fMeasure(minClassIndex);

            //balio optimoak
            if (fMeasureKernel > fMeasureMaxKernel){
                fMeasureMaxKernel= fMeasureKernel;
                bestEx = ex;
            }
        }
        //---------------------<   >-----------------------


        //--------------<BEZEROARENTZAKO ENTREGA>--------------
        // - - - - - (Kalitatea) - - - - -
        //modeloa eraiki
        SMO smoKalitate= new SMO();
        PolyKernel kernelKalitate = new PolyKernel();
        kernelKalitate.setExponent(bestEx);
        smoKalitate.setKernel(kernelKalitate);
        smoKalitate.buildClassifier(trainData); //¿¿¿build clasiffier datu guztiekin egin daiteke????
        //ebaluazioa
        Evaluation evalKalitate = new Evaluation(testData);
        evalKalitate.evaluateModel(smoKalitate,testData,3,new Random(1));
        //emaitzak gorde
        fwKalitate.write(evalKalitate.toMatrixString());
        fwKalitate.write(evalKalitate.toSummaryString());

        // - - - - - (Modelo Entrega) - - - - -
        //modealoa eraiki
        SMO smoFinal = new SMO();
        PolyKernel kernelFinal = new PolyKernel();
        kernelFinal.setExponent(bestEx);
        smoFinal.setKernel(kernelFinal);
        smoFinal.buildClassifier(data); //datu guztiekin entrenatu
        //modeloa gorde
        SerializationHelper.write(modelPath,smoFinal);
        //---------------------<   >-----------------------

        //--------------------------------[ ]--------------------------------




        //-----------------------------[AMAIERA]-----------------------------
        // - - - - - (Writer itxi) - - - - -
        fwInfo.close();
        fwKalitate.close();
        //--------------------------------[ ]--------------------------------
    }
}
