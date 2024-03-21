package Ariketa4;

import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.*;
import weka.core.converters.ConverterUtils.DataSource;

import java.util.Random;

public class ParametroEkorketa {
    public static void main(String[] args) throws Exception {

        String inputPath = args[0];

        DataSource source = new DataSource(inputPath);
        Instances data = source.getDataSet();

        if (data.classIndex() == -1){
            data.setClassIndex(data.numAttributes() - 1);
        }

        AttributeStats attrStats = data.attributeStats(data.numAttributes() - 1);
        System.out.println("Instantzia kop: " + data.numInstances());

        int max = -1;
        double freqMax = 0;
        System.out.println(data.numClasses());

        for(int i=0; i<data.numClasses(); i++){
            System.out.println("i: " + i + "; izena: " + data.attribute(data.numAttributes()-1).value(i) + "; maiztasuna: " + attrStats.nominalCounts[i]);
            if (freqMax < attrStats.nominalCounts[i]){
                max = i;
                freqMax = attrStats.nominalCounts[i];
            }
        }

        System.out.println("\nAgeriena i: " + max + "; izena: " + data.attribute(data.numAttributes()-1).value(max) + "; maiztasuna: " + freqMax + "\n");


        int kant = 1;
        int kOP = 0;
        DistanceFunction distOP = null;
        double preciosionOP = 0;
        String etiketaOP = "";

        DistanceFunction[] fncs = new DistanceFunction[5];
        fncs[0] = new EuclideanDistance();
        fncs[1] = new ChebyshevDistance();
        fncs[2] = new ManhattanDistance();
        fncs[3] = new FilteredDistance();
        fncs[4] = new MinkowskiDistance();

        IBk ibk = new IBk();

        for (int k=1; k <= data.numInstances()/2; k++){
            ibk.setKNN(k);

            for (DistanceFunction dist: fncs){
                ibk.getNearestNeighbourSearchAlgorithm().setDistanceFunction(dist);

                for (int w=0; w < ibk.TAGS_WEIGHTING.length; w++){
                    ibk.setDistanceWeighting(new SelectedTag(ibk.TAGS_WEIGHTING[w].getID(), ibk.TAGS_WEIGHTING));
                    Evaluation eval = new Evaluation(data);
                    eval.crossValidateModel(ibk, data, 3, new Random());
                    System.out.println(kant + ": k =" + k + "; dist: " + dist.getClass() + "; etiketa: " + ibk.TAGS_WEIGHTING[w].getID());

                    if (eval.precision(0) > preciosionOP){
                        kOP = k;
                        distOP = dist;
                        etiketaOP = ibk.TAGS_WEIGHTING[w].getReadable();
                        preciosionOP = eval.precision(0);
                    }
                    kant++;
                }
            }
        }

        System.out.println("\n\nParametroen ekorketa amaitu da:");
        System.out.println("    K optimoa: " + kOP);
        System.out.println("    Distantzia optimoa: " + distOP.getClass());
        System.out.println("    Etiketa optimoa: " + etiketaOP);
        System.out.println("    Precision optimoa: " + preciosionOP);

    }
}
