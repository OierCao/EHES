package Ariketa1;

import weka.core.Attribute;
import weka.core.AttributeStats;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Ariketa1 {
    public static  void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.out.println("Mesedez, sartu .arff fitxategi baten path-a lehenengo argumentu gisa.");
            return;
        }
        String path = args[0];
        DataSource source = new DataSource(path);
        Instances data = source.getDataSet();

        if (data.classIndex() == -1){
            data.setClassIndex(data.numAttributes()-1);
        }

        System.out.println("Path-a: " + path);
        System.out.println("Instantzia kopurua: " + data.numInstances());
        System.out.println("Atributu kopurua: " + data.numAttributes());

        System.out.println();

        AttributeStats firstAttributeStats = data.attributeStats(0);
        System.out.println("Lehenengo atributuak har ditzakeen balio ezberdinak: " + firstAttributeStats.distinctCount);

        System.out.println();

        Attribute lastAttribute = data.attribute(data.numAttributes()-1);
        AttributeStats lastStats = data.attributeStats(data.numAttributes()-1);
        int i=0;
        while (lastAttribute.numValues()>i){
            System.out.println(data.classAttribute().value(i) +" balioa "+lastStats.nominalCounts[i]+ " instatzientan agertu da");
            i++;
        }
        int minIndex=0;
        i=1;
        while (lastAttribute.numValues()>i){
            if (lastStats.nominalCounts[i]<lastStats.nominalCounts[i-1]) {
                minIndex=i;
            }
            i++;
        }
        System.out.println();
        System.out.println(data.classAttribute().value(minIndex) +" balioa maiztasun txikiena daukana da, "+lastStats.nominalCounts[minIndex]+ " instatzientan agertu da");
        System.out.println();

        AttributeStats azkenaurreAttributeStats = data.attributeStats(data.numAttributes()-2);
        System.out.println("Azken aurreko atributuaren missing value-ak: "+ azkenaurreAttributeStats.missingCount);

    }
}

