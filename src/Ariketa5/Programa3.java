package Ariketa5;

import weka.classifiers.bayes.NaiveBayes;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.ObjectInputStream;

public class Programa3 {
    public static void main(String[] args) throws Exception{
        if(args.length == 0){
            System.out.println("\nHelburua: emandako partiketekin eta ereduarekin iragarpenak egitea.");
            System.out.println("Aurrebaldintzak:");
            System.out.println("1. argumentuan: iragartzeko .arff fitxategiaren path-a\n" +
                    "2. argumentuan: iragarpena egiteko .model eredua (NaiveBayes motakoa izan behar da" +
                    "3. argumentuan: iragarpenak gordetzeko fitxategiaren path-a");
        }
        else{
            String pathDev = args[0];
            DataSource devSource = new DataSource(pathDev);
            Instances devData = devSource.getDataSet();
            devData.setClassIndex(devData.numAttributes()-1);

            String pathModel = args[1];
            NaiveBayes nb = (NaiveBayes) SerializationHelper.read(pathModel);
            /*ObjectInputStream ois = new ObjectInputStream(new FileInputStream(pathModel));
            NaiveBayes nb = (NaiveBayes) ois.readObject();
            ois.close();*/

            Instances trainData = nb.getHeader();

            // train data atributu kopurua hartu
            int numAtt = trainData.numAttributes();

            // Hemen gordeko ditugu train dataren atributuen indizeak dev datan.
            int[] trainAttributes = new int[numAtt];

            // Gordetzen ditugu indizeak, atributuen izenen bidez
            for (int i = 0; i < numAtt; i++) {
                trainAttributes[i] = devData.attribute(trainData.attribute(i).name()).index();
            }

            Remove removeFilter = new Remove();
            removeFilter.setInvertSelection(true);
            removeFilter.setAttributeIndicesArray(trainAttributes);
            removeFilter.setInputFormat(devData);

            Instances newData = Filter.useFilter(devData, removeFilter);

            Instances iragarpenak = new Instances(devData); //printeas todos los atributos


            System.out.println("Iragarpenak egiten...");

            for (int i = 0; i < newData.numInstances(); i++) {
                double klasea = nb.classifyInstance(newData.instance(i));
                iragarpenak.instance(i).setClassValue(klasea);
            }

            String pathGorde = args[2];

            BufferedWriter writer = new BufferedWriter(new FileWriter(pathGorde));
            writer.write(iragarpenak.toString());
            writer.newLine();
            writer.flush();
            writer.close();

            System.out.println("Iragarpenak gorde dira");
        }
    }

}