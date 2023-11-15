package ru.silversource.root;


import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.IOException;

public class Main {

    public static void main(String[] args) throws IOException, InterruptedException {

        MultiLayerNetwork multiLayerNetwork = CustomNetwork.createNetwork2(2);

        String folderPathGroup1 = "C:/HEAP/ТТ_AI/met_min";

        DataSetIterator iterator_1 = CustomNetwork.getDataSetIterator(folderPathGroup1);

        multiLayerNetwork = CustomNetwork.teachNetwork(multiLayerNetwork, iterator_1, 5);
        //multiLayerNetwork.clear();

        CustomNetwork.saveTrainedNetwork(multiLayerNetwork, "net17.zip");

        System.out.println("start");
        System.out.println("COMPLETE");
    }
}
