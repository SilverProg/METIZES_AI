package ru.silversource.root;


import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import java.io.IOException;

public class Test {

    public static void main(String[] args) throws IOException, InterruptedException {

        String folderPathGroup1 = "C:/HEAP/ТТ_AI/met_min";
        String folderPathTest= "C:/HEAP/ТТ_AI/METIZES/set_test_1/hack";


       MultiLayerNetwork multiLayerNetwork = CustomNetwork.restoreModel("C:/HEAP/ТТ_AI/METIZES/nets/net17.zip");

        CustomNetwork.testingNetwork(folderPathTest, multiLayerNetwork);

        System.out.println("COMPLETE");
    }
}
