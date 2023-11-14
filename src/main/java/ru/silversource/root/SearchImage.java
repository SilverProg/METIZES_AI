package ru.silversource.root;


import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class SearchImage {

    public static void main(String[] args) throws IOException, InterruptedException {

        String folderPathTest= "C:/HEAP/ТТ_AI/METIZES/set2/test/test (61).jpg";

        File imageFile = new File(folderPathTest);
        BufferedImage bufferedImage = javax.imageio.ImageIO.read(imageFile);
        NativeImageLoader loader = new NativeImageLoader(300,400,3);


        MultiLayerNetwork multiLayerNetwork = CustomNetwork.restoreModel("C:/HEAP/ТТ_AI/METIZES/nets/net17.zip");

        INDArray imageINDArray = loader.asMatrix(bufferedImage);

        INDArray output = multiLayerNetwork.output(imageINDArray,false);

        System.out.println(output);
        System.out.println("COMPLETE");
    }
}
