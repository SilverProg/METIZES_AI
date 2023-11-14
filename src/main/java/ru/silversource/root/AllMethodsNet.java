package ru.silversource.root;

import java.io.IOException;

public class AllMethodsNet {

    public static void main(String[] args) throws IOException, InterruptedException {
        CustomNet net = new CustomNet(30,40,3,12,123);
        net.createNetwork2(2);
        net.getDataSetIterator("C:/HEAP/ТТ_AI/met_min");
        net.teachNetwork(5);
        System.out.println("COMPLETE");
    }
}
