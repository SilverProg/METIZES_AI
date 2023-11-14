package ru.silversource.root;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.listener.impl.LogRecordListener;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ConvolutionUtils;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.Random;

public class CustomNetwork {
    public static MultiLayerNetwork createNetwork(int numClasses) {
        int height = 300;
        int width = 400;
        int channels = 3; // 3 для RGB изображений
        int seed = 123;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(org.deeplearning4j.nn.conf.Updater.ADAM)
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        .nIn(channels)
                        .nOut(16)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(3, 3)
                        .stride(2, 2)
                        .build())
                .layer(2, new DenseLayer.Builder()
                        .nOut(64)
                        .activation(Activation.RELU)
                        .build())
                .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(numClasses)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutional(height, width, channels))
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(10));

        return model;
    }

    public static MultiLayerNetwork createNetwork1(int numClasses) {
        int height = 300;
        int width = 400;
        int channels = 3; // 3 для RGB изображений
        int seed = 12356;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(org.deeplearning4j.nn.conf.Updater.ADAM)
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        .nIn(channels)
                        .nOut(16)
                        .activation(Activation.RELU)
                        .build())
                /*.layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())*/
                /*.layer(2, new DenseLayer.Builder()
                        .nOut(256)
                        .activation(Activation.RELU)
                        .build())*/
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(numClasses)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutional(height, width, channels))
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(10));

        return model;
    }

    public static MultiLayerNetwork createNetwork2(int numClasses) {
        int height = 300;
        int width = 400;
        int channels = 3; // 3 для RGB изображений
        int seed = 12356;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS)
                .list()
                .layer(0, new ConvolutionLayer.Builder()
                        .nIn(channels)
                        .nOut(8)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(1, new SubsamplingLayer.Builder()
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(2, new ConvolutionLayer.Builder()
                        .nOut(128)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(3, new SubsamplingLayer.Builder()
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(4, new DenseLayer.Builder()
                        .nOut(64)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(5, new OutputLayer.Builder()
                        .nOut(numClasses)
                        .activation(Activation.SOFTMAX)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .setInputType(InputType.convolutionalFlat(height, width, channels)) // Устанавливаем тип входных данных
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(10));

        return model;
    }

    public static DataSetIterator getDataSetIterator(String folderPath) throws IOException, InterruptedException {
        int height = 300;
        int width = 400;
        int channels = 3; // 3 для RGB изображений
        int batchSize = 12; // Размер пакета данных
        // Определите объект FileSplit для указанной папки
        File parentDir = new File(folderPath);
        FileSplit fileSplit = new FileSplit(parentDir, NativeImageLoader.ALLOWED_FORMATS);

        // Создайте ImageRecordReader и настройте его
        ImageRecordReader imageRecordReader = new ImageRecordReader(height, width, channels, new ParentPathLabelGenerator());
        imageRecordReader.initialize(fileSplit);
        imageRecordReader.setListeners(new LogRecordListener()); // Логирование для отладки

        // Создайте итератор на основе ImageRecordReader
        DataSetIterator dataSetIterator = new org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator(
                imageRecordReader, batchSize, 1, 2); // 1 - индекс меток, 2 - количество классов

        return dataSetIterator;
    }

    public static DataSetIterator getDataSetIterator1(String folderPath) throws IOException, InterruptedException {
        int height = 300;
        int width = 400;
        int channels = 3; // 3 для RGB изображений
        int batchSize = 8; // Размер пакета данных
        // Определите объект FileSplit для указанной папки
        File parentDir = new File(folderPath);
        FileSplit fileSplit = new FileSplit(parentDir, NativeImageLoader.ALLOWED_FORMATS, new Random(123));

        // Создайте ImageRecordReader и настройте его
        ImageRecordReader imageRecordReader = new ImageRecordReader(height, width, channels, new ParentPathLabelGenerator());
        imageRecordReader.initialize(fileSplit);
        imageRecordReader.setListeners(new LogRecordListener()); // Логирование для отладки

        // Создайте итератор на основе ImageRecordReader
        DataSetIterator dataSetIterator = new org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator(
                imageRecordReader, batchSize, 1, 2); // 1 - индекс меток, 2 - количество классов

        return dataSetIterator;
    }

    public static MultiLayerNetwork teachNetwork(MultiLayerNetwork multiLayerNetwork, DataSetIterator iterator, int numEpochs) {

        for (int i = 0; i < numEpochs; i++) {
            while (iterator.hasNext()) {
                //System.out.println(iterator.getLabels() /*+ "::" + iterator.totalExamples()*/);
                multiLayerNetwork.fit(iterator.next());
            }
            iterator.reset();
        }

        return multiLayerNetwork;

    }

    public static void saveTrainedNetwork(MultiLayerNetwork network, String url) throws IOException {
        ModelSerializer.writeModel(network, new File(url), true);
    }

    public static MultiLayerNetwork restoreModel(String url) throws IOException {
        return ModelSerializer.restoreMultiLayerNetwork(new File(url));
    }

    public static void testingNetwork(String url, MultiLayerNetwork model) throws IOException, InterruptedException {
        DataSetIterator iterator = getDataSetIterator(url);
        int i = 0;
        Evaluation evaluation = new Evaluation(2);
        while (iterator.hasNext()) {
            org.nd4j.linalg.dataset.DataSet dataSet = iterator.next();
            System.out.println(iterator.getLabels() + "::" + i++);
            INDArray features = dataSet.getFeatures();
            INDArray labels = dataSet.getLabels();
            INDArray predicted = model.output(features, false);

            evaluation.eval(labels, predicted);
        }
        System.out.println(evaluation.stats());
    }


}
