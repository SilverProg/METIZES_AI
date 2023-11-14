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
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.Random;

public class CustomNet {
    private int height;
    private int width;
    private int channels;
    private int batchSize;
    private int seed;
    private MultiLayerNetwork model;
    private DataSetIterator iterator;

    public CustomNet(int height, int width, int channels, int batchSize, int seed) {
        this.height = height;
        this.width = width;
        this.channels = channels;
        this.batchSize = batchSize;
        this.seed = seed;
    }

    public CustomNet(int height, int width) {
        this(height,width,3,12,1325);
    }

    public int getHeight() {
        return height;
    }

    public void setHeight(int height) {
        this.height = height;
    }

    public int getWidth() {
        return width;
    }

    public void setWidth(int width) {
        this.width = width;
    }

    public int getChannels() {
        return channels;
    }

    public void setChannels(int channels) {
        this.channels = channels;
    }

    public int getBatchSize() {
        return batchSize;
    }

    public void setBatchSize(int batchSize) {
        this.batchSize = batchSize;
    }

    public int getSeed() {
        return seed;
    }

    public void setSeed(int seed) {
        this.seed = seed;
    }

    public MultiLayerNetwork getModel() {
        return model;
    }

    public void setModel(MultiLayerNetwork model) {
        this.model = model;
    }

    public DataSetIterator getIterator() {
        return iterator;
    }

    public void setIterator(DataSetIterator iterator) {
        this.iterator = iterator;
    }

    public void createNetwork(int numClasses) {

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.ADAM)
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

        this.model =  model;
    }

    private void createNetwork1(int numClasses) {

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.ADAM)
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        .nIn(channels)
                        .nOut(16)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(numClasses)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutional(height, width, channels))
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(10));

        this.model = model;
    }

    public void createNetwork2(int numClasses) {

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
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

        this.model =  model;
    }

    public void getDataSetIterator(String folderPath) throws IOException, InterruptedException {

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

        iterator = dataSetIterator;
    }

    public void getDataSetIterator1(String folderPath) throws IOException, InterruptedException {

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

        iterator= dataSetIterator;
    }

    public void teachNetwork(int numEpochs) {

        for (int i = 0; i < numEpochs; i++) {
            while (iterator.hasNext()) {
                model.fit(iterator.next());
            }
            iterator.reset();
        }
    }

    public void saveModel(MultiLayerNetwork network, String url) throws IOException {
        ModelSerializer.writeModel(network, new File(url), true);
    }

    public MultiLayerNetwork restoreModel(String url) throws IOException {
        return ModelSerializer.restoreMultiLayerNetwork(new File(url));
    }

    public void testing(String url, MultiLayerNetwork model) throws IOException, InterruptedException {
        //iterator = getDataSetIterator(url);//error
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
