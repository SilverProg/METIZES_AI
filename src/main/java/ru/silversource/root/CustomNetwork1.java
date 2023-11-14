package ru.silversource.root;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.listener.impl.LogRecordListener;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.schema.Schema;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.CombinedMultiDataSetPreProcessor;
import org.deeplearning4j.datasets.iterator.CombinedPreProcessor;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
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
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;

public class CustomNetwork1 {
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
                        .nOut(32)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(2, new DenseLayer.Builder()
                        .nOut(128)
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
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(2, new DenseLayer.Builder()
                        .nOut(256)
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

    public static DataSetIterator getDataSetIterator(String folderPath, String urlCSV) throws IOException, InterruptedException {
        int height = 300;
        int width = 400;
        int channels = 3; // 3 для RGB изображений
        int batchSize = 8; // Размер пакета данных

        File csvFile = new File(urlCSV);
        Schema schema = new Schema.Builder().addColumnInteger("mark").build();

        RecordReader labelsRR = new CSVRecordReader(1,',');
        labelsRR.initialize(new FileSplit(csvFile));

        RecordReaderDataSetIterator labelsIterator = new RecordReaderDataSetIterator(labelsRR,batchSize,0,2);
        DataNormalization scalar = new NormalizerStandardize();
        scalar.fit(labelsIterator);
        labelsIterator.setPreProcessor(scalar);


        // Определите объект FileSplit для указанной папки
        File parentDir = new File(folderPath);
        FileSplit fileSplit = new FileSplit(parentDir, NativeImageLoader.ALLOWED_FORMATS);

        // Создайте ImageRecordReader и настройте его
        ImageRecordReader imageRecordReader = new ImageRecordReader(height, width, channels, new ParentPathLabelGenerator());
        imageRecordReader.initialize(fileSplit);
        imageRecordReader.setListeners(new LogRecordListener()); // Логирование для отладки

        RecordReaderDataSetIterator imageIterator = new RecordReaderDataSetIterator(imageRecordReader,batchSize,1,2);
        scalar.fit(imageIterator);
        imageIterator.setPreProcessor(scalar);

        CombinedPreProcessor preProcessor = new CombinedPreProcessor.Builder().build();
        //CombinedMultiDataSetPreProcessor multiDataSetPreProcessor = new MultipleEpochsIterator(10, labelsIterator,imageIterator);

        // Создайте итератор на основе ImageRecordReader
        DataSetIterator dataSetIterator = new org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator(
                imageRecordReader, batchSize, 1, 2); // 1 - индекс меток, 2 - количество классов

        return dataSetIterator;
    }

    public static MultiLayerNetwork teachNetwork(MultiLayerNetwork multiLayerNetwork, DataSetIterator iterator, int numEpochs) {

        for(int i = 0; i < numEpochs; i++) {
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
        DataSetIterator iterator = getDataSetIterator(url,url);

        Evaluation evaluation = new Evaluation(2);
        while (iterator.hasNext()) {
            org.nd4j.linalg.dataset.DataSet dataSet = iterator.next();
            System.out.println(iterator.getLabels());
            INDArray features = dataSet.getFeatures();
            INDArray labels = dataSet.getLabels();
            INDArray predicted = model.output(features, false);

            evaluation.eval(labels,predicted);
        }
        System.out.println(evaluation.stats());
    }
}
