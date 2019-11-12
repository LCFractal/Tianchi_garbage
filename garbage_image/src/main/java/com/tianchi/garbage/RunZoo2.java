package com.tianchi.garbage;

import com.alibaba.tianchi.garbage_image_util.ImageClassSink;
import com.alibaba.tianchi.garbage_image_util.ImageDirSource;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;

import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.util.Map;

public class RunZoo2 {
    public static void main(String[] args) throws Exception {
        //String modelPath = System.getenv("IMAGE_MODEL_PATH");
        String modelPath = System.getenv("IMAGE_MODEL_PACKAGE_PATH");
        boolean ifReverseInputChannels = true;
        int[] inputShape = {1, 224, 224, 3};
        float[] meanValues = {123.68f, 116.78f, 103.94f};
        float scale = 1.0f;
        String input = "input_1";

        System.out.println((String.format("load tar model: %s",modelPath)));

        long fileSize = new File(modelPath).length();
        InputStream inputStream = new FileInputStream(modelPath);
        byte[] savedModelBytes = new byte[(int)fileSize];
        inputStream.read(savedModelBytes);

        ClassIndex.InitTable();
        Map<Integer, String> labelTable = ClassIndex.getTable();

        StreamExecutionEnvironment flinkEnv = StreamExecutionEnvironment.getExecutionEnvironment();
        flinkEnv.setParallelism(1);
        ImageDirSource source = new ImageDirSource();
        flinkEnv.addSource(source).setParallelism(1)
                .flatMap(new ImageProcessing(inputShape)).setParallelism(1)
                .flatMap(new ModelPredictionMapFunction(savedModelBytes,inputShape,ifReverseInputChannels,meanValues,scale,
                        input, labelTable)).setParallelism(1)
                .addSink(new ImageClassSink()).setParallelism(1);
        flinkEnv.execute();
    }
}

