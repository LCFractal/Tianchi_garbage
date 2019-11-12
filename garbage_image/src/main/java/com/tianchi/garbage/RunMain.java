package com.tianchi.garbage;

import com.alibaba.tianchi.garbage_image_util.*;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;

import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;

public class RunMain {
    public static void main(String[] args) throws Exception {
        String modelPath = System.getenv("IMAGE_MODEL_PACKAGE_PATH");
        boolean ifReverseInputChannels = true;
        int[] inputShape = {1, 224, 224, 3};
        float[] meanValues = {123.68f, 116.78f, 103.94f};
        float scale = 1.0f;
        String input = "input_1";
        /*
        long fileSize = new File(modelPath).length();
        InputStream inputStream = new FileInputStream(modelPath);
        byte[] savedModelBytes = new byte[(int)fileSize];
        inputStream.read(savedModelBytes);
        */
        StreamExecutionEnvironment flinkEnv = StreamExecutionEnvironment.getExecutionEnvironment();
        flinkEnv.setParallelism(1);
        ImageDirSource source = new ImageDirSource();
        flinkEnv.addSource(source).setParallelism(1)
                .flatMap(new ImageTest()).setParallelism(1)
                .addSink(new ImageClassSink()).setParallelism(1);
        flinkEnv.execute();
    }
}