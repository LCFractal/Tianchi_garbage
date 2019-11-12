package com.tianchi.garbage;

import com.alibaba.tianchi.garbage_image_util.ImageClassSink;
import com.alibaba.tianchi.garbage_image_util.ImageDirSource;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;

import java.util.Map;

public class RunZoo3 {
    public static void main(String[] args) throws Exception {
        String modelPath = System.getenv("MODEL_INFERENCE_PATH")+"/SavedModel";
        boolean ifReverseInputChannels = true;
        int[] inputShape = {1, 380, 380, 3};
        //float[] meanValues = {123.68f, 116.78f, 103.94f};
        float[] meanValues = {127.5f, 127.5f, 127.5f};
        //float[] meanValues = {103.94f, 116.78f, 123.68f};
        //float[] meanValues = {0f, 0f, 0f};
        float scale = 127.5f;
        String input = "input_1";

        ClassIndex.InitTable();
        Map<Integer, String> labelTable = ClassIndex.getTable();

        StreamExecutionEnvironment flinkEnv = StreamExecutionEnvironment.getExecutionEnvironment();
        flinkEnv.setParallelism(1);
        ImageDirSource source = new ImageDirSource();
        flinkEnv.addSource(source).setParallelism(1)
                .flatMap(new ModelPredictionMapFunction2(modelPath,inputShape,ifReverseInputChannels,meanValues,scale,
                        input,labelTable)).setParallelism(1)
                .addSink(new ImageClassSink()).setParallelism(1);
                //.addSink(new ClassSink()).setParallelism(1);
        flinkEnv.execute();
    }
}

