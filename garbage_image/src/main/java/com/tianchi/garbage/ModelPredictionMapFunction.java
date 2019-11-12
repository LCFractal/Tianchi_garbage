package com.tianchi.garbage;

import com.alibaba.tianchi.garbage_image_util.IdLabel;
import com.intel.analytics.bigdl.transform.vision.image.opencv.OpenCVMat;
import com.intel.analytics.zoo.feature.image.OpenCVMethod;
import com.intel.analytics.zoo.pipeline.inference.JTensor;
import org.apache.flink.api.common.functions.RichFlatMapFunction;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.util.Collector;
import com.intel.analytics.zoo.pipeline.inference.OpenVinoInferenceSupportive;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.framework.ConfigProto;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.net.URL;
import java.util.*;

public class ModelPredictionMapFunction extends RichFlatMapFunction<Tuple2<String, JTensor>, IdLabel> {
    private ImageModel model;
    private Map<Integer, String> labelTable;
    private byte[] modelByte;
    private String modelPath;
    private int[] inputShape;
    private boolean ifReverseInputChannels;
    private float[] meanValues;
    private float scale;
    private String input;
    private int cropWidth;
    private int cropHeight;
    private int[] outShape;

    ModelPredictionMapFunction(byte[] modelByte, int[] inputShape, boolean ifReverseInputChannels,
                               float[] meanValues, float scale, String input, Map<Integer, String> labelTable) {
        this.modelByte = modelByte;
        this.modelPath = null;
        this.inputShape = inputShape;
        this.ifReverseInputChannels = ifReverseInputChannels;
        this.meanValues = meanValues;
        this.scale = scale;
        this.input = input;
        this.labelTable = labelTable;
        this.cropWidth = inputShape[1];
        this.cropHeight = inputShape[2];
        this.outShape = inputShape;
    }

    ModelPredictionMapFunction(String modelPath, int[] inputShape, boolean ifReverseInputChannels,
                               float[] meanValues, float scale, String input, Map<Integer, String> labelTable) {
        this.modelByte = null;
        this.modelPath = modelPath;
        this.inputShape = inputShape;
        this.ifReverseInputChannels = ifReverseInputChannels;
        this.meanValues = meanValues;
        this.scale = scale;
        this.input = input;
        this.labelTable = labelTable;
        this.cropWidth = inputShape[1];
        this.cropHeight = inputShape[2];
        this.outShape = inputShape;
    }

    @Override
    public void flatMap(Tuple2<String, JTensor> value, Collector<IdLabel> out) throws Exception {
        List<JTensor> data = Arrays.asList(value.f1);
        List<List<JTensor>> inputs = new ArrayList<>();
        inputs.add(data);
        float[] outputData = model.doPredict(inputs).get(0).get(0).getData();
        int predictIndex = getMaxIndex(outputData);
        System.out.println((String.format("predictIndex: %d",predictIndex)));
        String predictLabel = labelTable.get(predictIndex);
        System.out.println((String.format("mapsize: %d",labelTable.size())));
        System.out.println((String.format("predictLabel: %s",predictLabel)));
        IdLabel idLabel = new IdLabel();
        idLabel.setId(value.f0);
        idLabel.setLabel(predictLabel);
        out.collect(idLabel);
    }

    @Override
    public void open(Configuration parameters) throws Exception {
        long t1 = System.currentTimeMillis();
        ConfigProto configProto = ConfigProto.newBuilder()
                .setAllowSoftPlacement(true).build();
        model = new ImageModel();
        if(modelByte == null){
            model.loadTF(modelPath,inputShape,ifReverseInputChannels,meanValues,scale,input);
        }else {
            model.loadTF(modelByte,inputShape,ifReverseInputChannels,meanValues,scale,input);
        }

        long t2 = System.currentTimeMillis();
        System.out.println((String.format("load model time %d ms",t2-t1)));
    }
    @Override
    public void close() throws Exception{
        model.release();
    }

    private int getMaxIndex(float[] arr){
        if(arr==null||arr.length==0){
            return 0;
        }
        int maxIndex=0;
        for(int i=0;i<arr.length-1;i++){
            if(arr[maxIndex]<arr[i+1]){
                maxIndex=i+1;
            }
        }
        return maxIndex;
    }

    private Mat byteToMat(byte[] input) {
        OpenCVMat mat = OpenCVMethod.fromImageBytes(input, Imgcodecs.CV_LOAD_IMAGE_UNCHANGED);
        Mat resize_mat = new Mat(cropWidth,cropHeight,mat.type());
        Imgproc.resize(mat,resize_mat,resize_mat.size(),0,0,Imgproc.INTER_CUBIC);
        return resize_mat;
    }

    private float[] fromHWC2CHW(float[] input, int length, int[] size){
        float[] outData= new float[length];
        for(int h=0;h<224;h++){
            for(int w=0;w<224;w++){
                for(int c=0;c<3;c++){
                    outData[c*224*224+h*224+w]=input[h*224*3+w*3+c];
                }
            }
        }
        return outData;
    }
}