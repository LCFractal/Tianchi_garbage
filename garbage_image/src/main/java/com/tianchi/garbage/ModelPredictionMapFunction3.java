package com.tianchi.garbage;

import com.alibaba.tianchi.garbage_image_util.IdLabel;
import com.alibaba.tianchi.garbage_image_util.ImageData;
import com.intel.analytics.bigdl.transform.vision.image.opencv.OpenCVMat;
import com.intel.analytics.zoo.feature.image.OpenCVMethod;
import com.intel.analytics.zoo.pipeline.inference.JTensor;
import org.apache.flink.api.common.functions.RichFlatMapFunction;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.util.Collector;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.framework.ConfigProto;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

public class ModelPredictionMapFunction3 extends RichFlatMapFunction<ImageData, IdLabel> {
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

    ModelPredictionMapFunction3(byte[] modelByte, int[] inputShape, boolean ifReverseInputChannels,
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

    ModelPredictionMapFunction3(String modelPath, int[] inputShape, boolean ifReverseInputChannels,
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
    public void flatMap(ImageData value, Collector<IdLabel> out) throws Exception {
        // Image Processing
        Tuple2<String, JTensor> outputs = new Tuple2<String, JTensor>();

        Mat mat = this.byteToMat(value.getImage());

        int size = (int)(mat.total()*mat.channels());
        float[] floatData= new float[size];
        OpenCVMat.toFloatPixels(mat,floatData);

        int[] shape = {mat.height(),mat.width(),mat.channels()};
        floatData = this.fromHWC2CHW(floatData,size,shape);
        JTensor image = new JTensor(floatData,outShape);

        // Image Predict
        List<JTensor> data = Arrays.asList(image);
        List<List<JTensor>> inputs = new ArrayList<>();
        inputs.add(data);
        float[] outputData = model.doPredict(inputs).get(0).get(0).getData();
        int predictIndex = getPreIndex(outputData);
        System.out.println((String.format("predictIndex: %d",predictIndex)));
        String predictLabel = labelTable.get(predictIndex);
        System.out.println((String.format("mapsize: %d",labelTable.size())));
        System.out.println((String.format("predictLabel: %s",predictLabel)));
        IdLabel idLabel = new IdLabel();
        idLabel.setId(value.getId());
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

    private int getPreIndex(float[] arr){
        if(arr==null||arr.length==0){
            return 0;
        }

        int[] Index=Arraysort(arr,true);
        int outIndex=Index[0];
        //if(arr[0]/2<arr[1])
        //    outIndex=Index[1];

        return outIndex;
    }

    private Mat byteToMat(byte[] input) {
        OpenCVMat mat = OpenCVMethod.fromImageBytes(input, Imgcodecs.CV_LOAD_IMAGE_UNCHANGED);
        // init
        float zoom=0.2f;

        //=======Image Processing=======
        // zoom
        int zoom_width=(int)(cropWidth*(1+zoom));
        int zoom_height=(int)(cropHeight*(1+zoom));
        Mat resize_mat = new Mat(zoom_width,zoom_height,mat.type());
        Imgproc.resize(mat,resize_mat,resize_mat.size(),0,0,Imgproc.INTER_LINEAR);

        // crop
        int start_x=(zoom_width-cropWidth)/2;
        int start_y=(zoom_height-cropHeight)/2;
        Rect rect = new Rect(start_x,start_y,cropWidth,cropHeight);
        Mat outMat = new Mat(resize_mat,rect);
        return outMat;
    }

    private float[] fromHWC2CHW(float[] input, int length, int[] size){
        float[] outData= new float[length];
        for(int h=0;h<cropHeight;h++){
            for(int w=0;w<cropWidth;w++){
                for(int c=0;c<3;c++){
                    outData[c*cropHeight*cropWidth+h*cropWidth+w]=input[h*cropWidth*3+w*3+c];
                }
            }
        }
        return outData;
    }


    private static int[] Arraysort(float[] arr, boolean desc) {
        float temp;
        int index;
        int k = arr.length;
        int[] Index = new int[k];
        for (int i = 0; i < k; i++) {
            Index[i] = i;
        }

        for (int i = 0; i < arr.length; i++) {
            for (int j = 0; j < arr.length - i - 1; j++) {
                if (desc) {
                    if (arr[j] < arr[j + 1]) {
                        temp = arr[j];
                        arr[j] = arr[j + 1];
                        arr[j + 1] = temp;

                        index = Index[j];
                        Index[j] = Index[j + 1];
                        Index[j + 1] = index;
                    }
                } else {
                    if (arr[j] > arr[j + 1]) {
                        temp = arr[j];
                        arr[j] = arr[j + 1];
                        arr[j + 1] = temp;

                        index = Index[j];
                        Index[j] = Index[j + 1];
                        Index[j + 1] = index;
                    }
                }
            }
        }
        return Index;
    }
}