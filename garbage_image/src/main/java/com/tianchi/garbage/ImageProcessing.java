package com.tianchi.garbage;

import com.alibaba.tianchi.garbage_image_util.ImageData;
import com.intel.analytics.bigdl.transform.vision.image.opencv.OpenCVMat;
import com.intel.analytics.zoo.pipeline.inference.JTensor;
import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.util.Collector;

import com.intel.analytics.zoo.feature.image.OpenCVMethod;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public class ImageProcessing implements FlatMapFunction<ImageData, Tuple2<String, JTensor>> {
    private int cropWidth;
    private int cropHeight;
    private int[] outShape;

    ImageProcessing(int[] inputShape) {
        cropWidth = inputShape[1];
        cropHeight = inputShape[2];
        outShape = inputShape;
    }

    @Override
    public void flatMap(ImageData value, Collector<Tuple2<String, JTensor>> out) throws Exception {
        Tuple2<String, JTensor> outputs = new Tuple2<String, JTensor>();

        Mat mat = this.byteToMat(value.getImage());

        int size = (int)(mat.total()*mat.channels());
        float[] floatData= new float[size];
        OpenCVMat.toFloatPixels(mat,floatData);

        int[] shape = {mat.height(),mat.width(),mat.channels()};
        floatData = this.fromHWC2CHW(floatData,size,shape);
        JTensor image = new JTensor(floatData,outShape);

        outputs.f0=value.getId();
        outputs.f1=image;
        out.collect(outputs);
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