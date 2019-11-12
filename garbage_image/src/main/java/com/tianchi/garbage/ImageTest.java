package com.tianchi.garbage;

import com.alibaba.tianchi.garbage_image_util.IdLabel;
import com.alibaba.tianchi.garbage_image_util.ImageData;
import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.util.Collector;

public class ImageTest implements FlatMapFunction<ImageData, IdLabel> {
    private ClassIndex classIndex;
    public ImageTest() {
        classIndex =new ClassIndex();
    }

    public void flatMap(ImageData value, Collector<IdLabel> out) throws Exception {
        IdLabel idLabel = new IdLabel();
        idLabel.setId(value.getId());
        int predictIndex = 50;
        String predictLabel = classIndex.getName(predictIndex);

        idLabel.setLabel(predictLabel);
        out.collect(idLabel);
    }
}