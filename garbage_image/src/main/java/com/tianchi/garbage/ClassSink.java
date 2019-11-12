package com.tianchi.garbage;

import com.alibaba.tianchi.garbage_image_util.IdLabel;
import com.alibaba.tianchi.garbage_image_util.LogTimestampPlugin;
import com.alibaba.tianchi.garbage_image_util.PrintLogPlugin;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.streaming.api.functions.sink.RichSinkFunction;
import org.apache.flink.streaming.api.functions.sink.SinkFunction.Context;
import java.lang.String;

public class ClassSink extends RichSinkFunction<IdLabel> {
    private int index;
    private transient LogTimestampPlugin plugin = null;
    private String pluginClassName = System.getenv("LOG_PLUGIN_CLASSNAME");
    private String logPath;
    private int num;

    public ClassSink() {
        if (null == this.pluginClassName) {
            this.pluginClassName = PrintLogPlugin.class.getName();
        }

        this.logPath = System.getenv("SCORE_LOG_PATH");
        if (null == this.logPath) {
            this.logPath = "";
        }
        num=0;
        System.out.println(String.format("Plugin %s %s", this.pluginClassName, this.logPath));
    }

    public void open(Configuration parameters) throws Exception {
        System.out.println(String.format("Plugin %s %s", this.pluginClassName, this.logPath));
        this.index = this.getRuntimeContext().getIndexOfThisSubtask();
        if (this.pluginClassName.isEmpty()) {
            this.plugin = null;
        } else {
            this.plugin = (LogTimestampPlugin)Class.forName(this.pluginClassName).newInstance();
        }

        if (null != this.plugin) {
            this.plugin.open(this.logPath, "sink", this.index);
        }

    }

    public void close() throws Exception {
        if (null != this.plugin) {
            this.plugin.close();
        }

    }

    public void invoke(IdLabel value, Context context) throws Exception {
        if (null != this.plugin) {
            this.plugin.logTimestamp(value.getId(), System.currentTimeMillis(), value.getLabel());
            String realLabel = value.getId().split("-")[0];
            if(realLabel.equals(value.getLabel())){
                num++;
            }
            System.out.println((String.format("current right num: %d",num)));
        }

    }
}
