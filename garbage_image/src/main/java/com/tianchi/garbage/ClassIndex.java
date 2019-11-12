package com.tianchi.garbage;

import org.apache.commons.io.IOUtils;
import org.apache.hadoop.yarn.webapp.hamlet.Hamlet;
import sun.nio.ch.IOUtil;

import java.io.*;
import java.net.URL;
import java.util.Map;
import java.util.HashMap;

public class ClassIndex {
    static Map<Integer, String> mapTable = new HashMap<Integer, String>();

    static void InitTable() throws IOException {
        ClassLoader classLoader = ClassIndex.class.getClassLoader();
        InputStream classStream = classLoader.getResourceAsStream("class_index.txt");
        //URL resource = classLoader.getResource("class_index.txt");
        //File f = new File(resource.getFile());
        String value = IOUtils.toString(classStream,"UTF-8");
        String[] line = value.split("\n");
        for (int i=0;i<line.length;i++){
            //System.out.println((String.format("Line: %s",line[i])));
            String[] res = line[i].split(" ");
            String name=res[0];
            int id = 0;
            try{
                id = Integer.parseInt(res[1]);
            }
            catch (Exception e){
            }
            mapTable.put(id,name);
        }
        System.out.println((String.format("mapsize: %d",mapTable.size())));
    }

    static Map<Integer, String> getTable()
    {
        return mapTable;
    }

    static String getName(int id)
    {
        try{
            return mapTable.get(id);
        }catch (Exception e)
        {
            return "";
        }
    }

    static int getMaxIndex(float[] arr){
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

}
