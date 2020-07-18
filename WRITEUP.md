# Project Write-Up


For this project, you’ll first find a useful person detection model and convert it to an Intermediate Representation for use with the Model Optimizer. Utilizing the Inference Engine, you'll use the model to perform inference on an input video, and extract useful data concerning the count of people in frame and how long they stay in frame. You'll send this information over MQTT, as well as sending the output frame, in order to view it from a separate UI server over a network.


```
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
tar -xvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz
cd ssd_mobilenet_v2_coco_2018_03_29
python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
python main.py --input resources/Pedestrian_Detect_2_1_1.mp4 --model ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.xml --cpu_extension /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so --device CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
```

## Explaining Custom Layers

The process behind converting custom layers involves:
* Register those layers as extensions to the Model Optimizer. In this case, the Model Optimizer generates a valid and optimized Intermediate Representation.
* Some other options available depending on the original model framework:
  * For Caffe, 2nd option is to register the layers as Custom, then use Caffe to calculate the output shape of the layer.
  * For TensorFlow, 
    * 2nd option is to actually replace the unsupported subgraph with a different subgraph.
    * 3rd option is to actually offload the computation of the subgraph back to TensorFlow during inference. 


Some of the potential reasons for handling custom layers are:
* For these reasons, it is best to use the Model Optimizer extensions for Custom Layers: you do not depend on the framework and fully control the workflow.



## Comparing Model Performance

I have been impregnated by the following links during my choice selection: https://cocodataset.org/#detection-eval ,
https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow.html,
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

I initialy hesitated between two pre-trained models  ssd_mobilenet_v2_coco_2018_03_29 and faster_rcnn_inception_v2_coco but did my choice on the mean average precision and have choosen the 
faster_rcnn_inception_v2_coco model. Even though the speed is 58 and less than Ssd_inception_v2_coco but the COCO mAP is higher 28 instead of 24. 
I prefered the precision than the speed as I am looking at the quality efficiency not the time efficiency. But after I started to run my program it was failing for some reasons I can't explain. 
Therefore I swapped on my choice with ssd_mobilenet_v2_coco_2018_03_29 which worked perfectely.
Intel openVINO already contains extensions for custom layers used in TensorFlow Object Detection Model Zoo.

Model-1: ssd_mobilenet_v2_coco_2018_03_29

Converted the model to intermediate representation using the following command. Further, this model lacked accuracy as it didn't detect people correctly in the video. Made some alterations to the threshold for increasing its accuracy but the results were not fruitful.
```
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
tar -xvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz
python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
```

[ SUCCESS ] Generated IR model.
[ SUCCESS ] XML file: /home/workspace/ssd_mobilenet_v2_coco_2018_03_29/./frozen_inference_graph.xml
[ SUCCESS ] BIN file: /home/workspace/ssd_mobilenet_v2_coco_2018_03_29/./frozen_inference_graph.bin
[ SUCCESS ] Total execution time: 60.17 seconds. 

	- Precision of IR: FP32

root@8211074dd0a3:/home/workspace/ssd_mobilenet_v2_coco_2018_03_29# ls -ltr
total 203520
-rw-r--r-- 1 345018 89939  3496023 Mar 30  2018 model.ckpt.meta
-rw-r--r-- 1 345018 89939    15069 Mar 30  2018 model.ckpt.index
-rw-r--r-- 1 345018 89939 67505156 Mar 30  2018 model.ckpt.data-00000-of-00001
-rw-r--r-- 1 345018 89939       77 Mar 30  2018 checkpoint
drwxr-xr-x 3 345018 89939     4096 Mar 30  2018 saved_model
-rw-r--r-- 1 345018 89939     4204 Mar 30  2018 pipeline.config
-rw-r--r-- 1 345018 89939 69688296 Mar 30  2018 frozen_inference_graph.pb
-rw-r--r-- 1 root   root  67272876 Jul 16 15:51 frozen_inference_graph.bin
-rw-r--r-- 1 root   root    111739 Jul 16 15:51 frozen_inference_graph.xml
-rw-r--r-- 1 root   root     49370 Jul 16 15:51 frozen_inference_graph.mapping
root@8211074dd0a3:/home/workspace/ssd_mobilenet_v2_coco_2018_03_29# 

Size = 68mb
Inference time avg = 69ms
Accuracy FP32  with **0.6** threshold| False Positive = low <br /> False Negative = consequent for one particalur person <br /> 


Model-2: Faster_rcnn_inception_v2_coco_2018_01_28

Converted the model to intermediate representation using the following command. Model -2 i.e. Faster_rcnn_inception_v2_coco, performed really well in the output video. After using a threshold of 0.4, the model works better than all the previous approaches.
```
python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config faster_rcnn_inception_v2_coco_2018_01_28/pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json

```

[ SUCCESS ] Generated IR model.
[ SUCCESS ] XML file: /home/workspace/./frozen_inference_graph.xml
[ SUCCESS ] BIN file: /home/workspace/./frozen_inference_graph.bin
[ SUCCESS ] Total execution time: 137.99 seconds. 

	- Precision of IR: 	FP32
    
root@8211074dd0a3:/home/workspace/faster_rcnn_inception_v2_coco_2018_01_28# ls -ltr
total 165844
-rw-r--r-- 1 345018 5000    15927 Feb  1  2018 model.ckpt.index
-rw-r--r-- 1 345018 5000 53348500 Feb  1  2018 model.ckpt.data-00000-of-00001
-rw-r--r-- 1 345018 5000       77 Feb  1  2018 checkpoint
-rw-r--r-- 1 345018 5000  5685731 Feb  1  2018 model.ckpt.meta
drwxr-xr-x 3 345018 5000     4096 Feb  1  2018 saved_model
-rw-r--r-- 1 345018 5000     3244 Feb  1  2018 pipeline.config
-rw-r--r-- 1 345018 5000 57153785 Feb  1  2018 frozen_inference_graph.pb
-rw-r--r-- 1 root   root 53229380 Jul 16 16:08 frozen_inference_graph.bin
-rw-r--r-- 1 root   root   126399 Jul 16 16:08 frozen_inference_graph.xml
-rw-r--r-- 1 root   root    45416 Jul 16 16:08 frozen_inference_graph.mapping


I tried to use an OpenVino toolkit to extract some interesting metrics and give a deep comparison but unfortunately it didn't work as that was the only thing I had in my possession at this time. If you can give some tips and tricks on how we can do it otherwise it will be very appreciate.

```
cd /opt/intel/openvino/deployment_tools/tools/benchmark_tool/
python3 benchmark_app.py -m /home/workspace/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.xml  -d CPU -api async -i /home/workspace/resources/Pedestrian_Detect_2_1_1.mp4  -progress true -b 1
```





## Assess Model Use Cases

Some of the potential use cases of the people counter app are:

* Monitoring people entering/leaving/duration at specifc location (polling entity) to check peak hours and may give some alerting once specifc count/duration exceeded
* Monitoring people durations at customer check outs and enhance the customer service process (SLA/queues etc..)
* Monitoring people on train station and how much time they spent in front each train track and try to enhance the congestion to take actions against covid19 spread for example

Each of these use cases would be useful because:
* they will automatically and visually allow the monitoring and handling the required space all around the clock time with the possibility and ability to send alerting on the spot so that some actions can be taken.


## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows:
* Accuracy: to avoid false positives and  false negatives
* Lightening: to avoid incorrect bounding boxes all around the people, maybe smaller and could false the counter.
* Focal length/image size: Each model is sensitive to the size of the image, and more than that each hardware is reacting differently on the image size which could have some technical impacts: Inference time slower, processing slower...


## Model Research

[This heading is only required if a suitable model was not found after trying out at least three
different models. However, you may also use this heading to detail how you converted 
a successful model.]

In investigating potential people counter models, I tried each of the following three models:

- Model 1: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...
  
- Model 2: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...

- Model 3: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...
