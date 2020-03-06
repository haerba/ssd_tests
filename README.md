# ssd_tests

Examples demonstrating how to optimize caffe/tensorflow/darknet models with TensorRT and run inferencing on NVIDIA Jetson or x86_64 PC platforms.  Highlights:  (The following FPS numbers have been updated using test results against JetPack 4.3, i.e. TensorRT 6, on Jetson Nano.)

* Run an optimized 'GoogLeNet' image classifier at ~60 FPS on Jetson Nano.
* Run a very accurate optimized 'MTCNN' face detector at 6~11 FPS on Jetson Nano.
* Run an optimized 'ssd_mobilenet_v1_coco' object detector ('trt_ssd_async.py') at 27~28 FPS on Jetson Nano.
* Run an optimized 'yolov3-416' object detector at ~3 FPS on Jetson Nano.
* All demos work on Jetson TX2 and AGX Xavier ([link](https://github.com/jkjung-avt/tensorrt_demos/issues/19#issue-517897927) and [link](https://github.com/jkjung-avt/tensorrt_demos/issues/30)), and run much faster!
* Furthermore, all demos should work on x86_64 PC with NVIDIA GPU(s) as well.  Some minor tweaks would be needed.  Please refer to [README_x86.md](https://github.com/jkjung-avt/tensorrt_demos/blob/master/README_x86.md) for more information.

Table of contents
-----------------

* [Prerequisite](#prerequisite)
* [Demo #3: SSD](#ssd)


<a name="prerequisite"></a>
Prerequisite
------------

The code in this repository was tested on both Jetson Nano and Jetson TX2 Devkits.  In order to run the demos below, first make sure you have the proper version of image (JetPack) installed on the target Jetson system.  For example, reference for Jetson Nano: [Setting up Jetson Nano: The Basics](https://jkjung-avt.github.io/setting-up-nano/).

More specifically, the target Jetson system must have TensorRT libraries installed.  **Demo #1 and Demo #2 should work for TensorRT 3.x, 4.x, 5.x, 6.x.  But Demo #3 and Demo #4 would require TensorRT 5.x or 6.x.**

You could check which version of TensorRT has been installed on your Jetson system by looking at file names of the libraries.  For example, TensorRT v5.1.6 (from JetPack-4.2.2) was present on my Jetson Nano DevKit.

```shell
$ ls /usr/lib/aarch64-linux-gnu/libnvinfer.so*
/usr/lib/aarch64-linux-gnu/libnvinfer.so
/usr/lib/aarch64-linux-gnu/libnvinfer.so.5
/usr/lib/aarch64-linux-gnu/libnvinfer.so.5.1.6
```

Furthermore, the demo programs require 'cv2' (OpenCV) module in python3.  You could, for example, refer to [Installing OpenCV 3.4.6 on Jetson Nano](https://jkjung-avt.github.io/opencv-on-nano/) for how to install opencv-3.4.6 on your Jetson system.

Lastly, if you plan to run Demo #3 (SSD), you'd also need to have 'tensorflowi-1.x' installed.  You could refer to [Building TensorFlow 1.12.2 on Jetson Nano](https://jkjung-avt.github.io/build-tensorflow-1.12.2/) for how to install tensorflow-1.12.2 on the Jetson system.


Demo #3: SSD
------------

This demo shows how to convert trained tensorflow Single-Shot Multibox Detector (SSD) models through UFF to TensorRT engines, and to do real-time object detection with the optimized TensorRT engines.

NOTE: This particular demo requires TensorRT 'Python API', which is only available in TensorRT 5.x+ on the Jetson systems.  In other words, this demo only works on Jetson systems properly set up with JetPack-4.2+, but **not** JetPack-3.x or earlier versions.

Assuming this repository has been cloned at '${HOME}/project/tensorrt_demos', follow these steps:

1. Install requirements (pycuda, etc.) and build TensorRT engines from the trained SSD models.

   ```shell
   $ cd ${HOME}/project/tensorrt_demos/ssd
   $ ./install.sh
   $ ./build_engines.sh
   ```

   NOTE: On my Jetson Nano DevKit with TensorRT 5.1.6, the version number of UFF converter was "0.6.3".  When I ran 'build_engine.py', the UFF library actually printed out: `UFF has been tested with tensorflow 1.12.0. Other versions are not guaranteed to work.`  So I would strongly suggest you to use **tensorflow 1.12.x** (or whatever matching version for the UFF library installed on your system) when converting pb to uff.

2. Run the 'trt_ssd.py' demo program.  The demo supports 4 models: 'ssd_mobilenet_v1_coco', 'ssd_mobilenet_v1_egohands', 'ssd_mobilenet_v2_coco', or 'ssd_mobilenet_v2_egohands'.  For example, I tested the 'ssd_mobilenet_v1_coco' model with the 'huskies' picture.

   ```shell
   $ cd ${HOME}/project/tensorrt_demos
   $ python3 trt_ssd.py --model ssd_mobilenet_v1_coco \
                        --image \
                        --filename ${HOME}/project/tf_trt_models/examples/detection/data/huskies.jpg
   ```

   Here's the result (JetPack-4.2.2, i.e. TensorRT 5).  Frame rate was good (over 20 FPS).

   ![Huskies detected](https://raw.githubusercontent.com/jkjung-avt/tensorrt_demos/master/doc/huskies.png)

   NOTE: When running this demo with TensorRT 6 (JetPack-4.3) on the Jetson Nano, I encountered the following error message which could probably be ignored for now.  Quote from [NVIDIA's NVES_R](https://devtalk.nvidia.com/default/topic/1065233/tensorrt/-tensorrt-error-could-not-register-plugin-creator-flattenconcat_trt-in-namespace-/post/5394191/#5394191): `This is a known issue and will be fixed in a future version.`

   ```
   [TensorRT] ERROR: Could not register plugin creator: FlattenConcat_TRT in namespace
   ```

   I also tested the 'ssd_mobilenet_v1_egohands' (hand detector) model with a video clip from YouTube, and got the following result.  Again, frame rate was pretty good.  But the detection didn't seem very accurate though :-(

   ```shell
   $ python3 trt_ssd.py --model ssd_mobilenet_v1_egohands \
                        --file \
                        --filename ${HOME}/Videos/Nonverbal_Communication.mp4
   ```

   (Click on the image below to see the whole video clip...)

   [![Hands detected](https://raw.githubusercontent.com/jkjung-avt/tensorrt_demos/master/doc/hands.png)](https://youtu.be/3ieN5BBdDF0)

3. The 'trt_ssd.py' demo program could also take various image inputs.  Refer to step 5 in Demo #1 again.

4. Referring to this comment, ['#TODO enable video pipeline'](https://github.com/AastaNV/TRT_object_detection/blob/master/main.py#L78), in the original TRT_object_detection code, I did implement an 'async' version of ssd detection code to do just that.  When I tested 'ssd_mobilenet_v1_coco' on the same huskies image with the async demo program, frame rate improved 3~4 FPS.

   ```shell
   $ cd ${HOME}/project/tensorrt_demos
   $ python3 trt_ssd_async.py --model ssd_mobilenet_v1_coco \
                              --image \
                              --filename ${HOME}/project/tf_trt_models/examples/detection/data/huskies.jpg
   ```

5. To verify accuracy (mAP) of the optimized TensorRT engines and make sure they do not degrade too much (due to reduced floating-point precision of 'FP16') from the original TensorFlow frozen inference graphs, you could prepare validation data and run 'eval_ssd.py'.  Refer to [README_eval_ssd.md](README_eval_ssd.md) for details.

   I compared mAP of the TensorRT engine and the original tensorflow model for both 'ssd_mobilenet_v1_coco' and 'ssd_mobilenet_v2_coco' using COCO 'val2017' data.  The results were good.  In both cases, mAP of the optimized TensorRT engine matched the original tensorflow model.  The FPS (frames per second) numbers in the table were measured using 'trt_ssd_async.py' on my Jetson Nano DevKit with JetPack-4.3.

   | TensorRT engine         | mAP @<br>IoU=0.5:0.95 |  mAP @<br>IoU=0.5  | FPS on Nano |
   |:------------------------|:---------------------:|:------------------:|:-----------:|
   | mobilenet_v1 TF         |          0.232        |        0.351       |      --     |
   | mobilenet_v1 TRT (FP16) |          0.232        |        0.351       |     27.7    |
   | mobilenet_v2 TF         |          0.248        |        0.375       |      --     |
   | mobilenet_v2 TRT (FP16) |          0.248        |        0.375       |     22.7    |

6. Check out my blog posts for implementation details:

   * [TensorRT UFF SSD](https://jkjung-avt.github.io/tensorrt-ssd/)
   * [Speeding Up TensorRT UFF SSD](https://jkjung-avt.github.io/speed-up-trt-ssd/)
   * [Verifying mAP of TensorRT Optimized SSD and YOLOv3 Models](https://jkjung-avt.github.io/trt-detection-map/)
   * Or if you'd like to learn how to train your own custom object detectors which could be easily converted to TensorRT engines and inferenced with 'trt_ssd.py' and 'trt_ssd_async.py': [Training a Hand Detector with TensorFlow Object Detection API](https://jkjung-avt.github.io/hand-detection-tutorial/)

