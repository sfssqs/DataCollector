# DataCollector
数据采集器



# FaceSwap
This is the code behind the Switching Eds blog post.

See the link for an explanation of the code.

To run the script you'll need to install dlib (http://dlib.net) including its Python bindings, and OpenCV. You'll also need to obtain the trained model from sourceforge.

Unzip with bunzip2 and change PREDICTOR_PATH to refer to this file. The script is run like so:

./faceswap.py <head image> <face image>
If successful, a file output.jpg will be produced with the facial features from <head image> replaced with the facial features from <face image>.
