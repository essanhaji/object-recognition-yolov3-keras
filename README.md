# Object Recognition With YOLOv3 in Keras

## Summary

Object recognition with YOLOv3 for Keras users based on [keras yolo3 GitHub repository](https://github.com/experiencor/keras-yolo3).

<br>

## Requirements

- Get the package manager **PyPi** ready.

- All requirements that you will need with its version it's exist in `requirements.txt` so you need just to run this command to install it all :


   ```
   !pip install -r requirements.txt
   ```

- Get the YOLOv3 pretrained weights from [here](https://pjreddie.com/media/files/yolov3.weights) and add it to the repository.


<br>

## How to run it

- When you install all requirements pick in the repository and open the terminal then :

   ```
   python build_model.py
   ```
   to encode the `YOLOv3` model format to `Keras` model format.

- Run the `main.py` to get the predected result.

- Try to change the input image to experiment with the model in other objects.

**Note :** tha your input image should be one of the objects that the model trained on, you can find them in the [.classes](/.classes) file.



<br>

## Authors

* [El Houcine ES SANHAJI](https://www.linkedin.com/in/essanhaji/)

<br>

## Thank you.