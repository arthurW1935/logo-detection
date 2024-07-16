# Logo Detection 

### Overview
This project aims to detect logo in videos and register the timestamps using YOLOv8, a state-of-the-art ML Model for Object Detection. 
In this project, we tried to detect logos of Pepsico and Coca-Cola, but this can be extended to any number of logos with proper dataset and training.
If you want to learn more about how I approached this project, you can read it [here](https://docs.google.com/document/d/1vhl_bZjmxmsfXPtpjJRrli6MWCVRBBD0KGYQStSDqic/edit?usp=sharing).

### Technologies Used
- Language: Python
- Model Architechture: YOLOv8n
- Modules and Frameworks: PyTorch, Ultralytics, PyAV, OpenCV


## Setup Instructions
Clone the repository
```
git clone https://github.com/yourusername/logo-detection.git
cd logo-detection/
```

Create a virtual environment (You must have Python in your system)
```
python -m venv venv
venv\Scripts\activate
```

Now install all the dependencies using requirements.txt
```
pip install -r requirements.txt
```

## Running the application
To run the program, you can run the following in your terminal

```
python process_video.py path\to\your\video.mp4
```

If you want to change the batch size, number of frames to be skipped, or the path to json output, you can add those args in the command
```
python process_video.py path\to\your\video --batch_size 32 --frame_skip 4 --output_json path\to\your\output\json
```

If you want to see things in action and what is being predicted, you can run the following command
```
python visualise_process_video.py path\to\your\video
```


### Training the model
If you want to train your model with your own dataset, you can checkout train.ipynb. Or if you want to train it locally without using Jupyter, then check out the train.py. 
I did the training in Google Colab, you can have a look at it [here](https://colab.research.google.com/drive/1A4ZpxHfb8aIH6hP3qQl0Ed6Sew0-o6z2?usp=sharing).
You can use my dataset from [here](https://drive.google.com/file/d/1AQF5bBC7dEbwA5H26U410UZc4zqzl-D9/view?usp=drive_link).

Dataset Credits:
- [Dataset 1](https://universe.roboflow.com/detectionanas/pepsi-logo-detection/dataset/1)
- [Dataset 2](https://universe.roboflow.com/roboflow-xuntf/coca-cola-detection-weydo/dataset/4)
- [Dataset 3](https://universe.roboflow.com/aiforengineer/ai_for_engineer_class/dataset/3)
- [Dataset 4](https://universe.roboflow.com/contact-brockmann-gmail-com/cocacola-xhprp/dataset/6)
- [Dataset 5](https://universe.roboflow.com/test01-fr735/brands-qv6fs/dataset/1)


