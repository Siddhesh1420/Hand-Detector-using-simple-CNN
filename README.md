# Hand-Detector-using-simple-CNN
## Description
In this project, I have tried to build a CNN model which trains on the dataset containing hands, and tries to predict in realtime if a hand is present in the frame.
## How to use
1. Collect data using Data_Collector.py
1.1. Press 'S' to start saving and press 'S' to stop saving
1.2. Save the images so that first n images are the ones which contains hand inside the green box, and the next images don't have any hands in the box.
1.3. Press 'Q' to quit
2. Train the model using train_hand_detector.py
2.1. Replace the value of Z in the code to n, where n is the number of the last saved image with a hand in it. Note that images from 0 to n should have a hand in the box, and next images must not.
2.2. Run the file and wait for the model to train.
3. Run the realtime_detect.py to detect the presence of a hand
3.1. Keep the hand inside the green box, and see the chances of being a hand in the box rise.
3.2. Keep the hand out of the box, and see the chances of being a hand inside the box fall.

## Enjoy!!!
