# Sign-Language-Character-Recognition-and-real-time-implementation


The Sign Language Recognition with deep learning project aims to develop a system that can  recognize and translate sign language into text. This project employs a deep learning approach to  process and analyze the complex and dynamic patterns of sign language gestures.

The system uses a convolutional neural network (CNN), and implemented using MediaPipe and OpenCV.

The dataset used for this project is the “Sign Language MNIST”
.
The performance of the model is evaluated based on accuracy and precision, and the proposed system  aim to accurately recognize and translate sign language gestures into text.

The proposed system has the potential to improve the communication and accessibility for the deaf  and hard-of-hearing community
![image](https://user-images.githubusercontent.com/57862480/229497752-c7c1cd70-6796-4294-a5ec-deee1ec8a79f.png)




The libraries used are matplotlib, seaborn, pandas, numpy and keras
 
 
the file " SIGN LANGUAGE CHARACTER RECOGNITION CODE" includes the following: 
-importing libraries
-Loading the ASL dataset
-Data Visualization and Preprocessing
-Data Augmentation
-Training The Model
-Analysis after Model Training
-Predcitions on random images 
-Heatmap: confusion matrix 

The model is trained with 20 epochs with softmax activation and adam optimizer. This CNN model is designed to extract relevant features from images, reduce their dimensions through pooling, and classify them using fully connected layers with appropriate activation functions. The use of Dropout and BatchNormalization layers can improve its generalization ability and prevent overfitting. the model accuracy is 99.4 % . From the confusion matrix, we can see that the model is generally performing well, with high numbers along the diagonal, indicating that the model is correctly identifying many of the sign language characters.





Prediction on random images: 
![image](https://user-images.githubusercontent.com/57862480/229501802-ddcf028d-6559-407e-a788-d144441af6dd.png)



The file" Imeplementation in real time using MediapPipe and OpenCV": 
The system follows the workflow as shown in the image. 
![image](https://user-images.githubusercontent.com/57862480/229503013-13ad7e2e-3f42-420d-8841-675cdb039b91.png)




Our user interface includes our OpenCV webcam that displaces the camera window along with the predicted outputs. The code output displays the confidence % of the predictions. 
OpenCV captures the frame and MediaPipe detects the hand landmarks creating a bounding box around it. 
•	The model is then applied and the predictions are displayed on the window along with the confidence %. 
•	The output accuracy depends on the lights and quality of the camera, and in real time the system detects characts that are very distinguishable accurately while characters sign that are very similar such a M and N and K and V are detected incorrectly in some cases with a confidence of 70%
![image](https://user-images.githubusercontent.com/57862480/229502586-725d3cbe-4699-45b6-a2bc-8566b378c0c2.png)




