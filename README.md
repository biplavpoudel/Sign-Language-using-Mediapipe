TLDR; This project is uses Google's Mediapipe Hand Landmarks to detect key landmarks and to create a skeleton by joining the keypoints on a white background image.
The images is then trained on CNN, where VGG16 is used as the base model, pertaining to the fact that the dataset is created by the user and is too small for the CNN model.
![The-Architecture-of-VGG-16](https://github.com/biplavpoudel/Sign-Language-using-Mediapipe/assets/60846036/4e6a8d74-ecbb-42ca-9460-0c6cfbdb66c1)

The saved model is then used to predict hand gestures by comparing with the hand landmark "skeleton".
The prediction is then used by another prediction function that explicilty compares the landmark coordinates to make sure the predictions are correct. 
![image](https://github.com/biplavpoudel/Sign-Language-using-Mediapipe/assets/60846036/145d3df3-0fb3-4c31-abb1-93fe3ff151dc)

PyQt6 is chosen as the GUI for displaying the result.
The user can further press a "Speak" button to pronounce the character, which is based on Pyttsx3 library.
![image](https://github.com/biplavpoudel/Sign-Language-using-Mediapipe/assets/60846036/c95a0953-54a9-4fa9-ab62-e3b5e744bb33)

Still few bugs remain to smoothen out!
