# CNN_Katagana_Recognition
Personnal project of deep learning, with a Convolutional Neural Network (CNN) in order to recognize katakana character.


### Dataset and credits 
Dataset found on ETL website ( http://etlcdb.db.aist.go.jp ), Please visit this in order to get the dataset (not in files given here).

Read of the dataset and initial usage of this usage based on the work by Aiyu Kamate ( https://towardsdatascience.com/creating-a-japanese-handwriting-recognizer-70be12732889 ) in order to use the dataset. Thank for his work.

Creation of the CNN and after analysis and personnal test, it's my own work.

## Goal of the project

The goal of this project is to recognize a Katakana ideogram. It's a personnal project that we will be used for the VR game project.

Here you can see the list of Katakana : \
![image](https://user-images.githubusercontent.com/80623426/158269678-abcc50db-b321-4db9-b75c-8da072fa92ff.png) 

### Dataset explaination
In the dataset, there is 51 classes, where 3 are repetitions of others classes (36 = 1, 38 = 3, 47 = 2) \
Then, we have 48 classes, for 46 ideograms used actually (see the list before) and 2 old ideograms that are no longer in use (WI - ヰ and WE - ヱ). \

Files are crypted, so a key is needed in order to access to data (please check in read_kana function) \
![image](https://user-images.githubusercontent.com/80623426/158273471-2325d309-b319-4974-849a-53bf5d54c448.png)


### Summary of the CNN model
Here you can access to a summary of the CNN I used in order to complete my project, you can access to the creation part in the file if you want to know hyper parameters used. \
![image](https://user-images.githubusercontent.com/80623426/158270553-e524cca5-af10-48ce-84ea-2330ef193f00.png)

### Test accuracy and loss
With a train/test split (ratio of 20% of the dataset used for test) we can obtain a accuracy and the loss for the test phase, after the completion of the train phase. \
![image](https://user-images.githubusercontent.com/80623426/158271388-54fc7bba-acf0-4043-8914-56875a08a1a9.png)

### Usage of the draw test
In order to check if the CNN is working well, and especially in a context where we will use the CNN like we want to use it in the VR game project, \
I created a sample of home made picture (48x48 pixels like in the dataset), and I tested those with the predict feature, with the objective of check \
if the CNN can be used to check if a drawing made with the VR game is the asked one. \
HO - ホ : \
![image](https://user-images.githubusercontent.com/80623426/158272900-c485c036-fc1e-4f49-851a-9a3b71b4a376.png)

![image](https://user-images.githubusercontent.com/80623426/158273070-22121764-05dc-43b3-b5fc-fa88ea7dd62c.png)

![image](https://user-images.githubusercontent.com/80623426/158273519-a3e5d8d5-9dfb-4947-aedc-5a534d58c59b.png)


