import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.layers import Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import cv2
from sklearn.model_selection import train_test_split
import pickle
import os
import pandas as pd
import random
from keras.preprocessing.image import ImageDataGenerator
import seaborn as sns  # For confusion matrix

################# Parameters #####################
# Parameters
path = "Train"
labelFile = 'labels.csv'
batch_size_val = 30
steps_per_epoch_val = 2000
epochs_val = 10
imageDimesions = (32, 32, 3) # keep the color channel
testRatio = 0.2
validationRatio = 0.2

# Importing the Images
count = 0
images = []
classNo = []
myList = os.listdir(path)
print("Total Classes Detected:", len(myList))
noOfClasses = len(myList)
print("Importing Classes.....")
for x in range(0, len(myList)):
    myPicList = os.listdir(path + "/" + str(count))
    for y in myPicList:
        curImg = cv2.imread(path + "/" + str(count) + "/" + y)
        curImg = cv2.resize(curImg, imageDimesions[:2]) # resize the image to the required dimensions
        images.append(curImg)
        classNo.append(count)
    print(count, end=" ")
    count += 1
print(" ")
images = np.array(images)
classNo = np.array(classNo)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validationRatio)

# Check if the number of images matches the number of labels for each dataset
print("Data Shapes")
print("Train", end=""); print(X_train.shape, y_train.shape)
print("Validation", end=""); print(X_validation.shape, y_validation.shape)
print("Test", end=""); print(X_test.shape, y_test.shape)
assert (X_train.shape[0] == y_train.shape[0]), "The number of images is not equal to the number of labels in the training set"
assert (X_validation.shape[0] == y_validation.shape[0]), "The number of images is not equal to the number of labels in the validation set"
assert (X_test.shape[0] == y_test.shape[0]), "The number of images is not equal to the number of labels in the test set"
assert (X_train.shape[1:] == (imageDimesions)), "The dimensions of the training images are wrong"
assert (X_validation.shape[1:] == (imageDimesions)), "The dimensions of the validation images are wrong"
assert (X_test.shape[1:] == (imageDimesions)), "The dimensions of the test images are wrong"

# Read CSV file
data = pd.read_csv(labelFile)
print("data shape ", data.shape, type(data))

# Display some sample images of all the classes
num_of_samples = []
cols = 5
num_classes = noOfClasses
fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(5, 300))
fig.tight_layout()
for i in range(cols):
    for j, row in data.iterrows():
        x_selected = X_train[y_train == j]
        axs[j][i].imshow(x_selected[random.randint(0, len(x_selected) - 1), :, :], cmap=plt.get_cmap("viridis")) # use viridis colormap to display color images
        axs[j][i].axis("off")
        if i == 2:
            axs[j][i].set_title(str(j) + "-" + row["Name"])
            num_of_samples.append(len(x_selected))

# Display a bar chart showing the number of samples for each category
print(num_of_samples)
plt.figure(figsize=(12, 4))
plt.bar(range(0, num_classes), num_of_samples)
plt.title("Distribution of the training dataset")
plt.xlabel("Class number")
plt.ylabel("Number of images")
plt.show()

# Preprocessing the images
def preprocess(img):
    img = img / 255.0 # normalize the pixel values to [0, 1]
    return img

X_train = np.array([preprocess(img) for img in X_train])
X_validation = np.array([preprocess(img) for img in X_validation])
X_test = np.array([preprocess(img) for img in X_test])

############################### AUGMENTATAION OF IMAGES: TO MAKEIT MORE GENERIC
dataGen= ImageDataGenerator(width_shift_range=0.1,   # 0.1 = 10%     IF MORE THAN 1 E.G 10 THEN IT REFFERS TO NO. OF  PIXELS EG 10 PIXELS
                            height_shift_range=0.1,
                            zoom_range=0.2,  # 0.2 MEANS CAN GO FROM 0.8 TO 1.2
                            shear_range=0.1,  # MAGNITUDE OF SHEAR ANGLE
                            rotation_range=10)  # DEGREES
dataGen.fit(X_train)
batches= dataGen.flow(X_train,y_train,batch_size=20)  # REQUESTING DATA GENRATOR TO GENERATE IMAGES  BATCH SIZE = NO. OF IMAGES CREAED EACH TIME ITS CALLED
X_batch,y_batch = next(batches)

# TO SHOW AGMENTED IMAGE SAMPLES
fig,axs=plt.subplots(1,15,figsize=(20,5))
fig.tight_layout()

for i in range(15):
    axs[i].imshow(X_batch[i].reshape(imageDimesions[0],imageDimesions[1], 3)) # display color images
    axs[i].axis('off')
plt.show()

# Convert class labels to one-hot encoded vectors
y_train = to_categorical(y_train, num_classes)
y_validation = to_categorical(y_validation, num_classes)
y_test = to_categorical(y_test, num_classes)

############################### CONVOLUTION NEURAL NETWORK MODEL
# Define the model architecture
model = Sequential()
model.add(Conv2D(60, (5, 5), input_shape=(imageDimesions[0], imageDimesions[1], 3), activation='relu')) # keep the color channel
model.add(Conv2D(60, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(30, (3, 3), activation='relu'))
model.add(Conv2D(30, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model using the dataset
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.shuffle(len(X_train)).repeat().batch(batch_size_val)

history = model.fit(train_dataset, steps_per_epoch=steps_per_epoch_val, epochs=epochs_val, 
                    validation_data=(X_validation, y_validation), shuffle=1)

# Plotting the training and validation loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plotting the training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()  # Adjust the subplots to avoid overlapping
plt.show()
# Evaluate the model on the test set
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Confusion Matrix
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)
confusion_mtx = tf.math.confusion_matrix(y_true, y_pred_classes)

# Plot
plt.figure(figsize=(10, 8))  # Set the figure size
sns.set(font_scale=1.4)    # Adjust font size 

# Create the heatmap with custom colors
sns.heatmap(confusion_mtx, annot=True, fmt='d', cmap='YlOrRd', # Change color scheme to match
            cbar=False,    # Remove color bar
            xticklabels=data['Name'], yticklabels=data['Name'],
            linewidths=2, linecolor='gray')  # Add cell borders

# Add axis labels
plt.xlabel('Predicted Condition', fontsize=16)
plt.ylabel('Actual Condition', fontsize=16)

# Add title
plt.title('Confusion Matrix', fontsize=18)

plt.show()
# STORE THE MODEL AS A PICKLE OBJECT
pickle_out= open("model_trained_new.p","wb")  # wb = WRITE BYTE
pickle.dump(model,pickle_out)
pickle_out.close()
cv2.waitKey(0)