#### 1. Load modules
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.engine import Model
from keras.layers import Flatten, Dense, Reshape, LocallyConnected2D, Conv2D
from keras_vggface.vggface import VGGFace
from sklearn.metrics import accuracy_score, precision_score,recall_score,f1_score, roc_auc_score
from statsmodels.stats.proportion import proportions_ztest
from keras.layers import Layer
import keras.backend as K

#### 2. Load Image Data
## Face feature selection 
## "FACE_" means full face;  "EYES_" means eyse; "MOUTH_" means mouth; "EYEBROWS_" means eyebrows; 
## "Nose_" means nose; "OUTLINE_" means outline; 'WHOLE_' means composite face

import pickle
INPUT =  "allfeatures_parse_224.pickle"#"allfeatures_224.pickle"
with open(INPUT, 'rb') as f:
    data= pickle.load(f)
part_array = data['MOUTH_PARSE'] + data['EYES'] + data['NOSE_PARSE'] + data['EYEBROWS_PARSE'] +data['OUTLINE_PARSE']
mouth_array = data['MOUTH_PARSE']
eyes_array = data['EYES']
nose_array = data['NOSE_PARSE']
eyebrows_array = data['EYEBROWS_PARSE']
outline_array = data['OUTLINE_PARSE']
file_name = data['filename']
print('Loaded Parts:', file_name.shape, part_array.shape)
#Image.fromarray(part_array[152])
data_name = pd.DataFrame(file_name, columns = ["img_name"])  ## a list of file names of the processed photos 

#### 3. Load Label Data

## label selection (for each face perception task)
## "appr": approachability; "comp": competence; "trust":trust; "labm": lab meat customer; "persona":starbucks persona
##	"starb": starbucks customer; "gender":gender
label = "trust" ## pick the label

## There are 3384 images in the datasets, but only 3383 images have been labeled 
## and the order of image name is different in the data_name and label datasets
file = "label.xlsx"
df_mean = pd.read_excel(file)
df_mean.drop(columns=["Unnamed: 0"],axis=1, inplace=True)

## old data to extract the image name in the worksheet "RA1"
dir_file ="label1.xlsx"
RA1 = pd.read_excel(dir_file, sheet_name= 'RA1')
df_mean["img_name"]=RA1["img_name"]

#### 4. Select and Split the Data
df = df_mean.copy() ## adjust this line according to the actual needs
df['img_name'] = pd.Series(df['img_name'].str.replace('%','_')) ## replace "%" with "_" for consistency with the other document
results = pd.merge(data_name, df, on=['img_name']) ## realign the order of ranking based on the order of photos
## checking with the data
#results
#results[results["img_name"] == "libolun_01.png"]
#len(results[results["persona"]>=3])

#identify the order of the 3383 photos in "part_array" and align with "results"
index_data = []
Findex =  data_name['img_name'].tolist()
for name in results['img_name']:
    index = Findex.index(name)
    index_data.append(index)

## X_data
x_all = part_array[index_data]  
num_samples = x_all.shape[0]

## Y_data
scores = np.array(results[label]).reshape(-1,1)

## RA1 delete the redundant samples 
def del_samples_RA(x_images, results_sel, label, coef=1):
  """
  input:
  x_images - images to be selected 
  results_sel - results(including all the lables) to be selected 
  label - chosen label, e.g., gender
  coef - idicates the percentage to choose  e.g., 0.1, means top 10% of the datapoints
  criteria - the criteria to classify data

  output: 
  X: images_data: the selected images 
  Y: labels_data: the selected label results with specific lable.,e.g., gender
  
  """
  y_scores = results_sel[label]

  if label == "gender" :
    dict_filter = 0.5
  else:
    dict_filter = 3

  label_0 = np.sum(y_scores < dict_filter )
  label_1 = np.sum(y_scores >= dict_filter)

  y_scores = np.array(y_scores).reshape(-1,1)
  sorted_idx = np.argsort(y_scores, axis=0)[::-1, 0]
  x_images = x_images[sorted_idx]

  if label_0 > label_1:
    num = label_1
  else:
    num = label_0
  
  chosen_num = int(num * coef)

  positive_x = x_images[:chosen_num]
  negative_x = x_images[-chosen_num:]

  images_data = np.concatenate([positive_x, negative_x], axis=0)
  labels_data = np.concatenate([np.ones(chosen_num, dtype=np.int),
                           np.zeros(chosen_num, dtype=np.int)], axis=0)
  
  df ={"X":images_data,"Y":labels_data}    
  return df

### To make classes balanced, and chose the top XX % of the data points
x_images = del_samples_RA(x_all, results, label)["X"]
y_labels = del_samples_RA(x_all, results, label)["Y"]

##split into main datasets including training and test datasets
##make sure the percentage of each class is consistent in the splitting (stratify=y)
x_main, x_test, y_main, y_test,= train_test_split(x_images, y_labels, stratify=y_labels, test_size=0.15, random_state=567)

# split main datasets into training and validation, and hold classes balanced
##make sure the percentage of each class is consistent in the splitting (stratify=y) 
train_data, val_data, train_label, val_label = train_test_split(x_main, y_main, 
        stratify=y_main, test_size=0.18, random_state=567)
print(train_data.shape, val_data.shape, x_test.shape)

# Generate dummies for labels in the training , validation and holdouts sets 
dummies = pd.get_dummies(y_test) # Classification 
products = dummies.columns
y_test=dummies.values

dummies = pd.get_dummies(train_label) # Classification 
products = dummies.columns
y_train= dummies.values

dummies = pd.get_dummies(val_label) # Classification 
products = dummies.columns
y_val=dummies.values


#### 5. Transfer Learning (VGGFace)
#ref:https://machinelearningmastery.com/how-to-perform-face-recognition-with-vggface2-convolutional-neural-network-in-keras/
#https://github.com/rcmalli/keras-vggface

#custom parameters
nb_class = 2
hidden_dim = 64
    
#VGG
vgg_model = VGGFace(include_top=False, input_shape=(224, 224, 3))

for layer in vgg_model.layers:
    layer.trainable = False

# Add attention layer to the deep learning network
x = vgg_model.get_layer('pool5').output
pt_depth = vgg_model.get_output_shape_at(0)[-1]
attn_layer = LocallyConnected2D(1, kernel_size = (1,1), padding = 'valid', activation = 'sigmoid')(x)
# fan it out to all of the channels
up_c2_w = np.ones((1, 1, 1, pt_depth))
up_c2 = Conv2D(pt_depth, kernel_size = (1,1), padding = 'same', activation = 'linear', use_bias = False, weights = [up_c2_w])
#up_c2.trainable = False
attn_layer = up_c2(attn_layer)

x = Flatten(name='flatten')(attn_layer)
x = Dense(hidden_dim, activation='relu', name='fc6')(x)
x = Dense(hidden_dim, activation='relu', name='fc7')(x)
out = Dense(nb_class, activation='sigmoid', name='fc8')(x)

train_model = Model(vgg_model.input, out)
train_model.summary()
print('This is the number of trainable weights after freezing the conv base',len(train_model.trainable_weights))

import time
tic = time.time()

train_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) #categorical_crossentropy
# optimizer= keras.optimizers.Adam()

## Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=(0.5,2.),
    fill_mode = 'nearest')

test_datagen = ImageDataGenerator(rescale=1./255) ## no enhancement for the validation set 

## Early stopping 
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, 
        verbose=1, mode='auto', restore_best_weights=True)

history = train_model.fit(train_datagen.flow(train_data,y_train), 
                          validation_data=test_datagen.flow(val_data,y_val), 
                          callbacks=[monitor],verbose=2,epochs=1) #

print ("Done!")
toc = time.time()
print(f"running time {toc-tic} s") 

#### 6. Model Evaluation and Prediction
def prediction(X, Y):
    X_aug = X.astype('float32') / 255
    pred= train_model.predict(X_aug)
    predict_classes = np.argmax(pred, axis=1)
    expected_classes = np.argmax(Y, axis=1)
    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(expected_classes, predict_classes)
    # precision tp / (tp + fp)
    precision = precision_score(expected_classes, predict_classes)
    # recall: tp / (tp + fn)
    recall = recall_score(expected_classes, predict_classes)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(expected_classes, predict_classes)
    # ROC AUC
    auc = roc_auc_score(expected_classes, predict_classes)
    #perform one proportion z-test
    p_value = proportions_ztest(count=int(accuracy*len(Y)), nobs=len(Y), value=0.50)[1]
    num_0 = (expected_classes==0).sum()
    num_1 = (expected_classes==1).sum()
    data={"num_0":[num_0],
          "num_1":[num_1],
          "Accuracy":[accuracy],
          "p_value": [p_value],
          "Precision": [precision],
          "Recall":[recall],
          "F1":[f1],
          "AUC":[auc]}
    data=pd.DataFrame(data).transpose()
    return data

#time
print ("Evaluation Start \n")
acc_train = prediction(train_data, y_train)
print ("Training Completed")
print(acc_train)
acc_val = prediction(val_data, y_val)
print ("\n","Validation:")
print(acc_val)
acc_test = prediction(x_test, y_test)
print ("\n","Test:")
print(acc_test)

## Repeat step 6 to obtain the marginal prediction accuracies and normalize the vector
## ['mouth_acc', 'eyes_acc', 'nose_acc', 'eyebrows_acc', outline_acc']
## Update the aggregated face input based on the relative importance weights
## part_array = mouth_acc*data['MOUTH_PARSE'] + eyes_acc*data['EYES'] + nose_acc*data['NOSE_PARSE'] + eyebrows_acc*data['EYEBROWS_PARSE'] +outline_acc*data['OUTLINE_PARSE']
## Repeat step 4 to 6 until convergence
