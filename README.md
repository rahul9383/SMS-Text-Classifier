# SMS-Text-Classifier

This repository contains a complete workflow for building an SMS spam classifier using TensorFlow and Keras.  
The project trains a neural network to distinguish between **spam** and **ham (non-spam)** messages using a dataset of labeled SMS texts.

## 🗂 Dataset

The notebook uses the freeCodeCamp SMS spam dataset:

'train-data.tsv' – training data  
'valid-data.tsv' – validation

Each file typically contains:

'msg' / 'text' – the SMS message  
'type' / 'label' – 'spam(1)' or 'ham(0)' 

The dataset is downloaded with:


!wget https://cdn.freecodecamp.org/project-data/sms/train-data.tsv
!wget https://cdn.freecodecamp.org/project-data/sms/valid-data.tsv
