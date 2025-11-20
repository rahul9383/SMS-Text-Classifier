# -*- coding: utf-8 -*-
"""
Original file is located at
    https://colab.research.google.com/drive/1ZVNB_nKvhUFahVgWHzjW-wnrpYDwVcYf
"""

import tensorflow as tf
import pandas as pd
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

!wget https://cdn.freecodecamp.org/project-data/sms/train-data.tsv
!wget https://cdn.freecodecamp.org/project-data/sms/valid-data.tsv

train_file_path = "train-data.tsv"
test_file_path = "valid-data.tsv"

train_df = pd.read_csv(train_file_path, sep='\t', header=None, names=['type', 'msg'])
test_df = pd.read_csv(test_file_path, sep='\t', header=None, names=['type', 'msg'])

train_df['type'] = train_df['type'].map({'ham': 0, 'spam': 1})
test_df['type'] = test_df['type'].map({'ham': 0, 'spam': 1})


train_labels = train_df['type'].values
train_msg = train_df['msg'].values
test_labels = test_df['type'].values
test_msg = test_df['msg'].values

max_len = 50
vocab_size = 10000

encoder = tf.keras.layers.TextVectorization(
    max_tokens=vocab_size,
    output_mode='int',
    output_sequence_length=max_len
  )

encoder.adapt(train_msg)


model = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=64,
        mask_zero=True
    ),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy']
)


history = model.fit(
    x=train_msg,
    y=train_labels,
    validation_data=(test_msg, test_labels),
    epochs=10,
    validation_steps=30
)

def predict_message(pred_text):

  prediction_prob = model.predict(tf.constant([pred_text]))[0][0]

  if prediction_prob < 0.5:
      label = "ham"
  else:
      label = "spam"

  return (label)

pred_text = "sale today! to stop texts call 98912460324"

prediction = predict_message(pred_text)
print(prediction)

def test_predictions():
  test_messages = ["how are you doing today",
                   "sale today! to stop texts call 98912460324",
                   "i dont want to go. can we try it a different day? available sat",
                   "our new mobile video service is live. just install on your phone to start watching.",
                   "you have won £1000 cash! call to claim your prize.",
                   "i'll bring it tomorrow. don't forget the milk.",
                   "wow, is your arm alright. that happened to me one time too"
                  ]

  test_answers = ["ham", "spam", "ham", "spam", "spam", "ham", "ham"]
  passed = True

  for msg, ans in zip(test_messages, test_answers):
    prediction = predict_message(msg)
    print(prediction,ans)
    if prediction!= ans:
      passed = False

  if passed:
    print("Model has Passed the Test Cases")
  else:
    print("Model hasn't Passed the Test Cases")

test_predictions()

