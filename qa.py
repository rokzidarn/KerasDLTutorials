# QUESTION ANSWERING

from keras.models import Model
from keras import layers
from keras import Input
import numpy as np

# multi-input model, need to merge these by adding, concatenating them
# text (Embedding, LSTM) + question (Embedding, LSTM) -> Concatenate -> answer (Dense, softmax for prediction)
# TODO: 238 + 239 + 240
# TODO: other 418+

text_vocabulary_size = 5000  # NOTE: if size to too big it fails
question_vocabulary_size = 2000
answer_vocabulary_size = 250

# model
text_input = Input(shape=(None,), dtype='int32', name='text')  # name attribute = variable called "text"
embedded_text = layers.Embedding(64, text_vocabulary_size)(text_input)
encoded_text = layers.LSTM(32)(embedded_text)

question_input = Input(shape=(None,), dtype='int32', name='question')
embedded_question = layers.Embedding(32, question_vocabulary_size)(question_input)
encoded_question = layers.LSTM(16)(embedded_question)

concatenated = layers.concatenate([encoded_text, encoded_question], axis=-1)
answer = layers.Dense(answer_vocabulary_size, activation='softmax')(concatenated)

model = Model([text_input, question_input], answer)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

num_samples = 1000
max_length = 100

# data
text = np.random.randint(1, text_vocabulary_size, size=(num_samples, max_length))
question = np.random.randint(1, question_vocabulary_size, size=(num_samples, max_length))
answers = np.random.randint(0, 1, size=(num_samples, answer_vocabulary_size))

model.fit([text, question], answers, epochs=10, batch_size=128)
history = model.fit({'text': text, 'question': question}, answers, epochs=10, batch_size=128)

history_dict = history.history
print("\n", history_dict['acc'][-1])
