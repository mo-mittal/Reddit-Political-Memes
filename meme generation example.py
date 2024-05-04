'''
This script sets up most of the foundational elements needed to train a model that can generate captions for memes based on their images. 
It uses a combination of CNN for image feature extraction and LSTM for sequence processing, which is a common architecture for image captioning tasks.
For my specific project, I use it to generate political memes based on images collected from the `Bernie Sanders` subreddit and captions generated manually using GPT-4. It can be found here: https://drive.google.com/drive/folders/16-Iiq1_4ORRTOC7WcZFi66z2fFx1wGkY?usp=drive_link

Using the InceptionV3 model, it extracts features from images. These features capture various aspects of the image content, which are crucial for generating relevant captions.
The script handles text by loading captions, tokenizing them, converting them into sequences, and then padding these sequences. This prepares the textual data to be used in training alongside image data.
It constructs a neural network that combines image features and processed captions. The network uses these combined features to learn to predict the next word in a caption, effectively learning how to generate text based on the image content.
Lastly, the script trains the model on paired image and caption data. This training allows the model to learn associations between the content of images and the words in captions.
'''

# ! pip install --upgrade tensorflow

from google.colab import drive
drive.mount('/content/drive')

import tensorflow as tf

import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
import pandas as pd

def get_caption_for_image(img_path):
    caption_path = img_path.replace('/images/', '/captions/').replace('.jpg', '.txt').replace('.png', '.txt')
    if os.path.exists(caption_path):
        with open(caption_path, 'r') as file:
            return file.read()
    else:
        return None

df = pd.DataFrame({
    'image_path': image_paths,
    'caption': [get_caption_for_image(path) for path in image_paths]
})

df = df.dropna(subset=['caption'])

df.head()

processed_images = []
for idx, img_path in enumerate(df['image_path']): 
    try:
        img = image.load_img(img_path, target_size=(299, 299))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        processed_images.append(preprocess_input(img_array))
    except Exception as e:
        print(f"Error processing image {img_path}: {e}")

assert len(processed_images) == len(df['caption']), "The number of processed images and captions does not match"

# If the assertion passes without an error, we stack the images
processed_images_array = np.vstack(processed_images)
processed_images_array.shape
df['image_path'].count()
df['caption'].count()

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Set the maximum number of words to keep, based on word frequency
max_words = 5000  # size of your vocabulary

# create the tokenizer with the top 'max_words' words
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(df['caption'])

# convert the captions to sequences of integers
sequences = tokenizer.texts_to_sequences(df['caption'])

# pad the sequences to have the same length
max_sequence_len = 50 
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_len, padding='post')
# 'padded_sequences' contains padded integer sequences representing each caption

# padded_sequences.shape
# padded_sequences[1]

"""## Creating the Neural Network"""

from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Add, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import numpy as np
import os

# load InceptionV3 model pre-trained on ImageNet data, exclude the top fully connected layers
base_model = InceptionV3(weights='imagenet', include_top=False)

# define a model that will output the feature vector for each image using the InceptionV3 layers
model_feature = Model(inputs=base_model.input, outputs=base_model.output)
features = model_feature.predict(processed_images_array)

# apply GlobalAveragePooling to convert (64, 8, 8, 2048) to (64, 2048)
pooling_model = GlobalAveragePooling2D()
flattened_features = pooling_model(features)

processed_images_array.shape

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Add, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

embedding_dim = 256

# inputs
image_features_input = Input(shape=(2048,))  # Update the shape based on the flattened features
caption_input = Input(shape=(max_sequence_len,))

# define the image feature layer using a dense layer to decrease dimensionality
image_features_layer = Dense(embedding_dim, activation='relu')(image_features_input)

# Define the caption sequential model (LSTM)
captions_embedding = Embedding(    input_dim=max_words,        # Size of the vocab
                                  output_dim=embedding_dim,   # Dimension of dense embedding
                                  # input_length=max_sequence_len  # Length of input sequences
                              )(caption_input)
captions_lstm = LSTM(embedding_dim, return_sequences=True)(captions_embedding)

# combine the inputs from both models
decoder = Add()([image_features_layer, captions_lstm])
decoder = Dense(embedding_dim, activation='relu')(decoder)
outputs = Dense(max_words, activation='softmax')(decoder)

# define the model
model = Model(inputs=[image_features_input, caption_input], outputs=outputs)

# Ccompile the model
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
model.summary()

## time distributed

from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Add, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

embedding_dim = 256

image_features_input = Input(shape=(2048,))  # update the shape based on the flattened features
caption_input = Input(shape=(max_sequence_len,))

# image feature layer using a dense layer to decrease dimensionality
image_features_layer = Dense(embedding_dim, activation='relu')(image_features_input)

# the caption sequential model (LSTM)
captions_embedding = Embedding(    input_dim=max_words,        # Size of the vocab
                                  output_dim=embedding_dim,   # Dimension of dense embedding
                                  # input_length=max_sequence_len  # Length of input sequences
                              )(caption_input)

captions_lstm = LSTM(embedding_dim, return_sequences=True)(captions_embedding)

# Apply a TimeDistributed wrapper to the Dense layer: applies the same Dense layer independently to each time step of the sequence
decoder_output = TimeDistributed(Dense(max_words, activation='softmax'))(captions_lstm)

model2 = Model(inputs=[image_features_input, caption_input], outputs=decoder_output)
model2.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
model2.summary()

# outputs.shape

# creating captions_target by shifting the captions_input
captions_target = np.hstack([padded_sequences[:, 1:], np.zeros((len(padded_sequences), 1))])

# one-hot encoding the target captions (this can be memory intensive with a large vocabulary)
from tensorflow.keras.utils import to_categorical
captions_target_one_hot = np.zeros((captions_target.shape[0], captions_target.shape[1], max_words), dtype='float32')

# verify the dimensions first
# print("captions_target.shape:", captions_target.shape)
# print("max_words:", max_words)
# print("captions_target_one_hot.shape:", captions_target_one_hot.shape)

for i, seq in enumerate(captions_target):
    for j, word_index in enumerate(seq):
        # Convert word_index to integer
        word_index = int(word_index)
        # Proceed only if word_index is within the range of [0, max_words)
        if word_index < max_words:
            captions_target_one_hot[i, j, word_index] = 1

for i, seq in enumerate(captions_target):
    for j, word_index in enumerate(seq):
        # Convert word_index to integer
        word_index = int(word_index)
        # Proceed only if word_index is within the range of [0, max_words)
        if 0 <= word_index < max_words:
            captions_target_one_hot[i, j, word_index] = 1

outputs = Dense(max_words, activation='softmax')(decoder)

outputs

print(flattened_features.shape)  # should output something like (num_samples, 2048)
print(padded_sequences.shape)    # should output something like (num_samples, max_sequence_len)
print(captions_target_one_hot.shape)  # should output (num_samples, max_sequence_len, max_words)

history = model.fit(
    [flattened_features, padded_sequences],  # image_features is an array of shape (num_samples, 2048)
    captions_target_one_hot,                   # captions_target is an array of shape (num_samples, num_words) with one-hot encoding
    epochs=20,                         # Number of epochs to train the model
    batch_size=64,                     # Number of samples per gradient update
    validation_split=0.2               # Fraction of the data to be used as validation data
)
