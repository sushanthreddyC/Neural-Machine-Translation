import numpy as np
import pandas as pd
import typing
from typing import Any, Tuple

import einops
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import tensorflow as tf
import tensorflow_text as tf_text
import re
from sklearn.model_selection import train_test_split
import string
from keras.src.backend import dropout
BATCH_SIZE = 128
max_vocab_size = 50
UNITS = 256
vocab=['exp','sin','cos',')','(','-','+','*','/','^','[START]','[END]']




      
def masked_loss(y_true, y_pred):
    # Calculate the loss for each item in the batch.
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    loss = loss_fn(y_true, y_pred)

    # Mask off the losses on padding.
    mask = tf.cast(y_true != 0, loss.dtype)
    loss *= mask

    # Return the total.
    return tf.reduce_sum(loss)/tf.reduce_sum(mask)


def masked_acc(y_true, y_pred):
    # Calculate the loss for each item in the batch.
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.cast(y_pred, y_true.dtype)

    match = tf.cast(y_true == y_pred, tf.float32)
    mask = tf.cast(y_true != 0, tf.float32)

    return tf.reduce_sum(match)/tf.reduce_sum(mask)


def tokenize(s):
    v_ptr = r"sin|cos|\^|\/|exp|\d|\w|\(|\)|\+|-|\*+"
    out=' '
    return out.join(re.findall(v_ptr,s))

def load_data(path):
    data = pd.read_csv(path, sep="=", names=['function','derivative'])
    data.derivative.fillna('1', inplace=True)
    data.derivative = data.derivative.apply(tokenize)
    data.function   = data.function.apply(tokenize)
    X_train,X_test,Y_train,Y_test = train_test_split(data.function,data.derivative, train_size=0.85,shuffle=True, random_state=1024 )
    context = np.array(X_train)
    target  = np.array(Y_train)

    return target, context,list(X_test),list(Y_test)

def clean(text):

    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text, '[^ 0-9a-z()/*^+-]', '')
    text = tf.strings.strip(text)
    text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
    return text


context_text_processor = tf.keras.layers.TextVectorization(
    standardize=clean,
    max_tokens=max_vocab_size,
    vocabulary=vocab,
    ragged=True)

target_text_processor = tf.keras.layers.TextVectorization(
    standardize=clean,
    max_tokens=max_vocab_size,
    vocabulary=vocab,
    ragged=True)


def process_text(context, target):
    context = context_text_processor(context).to_tensor()
    target = target_text_processor(target)
    targ_in = target[:,:-1].to_tensor()
    targ_out = target[:,1:].to_tensor()
    return (context, targ_in), targ_out

def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, max(plt.ylim())])
    plt.xlabel('Epoch')
    plt.ylabel('CE')
    plt.legend()

def plot_accuracy(history):
    plt.plot(history.history['masked_acc'], label='accuracy')
    plt.plot(history.history['val_masked_acc'], label='val_accuracy')
    plt.ylim([0, max(plt.ylim())])
    plt.xlabel('Epoch ')
    plt.ylabel('CE')
    plt.legend()

class Export(tf.Module):
    def __init__(self, model):
        self.model = model

    @tf.function(input_signature=[tf.TensorSpec(dtype=tf.string, shape=[None])])
    def translate(self, inputs):
        return self.model.translate(inputs)
    
def score(true_derivative: str, predicted_derivative: str) -> int:
    """binary scoring function for model evaluation"""
    return int(true_derivative == predicted_derivative)


''' The Encoder ,decoder and Translator layers are taken as a referance from the official Tensorflow tutorial of Neural Machine Translation'''
class Encoder(tf.keras.layers.Layer):
  def __init__(self, text_processor, units):
    super(Encoder, self).__init__()
    self.text_processor = text_processor
    self.vocab_size = text_processor.vocabulary_size()
    self.units = units

    # The embedding layer converts tokens to vectors
    self.embedding = tf.keras.layers.Embedding(self.vocab_size, units,
                                               mask_zero=True)

    # The RNN layer processes those vectors sequentially.
    self.rnn = tf.keras.layers.Bidirectional(
        merge_mode='sum',
        layer=tf.keras.layers.GRU(units,
                            # Return the sequence and state
                            return_sequences=True,
                            recurrent_initializer='glorot_uniform'))

  def call(self, x):

    # 2. The embedding layer looks up the embedding vector for each token.
    x = self.embedding(x)

    # 3. The GRU processes the sequence of embeddings.
    x = self.rnn(x)

    # 4. Returns the new sequence of embeddings.
    return x

  def convert_input(self, texts):
    texts = tf.convert_to_tensor(texts)
    if len(texts.shape) == 0:
      texts = tf.convert_to_tensor(texts)[tf.newaxis]
    context = self.text_processor(texts).to_tensor()
    context = self(context)
    return context
  
class CrossAttention(tf.keras.layers.Layer):
  def __init__(self, units, **kwargs):
    super().__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(key_dim=units, num_heads=1, **kwargs)
    self.layernorm = tf.keras.layers.LayerNormalization()
    self.add = tf.keras.layers.Add()

  def call(self, x, context):


    attn_output, attn_scores = self.mha(
        query=x,
        value=context,
        return_attention_scores=True)


    # Cache the attention scores for plotting later.
    attn_scores = tf.reduce_mean(attn_scores, axis=1)
    
    self.last_attention_weights = attn_scores

    x = self.add([x, attn_output])
    x = self.layernorm(x)

    return x
  
class Decoder(tf.keras.layers.Layer):
  @classmethod
  def add_method(cls, fun):
    setattr(cls, fun.__name__, fun)
    return fun

  def __init__(self, text_processor, units):
    super(Decoder, self).__init__()
    self.text_processor = text_processor
    self.vocab_size = text_processor.vocabulary_size()
    self.word_to_id = tf.keras.layers.StringLookup(
        vocabulary=text_processor.get_vocabulary(),
        mask_token='', oov_token='[UNK]')
    self.id_to_word = tf.keras.layers.StringLookup(
        vocabulary=text_processor.get_vocabulary(),
        mask_token='', oov_token='[UNK]',
        invert=True)
    self.start_token = self.word_to_id('[START]')
    self.end_token = self.word_to_id('[END]')

    self.units = units


    # 1. The embedding layer converts token IDs to vectors
    self.embedding = tf.keras.layers.Embedding(self.vocab_size,
                                               units, mask_zero=True)

    # 2. The RNN keeps track of what's been generated so far.
    self.rnn = tf.keras.layers.GRU(units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

    # 3. The RNN output will be the query for the attention layer.
    self.attention = CrossAttention(units)

    # 4. This fully connected layer produces the logits for each
    # output token.
    self.output_layer = tf.keras.layers.Dense(self.vocab_size)



@Decoder.add_method
def call(self,
         context, x,
         state=None,
         return_state=False):
  

  # 1. Lookup the embeddings
  x = self.embedding(x)
  

  # 2. Process the target sequence.
  x, state = self.rnn(x, initial_state=state)

  # 3. Use the RNN output as the query for the attention over the context.
  x = self.attention(x, context)
  self.last_attention_weights = self.attention.last_attention_weights
  

  # Step 4. Generate logit predictions for the next token.
  logits = self.output_layer(x)

  if return_state:
    return logits, state
  else:
    return logits
  
@Decoder.add_method
def get_initial_state(self, context):
  batch_size = tf.shape(context)[0]
  start_tokens = tf.fill([batch_size, 1], self.start_token)
  done = tf.zeros([batch_size, 1], dtype=tf.bool)
  embedded = self.embedding(start_tokens)
  return start_tokens, done, self.rnn.get_initial_state(embedded)[0]

@Decoder.add_method
def tokens_to_text(self, tokens):
  words = self.id_to_word(tokens)
  result = tf.strings.reduce_join(words, axis=-1, separator=' ')
  result = tf.strings.regex_replace(result, '^ *\[START\] *', '')
  result = tf.strings.regex_replace(result, ' *\[END\] *$', '')
  return result

@Decoder.add_method
def get_next_token(self, context, next_token, done, state, temperature = 0.0):
  logits, state = self(
    context, next_token,
    state = state,
    return_state=True)

  if temperature == 0.0:
    next_token = tf.argmax(logits, axis=-1)
  else:
    logits = logits[:, -1, :]/temperature
    next_token = tf.random.categorical(logits, num_samples=1)

  # If a sequence produces an `end_token`, set it `done`
  done = done | (next_token == self.end_token)
  # Once a sequence is done it only produces 0-padding.
  next_token = tf.where(done, tf.constant(0, dtype=tf.int64), next_token)

  return next_token, done, state


class Translator(tf.keras.Model):
  @classmethod
  def add_method(cls, fun):
    setattr(cls, fun.__name__, fun)
    return fun

  def __init__(self, units,
               context_text_processor,
               target_text_processor):
    super().__init__()
    # Build the encoder and decoder
    encoder = Encoder(context_text_processor, units)
    decoder = Decoder(target_text_processor, units)

    self.encoder = encoder
    self.decoder = decoder

  def call(self, inputs):
    context, x = inputs
    context = self.encoder(context)
    logits = self.decoder(context, x)

    #TODO(b/250038731): remove this
    try:
      # Delete the keras mask, so keras doesn't scale the loss+accuracy.
      del logits._keras_mask
    except AttributeError:
      pass

    return logits
  

#@title
@Translator.add_method
def translate(self,
              texts,
              *,
              max_length=50,
              temperature=tf.constant(0.0)):
  
  context = self.encoder.convert_input(texts)
  batch_size = tf.shape(context)[0]
  

  next_token, done, state = self.decoder.get_initial_state(context)

  # initialize the accumulator
  tokens = tf.TensorArray(tf.int64, size=1, dynamic_size=True)

  for t in tf.range(max_length):
    # Generate the next token
    next_token, done, state = self.decoder.get_next_token(
        context, next_token, done, state, temperature)

    # Collect the generated tokens
    tokens = tokens.write(t, next_token)

    # if all the sequences are done, break
    if tf.reduce_all(done):
      break

  # Convert the list of generated token ids to a list of strings.
  tokens = tokens.stack()
  
  tokens = einops.rearrange(tokens, 't batch 1 -> batch t')
  

  text = self.decoder.tokens_to_text(tokens)
  

  return text



def main(filepath: str = "train.txt"):
    target_raw, context_raw,X_test,Y_test  = load_data(filepath)
    BUFFER_SIZE = len(context_raw)
    

    is_train = np.random.uniform(size=(len(target_raw),)) < 0.8

    train_raw = (
        tf.data.Dataset
        .from_tensor_slices((context_raw[is_train], target_raw[is_train]))
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE))
    val_raw = (
        tf.data.Dataset
        .from_tensor_slices((context_raw[~is_train], target_raw[~is_train]))
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE))
    
    for char in string.ascii_lowercase:
        vocab.append(char)
    for i in range(10):
        vocab.append(str(i))

    
    def process_text(context, target):
        context = context_text_processor(context).to_tensor()
        target = target_text_processor(target)
        targ_in = target[:,:-1].to_tensor()
        targ_out = target[:,1:].to_tensor()
        return (context, targ_in), targ_out
    
    train_ds = train_raw.map(process_text, tf.data.AUTOTUNE)
    val_ds = val_raw.map(process_text, tf.data.AUTOTUNE)

    # attention_layer = CrossAttention(UNITS)

    # # Attend to the encoded tokens
    # embed = tf.keras.layers.Embedding(target_text_processor.vocabulary_size(),output_dim=UNITS, mask_zero=True)

    # decoder = Decoder(target_text_processor, UNITS)

    model = Translator(UNITS, context_text_processor, target_text_processor)

    model.compile(optimizer='adam', loss=masked_loss, metrics=[masked_acc, masked_loss])

    model.evaluate(val_ds, steps=200, return_dict=True)

    model.summary()

    history = model.fit(
    train_ds.repeat(),
    epochs=100,
    steps_per_epoch = 2000,
    validation_data=val_ds,
    validation_steps = 1000,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=3)])
    
    plot_loss(history)
    plot_accuracy(history)

    export = Export(model)

    tf.saved_model.save(export, 'translator', signatures={'serving_default': export.translate})
    reloaded = tf.saved_model.load('translator')

    max_ind = len(X_test)
    start=0
    width=25000
    final_list=[]

    while(start<max_ind):
        end = min(start+width,max_ind)
        X_test_temp=X_test[start:end]
        predictions=reloaded.translate(tf.constant(X_test_temp))
        final_list.append(predictions)
        start=end
    
    final_pred=[]
    for result in final_list:
        pred = [x.numpy().decode() for x in result]
        final_pred = final_pred + pred
    
    scores = [score(re.sub(r'\s+', '', td), re.sub(r'\s+', '', pd)) for td, pd in zip(Y_test, final_pred)]
    print("Model Accuracy: ", np.mean(scores))

if __name__ == "__main__":
    main()
