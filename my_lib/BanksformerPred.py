import tensorflow as tf
import time

from .transformer_core import *


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.input_layer = point_wise_feed_forward_network(d_model, dff)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) 
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)
        
        
    def call(self, x, training, mask):

        seq_len = tf.shape(x)[1]

        x = self.input_layer(x)

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, 
                   rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers


        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) 
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)
    
    
    def call(self, x, enc_output, training, 
           look_ahead_mask, padding_mask):
    

        attention_weights = {}


        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                 look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i+1)] = block2


        return x, attention_weights
    
    
    
    
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
    
    def call(self, x, training, mask):

        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)
    
    
    def call(self, x, enc_output, training, 
               look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)


        out1 = self.layernorm1(attn1)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2

    
    

class Transformer(tf.keras.Model):
    
    def __init__(self, num_layers_enc, num_layers_dec, d_model, num_heads, dff, out_dim = 10,
                   rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers_enc, d_model, num_heads, dff, rate)

        self.decoder = Decoder(num_layers_dec, d_model, num_heads, dff, rate)

        self.final_layer = tf.keras.layers.Dense(out_dim, activation=None)

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.results = dict([(x, []) for x in ["loss", "acc", "val_loss", "val_acc", "preds", "val_loss_parts"]])        


    def call(self, inp, training, enc_padding_mask, 
               look_ahead_mask, dec_padding_mask):

        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        tar = tf.ones(shape=(inp.shape[0], 1, 1))


        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)


        final_output = self.final_layer(dec_output) 

        return tf.squeeze(final_output), attention_weights
    
    



#     @tf.function(input_signature=train_step_signature)
    def train_step(self, inp, tar):
#         print(inp.shape, tar.shape)
#         print(inp.dtype, tar.dtype)
        tar_inp = tar #[:, :-1]
        tar_real = tar #[:, 1:]

#         enc_padding_mask = create_masks(inp, tar_inp)
        with tf.GradientTape() as tape:
            predictions, _ = self(inp, 
                                     True, 
                                     None, None, None)

            loss, *_ = self.loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, self.trainable_variables)    
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.train_loss(loss)



    def val_update(self, inp, tar):    

        predictions, _ = self(inp, 
                                 False, 
                                 None, None, None)




        return self.acc_function(tar, predictions), self.loss_function(tar, predictions)


        
    def fit(self, train_batches, x_cv, y_cv, epochs, early_stop=2, print_every=50, ckpt_every=2):
        for epoch in range(epochs):
            start = time.time()

            self.train_loss.reset_states()


            for (batch, (inp, tar)) in enumerate(train_batches):
                self.train_step(inp, tar)

                if batch % print_every == 0:
                    print(f'Epoch {epoch + 1} Batch {batch} Loss {self.train_loss.result():.4f}')
                    

            print(f'Epoch {epoch + 1} Loss {self.train_loss.result():.4f}')

            v_acc_dict, (full_v_loss, *v_loss_parts) = self.val_update(x_cv, y_cv)

            self.results["loss"].append(self.train_loss.result().numpy())
            self.results["val_loss"].append(full_v_loss.numpy())
            self.results["val_acc"].append(v_acc_dict)
            self.results["val_loss_parts"].append(v_loss_parts)

            print(f"** on validation data loss is {full_v_loss.numpy():.4f} ")
            print(v_acc_dict)

            print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')


            if min(self.results["val_loss"] ) < min(self.results["val_loss"][-early_stop:] ):
                print(f"Stopping early, last {early_stop} val losses are: {self.results['val_loss'][-early_stop:]}\
                \nBest was {min(self.results['val_loss'] ):.3f}\n\n")
                break

                
            if (epoch + 1) % ckpt_every == 0:
                ckpt_save_path = self.ckpt_manager.save()
                print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')

 