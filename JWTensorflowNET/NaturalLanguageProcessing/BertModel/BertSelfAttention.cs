#region © 2025 Joe Arrowood (JoeWare)
//
// All rights reserved. Reproduction or transmission in whole or in part, in
// any form or by any means, electronic, mechanical, or otherwise, is prohibited
// without the prior written consent of the copyright owner.
//
#endregion

using Tensorflow;
using Tensorflow.Keras;
using Tensorflow.NumPy;
using Tensorflow.Keras.Engine;
using Tensorflow.Common.Types;
using Tensorflow.Operations.Initializers;

using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace BERT
{
    public partial class BertEmbedding
    {
        public class BertSelfAttention : Layer
        {
            ILayer key;
            ILayer query;
            ILayer value;
            ILayer dropout;
            BertConfig config;
            int all_head_size;
            int attention_head_size;
            double sqrt_att_head_size;

            // ------------------------------------------------

            public BertSelfAttention(BertConfig config) : base(config)
            {
                this.config = config;
                attention_head_size = config.hidden_size / config.num_attention_heads;

                all_head_size = config.num_attention_heads * attention_head_size;
                sqrt_att_head_size = Math.Sqrt(attention_head_size);
                query = keras.layers.Dense(units: all_head_size, kernel_initializer: new TruncatedNormal(config.initializer_range));
                key = keras.layers.Dense(units: all_head_size, kernel_initializer: new TruncatedNormal(config.initializer_range));
                value = keras.layers.Dense(units: all_head_size, kernel_initializer: new TruncatedNormal(config.initializer_range));
                dropout = keras.layers.Dropout(config.hidden_dropout_prob);

                StackLayers(query, key, value);
            }

            // ------------------------------------------------

            public Tensor transpose_for_scores(Tensor tensor, int batch_size)
            {
                tensor = tf.reshape(tensor: tensor, shape: (batch_size, -1, config.num_attention_heads, attention_head_size));
                return tf.transpose(tensor, perm: new int[] { 0, 2, 1, 3 });
            }

            // ------------------------------------------------

            protected override Tensors Call(Tensors inputs, Tensors state = null, bool? training = null, IOptionalArgs? optional_args = null)
            {
                var hidden_states  = inputs[0];
                var attention_mask = inputs[1];

                var batch_size = hidden_states.shape[0];
                var seq_len = hidden_states.shape[1];

                var mixed_query_layer = query.Apply(inputs: hidden_states);
                var key_layer = transpose_for_scores(key.Apply(inputs: hidden_states), (int)batch_size);
                var value_layer = transpose_for_scores(value.Apply(inputs: hidden_states), (int)batch_size);

                var query_layer = transpose_for_scores(mixed_query_layer, (int)batch_size);

                var attention_scores = tf.reshape(tf.batch_matmul(
                    tf.reshape(query_layer, (batch_size * config.num_attention_heads, seq_len, attention_head_size)),
                    tf.reshape(tf.transpose(key_layer, new int[] { 0, 1, 3, 2 }),
                               (batch_size * config.num_attention_heads, attention_head_size, seq_len))),
                               (batch_size, config.num_attention_heads, seq_len, seq_len));

                var dk = tf.cast(np.array(sqrt_att_head_size), dtype: attention_scores.dtype);

                attention_scores = tf.divide(attention_scores, dk);
                attention_scores = tf.add(attention_scores, attention_mask);

                var attention_probs = tf.nn.softmax(logits: attention_scores, axis: -1);
                attention_probs = dropout.Apply(inputs: attention_probs, training: training ?? false);

                var attention_output = tf.reshape(
                    tf.batch_matmul(tf.reshape(attention_probs, (batch_size * config.num_attention_heads, seq_len, seq_len)),
                    tf.reshape(value_layer, (batch_size * config.num_attention_heads, seq_len, attention_head_size))),
                    (batch_size, config.num_attention_heads, seq_len, attention_head_size));

                attention_output = tf.transpose(attention_output, perm: new int[] { 0, 2, 1, 3 });
                attention_output = tf.reshape(attention_output, (batch_size, -1, all_head_size));

                return attention_output;
            }
        }
    }
}
