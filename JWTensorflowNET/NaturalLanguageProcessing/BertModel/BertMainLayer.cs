#region © 2025 Joe Arrowood (JoeWare)
//
// All rights reserved. Reproduction or transmission in whole or in part, in
// any form or by any means, electronic, mechanical, or otherwise, is prohibited
// without the prior written consent of the copyright owner.
//
#endregion

using Tensorflow;
using Tensorflow.Keras.Engine;
using Tensorflow.Common.Types;

using static Tensorflow.Binding;

namespace BERT
{
    public partial class BertEmbedding
    {
        public class BertMainLayer : Layer
        {
            BertConfig config;
            BertPooler pooler;
            BertEncoder encoder;
            BertEmbedding embeddings;

            // ------------------------------------------------

            public BertMainLayer(BertConfig config) : base(config)
            {
                this.config = config;
                embeddings = new BertEmbedding(config);
                encoder = new BertEncoder(config);
                pooler = new BertPooler(config);
                StackLayers(embeddings, encoder, pooler);
            }

            // ------------------------------------------------

            public Tensors get_other_ids(Tensor input_ids)
            {
                var batch_size = input_ids.shape[0];
                var attention_mask = tf.reshape(tf.cast(tf.fill(input_ids.shape, 1), dtype: tf.int32), (batch_size, -1));
                var token_type_ids = tf.reshape(tf.cast(tf.fill(input_ids.shape, 0), dtype: tf.int32), (batch_size, -1));
                var position_ids = tf.expand_dims(tf.range(0, input_ids.shape[1]), axis: 0);
                return new Tensor[] { input_ids, attention_mask, token_type_ids, position_ids };
            }

            // ------------------------------------------------

            protected override Tensors Call(Tensors inputs, Tensors state = null, bool? training = null, IOptionalArgs? optional_args = null)
            {
                if(inputs.Length == 1) inputs = get_other_ids(inputs[0]);

                var input_ids = inputs[0];
                var attention_mask = inputs[1];
                var token_type_ids = inputs[2];
                var position_ids = inputs[3];

                var input_shape = input_ids.shape; //bsz seq_len dim
                var embedding_output = embeddings.Apply(new Tensor[] { input_ids, position_ids, token_type_ids });

                var attention_mask_shape = attention_mask.shape;

                var extended_attention_mask = tf.reshape(attention_mask, (attention_mask_shape[0], 1, 1, attention_mask_shape[1]));

                extended_attention_mask = tf.cast(extended_attention_mask, dtype: embedding_output.dtype);

                var one_cst = tf.constant(1.0, dtype: embedding_output.dtype);

                var ten_thousand_cst = tf.constant(-10000.0, dtype: embedding_output.dtype);
                extended_attention_mask = tf.multiply(tf.subtract(one_cst, extended_attention_mask), ten_thousand_cst);

                var encoder_outputs = encoder.Apply(new Tensor[] { embedding_output, extended_attention_mask });

                var pooled_output = pooler.Apply(encoder_outputs);
                return pooled_output;
            }
        }
    }
}
