#region © 2025 Joe Arrowood (JoeWare)
//
// All rights reserved. Reproduction or transmission in whole or in part, in
// any form or by any means, electronic, mechanical, or otherwise, is prohibited
// without the prior written consent of the copyright owner.
//
#endregion

using Tensorflow;
using Tensorflow.Keras;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Saving;
using Tensorflow.Common.Types;
using Tensorflow.Operations.Initializers;

using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace BERT
{
    // ----------------------------------------------------
    /// <summary>
    ///     BertEmbedding Description
    /// </summary>

    public partial class BertEmbedding : Layer
    {
        ILayer dropout;
        ILayer LayerNorm;
        BertConfig config;
        IVariableV1 weight;
        IVariableV1 position_embeddings;
        IVariableV1 token_type_embeddings;

        // ------------------------------------------------

        public BertEmbedding(BertConfig config) : base(config)
        {
            this.config = config;
            LayerNorm = keras.layers.LayerNormalization(axis: -1, epsilon: config.layer_norm_eps);
            dropout = keras.layers.Dropout(config.hidden_dropout_prob);

            StackLayers(LayerNorm);
        }

        // ------------------------------------------------

        public override void build(KerasShapesWrapper input_shape)
        {
            tf_with(ops.name_scope("word_embeddings"), scope =>
            {
                weight = add_weight(name: "weight",
                                    shape: (config.vocab_size, config.hidden_size),
                                    initializer: new TruncatedNormal(config.initializer_range)
                    );

            });

            tf_with(ops.name_scope("token_type_embeddings"), scope =>
            {
                token_type_embeddings = add_weight(name: "token_type_embedding",
                                    shape: (config.type_vocab_size, config.hidden_size),
                                    initializer: new TruncatedNormal(config.initializer_range)
                    );

            });

            tf_with(ops.name_scope("position_embeddings"), scope =>
            {
                position_embeddings = add_weight(name: "position_embedding",
                                    shape: (config.max_position_embeddings, config.hidden_size),
                                    initializer: new TruncatedNormal(config.initializer_range)
                    );

            });

            base.build(input_shape);
        }

        // ------------------------------------------------

        protected override Tensors Call(Tensors inputs, Tensors state = null, bool? training = null, IOptionalArgs? optional_args = null)
        {
            var input_ids = inputs[0];
            var position_ids = inputs[1];
            var token_type_ids = inputs[2];

            var inputs_embeds = tf.gather(weight.AsTensor(), input_ids);
            var position_embeds = tf.gather(position_embeddings.AsTensor(), indices: position_ids);
            var token_type_embeds = tf.gather(token_type_embeddings.AsTensor(), indices: token_type_ids);
            var final_embeddings = inputs_embeds + token_type_embeds + position_embeds;
            final_embeddings = LayerNorm.Apply(final_embeddings);
            final_embeddings = dropout.Apply(final_embeddings, training: training ?? false);

            return final_embeddings;
        }
    }
}
