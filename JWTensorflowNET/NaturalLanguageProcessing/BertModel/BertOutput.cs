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
using Tensorflow.Common.Types;
using Tensorflow.Operations.Initializers;

using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace BERT
{
    public partial class BertEmbedding
    {
        public class BertOutput : Layer
        {
            ILayer dense;
            ILayer dropout;
            ILayer LayerNorm;

            // ------------------------------------------------

            public BertOutput(BertConfig config) : base(config)
            {
                dense = keras.layers.Dense(units: config.hidden_size, 
                                           kernel_initializer: new TruncatedNormal(config.initializer_range));

                LayerNorm = keras.layers.LayerNormalization(axis: -1, epsilon: config.layer_norm_eps);
                dropout = keras.layers.Dropout(config.hidden_dropout_prob);

                StackLayers(dense, LayerNorm);
            }

            // ------------------------------------------------

            protected override Tensors Call(Tensors inputs, Tensors state = null, bool? training = null, IOptionalArgs? optional_args = null)
            {
                var hidden_states = inputs[0];
                var input_tensor  = inputs[1];

                hidden_states = dense.Apply(new Tensor[] { hidden_states });
                hidden_states = dropout.Apply(new Tensor[] { hidden_states }, training: training ?? false);

                hidden_states = LayerNorm.Apply(tf.add(hidden_states, input_tensor));

                return hidden_states;
            }
        }
    }
}
