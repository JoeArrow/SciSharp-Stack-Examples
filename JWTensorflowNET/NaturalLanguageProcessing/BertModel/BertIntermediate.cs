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
        public class BertIntermediate : Layer
        {
            BertConfig config;
            ILayer dense;

            // ------------------------------------------------

            public BertIntermediate(BertConfig config) : base(config)
            {
                this.config = config;
                dense = keras.layers.Dense(units: config.intermediate_size, kernel_initializer: new TruncatedNormal(config.initializer_range));
                StackLayers(dense);
            }

            // ------------------------------------------------

            public static Tensors gelu(Tensor x)
            {
                var cdf = 0.5 * (1.0 + tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))));
                return x * cdf;
            }

            // ------------------------------------------------

            protected override Tensors Call(Tensors inputs, Tensors state = null, 
                                            bool? training = null, IOptionalArgs? optional_args = null)
            {
                var hidden_states = inputs;

                hidden_states = dense.Apply(hidden_states);
                hidden_states = gelu(hidden_states);

                return hidden_states;
            }
        }
    }
}
