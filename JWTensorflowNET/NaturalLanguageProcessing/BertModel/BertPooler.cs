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
        public class BertPooler : Layer
        {
            ILayer dense;
            BertConfig config;

            // ------------------------------------------------

            public BertPooler(BertConfig config) : base(config)
            {
                this.config = config;
                dense = keras.layers.Dense(units: config.hidden_size,
                                           kernel_initializer: new TruncatedNormal(config.initializer_range),
                                           activation: keras.activations.Tanh);
                StackLayers(dense);
            }

            // ------------------------------------------------

            public static Tensors gelu(Tensor x)
            {
                var cdf = 0.5 * (1.0 + tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))));
                return x * cdf;
            }

            // ------------------------------------------------

            protected override Tensors Call(Tensors inputs, Tensors state = null, bool? training = null, IOptionalArgs? optional_args = null)
            {
                var hidden_states = inputs[0];

                var shape = hidden_states.shape;
                var first_token_tensor = tf.slice<int, int>(hidden_states, 
                                                            new int[] { 0, 0, 0 }, 
                                                            new int[] { (int)shape[0], 1, (int)shape[2] });

                return gelu(dense.Apply(tf.reshape(first_token_tensor, (shape[0], shape[2]))));
            }
        }
    }
}
