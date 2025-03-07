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

namespace BERT
{
    public partial class BertEmbedding
    {
        public class BertAttention : Layer
        {
            BertConfig config;
            BertSelfAttention self_attention;
            BertSelfOutput dense_output;
            public BertAttention(BertConfig config) : base(config)
            {
                this.config = config;
                self_attention = new BertSelfAttention(config);
                dense_output = new BertSelfOutput(config);
                StackLayers(self_attention, dense_output);
            }

            protected override Tensors Call(Tensors inputs, Tensors state = null, bool? training = null, IOptionalArgs? optional_args = null)
            {
                var input_tensor = inputs[0];
                var attention_mask = inputs[1];

                var self_outputs = self_attention.Apply(new Tensor[] { input_tensor, attention_mask }, training: training ?? false);
                var attention_output = dense_output.Apply(new Tensor[] { self_outputs, input_tensor }, training: training ?? false);

                return attention_output;
            }
        }
    }
}
