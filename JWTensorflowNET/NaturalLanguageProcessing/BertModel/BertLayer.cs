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
        public class BertLayer : Layer
        {
            BertConfig config;
            BertAttention attention;
            BertIntermediate intermediate;
            BertOutput bert_output;

            // ------------------------------------------------

            public BertLayer(BertConfig config) : base(config)
            {
                this.config = config;

                attention = new BertAttention(config);
                intermediate = new BertIntermediate(config);
                bert_output = new BertOutput(config);

                StackLayers(attention, intermediate, bert_output);
            }

            // ------------------------------------------------

            protected override Tensors Call(Tensors inputs, Tensors state = null, bool? training = null, IOptionalArgs? optional_args = null)
            {
                var hidden_states = inputs[0];
                var attention_mask = inputs[1];

                var attention_output = attention.Apply(new Tensor[] { hidden_states, attention_mask }, training: training ?? false);

                var intermediate_output = intermediate.Apply(attention_output, training: training ?? false);
                var layer_output = bert_output.Apply(new Tensor[] { intermediate_output, attention_output }, training: training ?? false);

                return layer_output;
            }
        }
    }
}
