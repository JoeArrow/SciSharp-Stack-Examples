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

namespace BERT
{
    public partial class BertEmbedding
    {
        public class BertEncoder : Layer
        {
            BertConfig config;
            List<ILayer> layers;

            // ------------------------------------------------

            public BertEncoder(BertConfig config) : base(config)
            {
                this.config = config;
                layers = new List<ILayer>();
                
                for(int i = 0; i < config.num_hidden_layers; i++) { layers.Add(new BertLayer(config)); }

                StackLayers(layers.ToArray());
            }

            // ------------------------------------------------

            protected override Tensors Call(Tensors inputs, Tensors state = null, bool? training = null, IOptionalArgs? optional_args = null)
            {
                var hidden_states = inputs[0];
                var attention_mask = inputs[1];

                foreach(var layer in layers)
                {
                    var layer_outputs = layer.Apply(new Tensor[] { hidden_states, attention_mask });
                    hidden_states = layer_outputs;
                }

                return hidden_states;
            }
        }
    }
}
