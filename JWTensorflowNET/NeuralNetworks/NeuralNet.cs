#region © 2025 Joe Arrowood (JoeWare)
//
// All rights reserved. Reproduction or transmission in whole or in part, in
// any form or by any means, electronic, mechanical, or otherwise, is prohibited
// without the prior written consent of the copyright owner.
//
#endregion

using Tensorflow;
using Tensorflow.Common.Types;
using Tensorflow.Keras;
using Tensorflow.Keras.Engine;

using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace JWTensorflowNET.NeuralNetworks
{
    public partial class FullyConnectedKeras
    {
        public class NeuralNet : Model
        {
            ILayer fc1;
            ILayer fc2;
            ILayer output;

            // ------------------------------------------------

            public NeuralNet(NeuralNetArgs args) :
                base(args)
            {
                var layers = keras.layers;

                // -----------------------------------
                // First fully-connected hidden layer.

                fc1 = layers.Dense(args.NeuronOfHidden1, activation: args.Activation1);

                // ------------------------------------
                // Second fully-connected hidden layer.

                fc2 = layers.Dense(args.NeuronOfHidden2, activation: args.Activation2);

                output = layers.Dense(args.NumClasses);

                StackLayers(fc1, fc2, output);
            }

            // ------------------------------------------------
            // Set forward pass.

            protected override Tensors Call(Tensors inputs, Tensors state = null, bool? training = null, IOptionalArgs? optional_args = null)
            {
                inputs = fc1.Apply(inputs);
                inputs = fc2.Apply(inputs);
                inputs = output.Apply(inputs);
                if(!training.Value)
                    inputs = tf.nn.softmax(inputs);
                return inputs;
            }
        }
    }
}
