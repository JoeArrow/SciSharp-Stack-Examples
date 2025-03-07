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
using Tensorflow.Keras.ArgsDefinition;

using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace JWTensorflowNET.ImageProcessing
{
    // ----------------------------------------------------
    /// <summary>
    ///     ConvNet Description
    /// </summary>
    
    public class ConvNet : Model
    {
        ILayer conv1;
        ILayer maxpool1;
        ILayer conv2;
        ILayer maxpool2;
        ILayer flatten;
        ILayer fc1;
        ILayer dropout;
        ILayer output;

        // ------------------------------------------------

        public ConvNet(ConvNetArgs args)
            : base(args)
        {
            var layers = keras.layers;

            // ---------------------------------------------------------
            // Convolution Layer with 32 filters and a kernel size of 5.

            conv1 = layers.Conv2D(32, kernel_size: 5, activation: keras.activations.Relu);

            // -------------------------------------------------------------------
            // Max Pooling (down-sampling) with kernel size of 2 and strides of 2.

            maxpool1 = layers.MaxPooling2D(2, strides: 2);

            // ---------------------------------------------------------
            // Convolution Layer with 64 filters and a kernel size of 3.

            conv2 = layers.Conv2D(64, kernel_size: 3, activation: keras.activations.Relu);

            // -------------------------------------------------------------------
            // Max Pooling (down-sampling) with kernel size of 2 and strides of 2. 

            maxpool2 = layers.MaxPooling2D(2, strides: 2);

            // ---------------------------------------------------------------
            // Flatten the data to a 1-D vector for the fully connected layer.

            flatten = layers.Flatten();

            // ----------------------
            // Fully connected layer.

            fc1 = layers.Dense(1024);

            // ----------------------------------------------------------------
            // Apply Dropout (if is_training is False, dropout is not applied).

            dropout = layers.Dropout(rate: 0.5f);

            // -------------------------------
            // Output layer, class prediction.

            output = layers.Dense(args.NumClasses);

            StackLayers(conv1, maxpool1, conv2, maxpool2, flatten, fc1, dropout, output);
        }

        // ------------------------------------------------
        /// <summary>
        ///     Set forward pass.
        /// </summary>
        /// <param name="inputs"></param>
        /// <param name="is_training"></param>
        /// <param name="state"></param>
        /// <returns></returns>

        protected override Tensors Call(Tensors inputs, Tensors state = null, bool? training = null, IOptionalArgs? optional_args = null)
        {
            inputs = tf.reshape(inputs, (-1, 28, 28, 1));
            inputs = conv1.Apply(inputs);
            inputs = maxpool1.Apply(inputs);
            inputs = conv2.Apply(inputs);
            inputs = maxpool2.Apply(inputs);
            inputs = flatten.Apply(inputs);
            inputs = fc1.Apply(inputs);
            inputs = dropout.Apply(inputs);
            inputs = output.Apply(inputs);

            if(!training.Value)
            {
                inputs = tf.nn.softmax(inputs);
            }

            return inputs;
        }
    }

    // ------------------------------------------------

    public class ConvNetArgs : ModelArgs
    {
        public int NumClasses { get; set; }
    }
}
