#region © 2025 Joe Arrowood (JoeWare)
//
// All rights reserved. Reproduction or transmission in whole or in part, in
// any form or by any means, electronic, mechanical, or otherwise, is prohibited
// without the prior written consent of the copyright owner.
//
#endregion

using Tensorflow.NumPy;

using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace JWTensorflowNET.NeuralNetworks
{
    // ----------------------------------------------------
    /// <summary>
    ///     NeuralNetXorEager Description
    /// </summary>

    public class NeuralNetXorEager : absSciSharpBase, IJWTensorBase
    {
        public int num_steps = 10000;

        private NDArray data;
        private NDArray label;

        // ------------------------------------------------

        public BaseConfig InitConfig() => Config = new BaseConfig
        {
            Name = "NN XOR in Eager Mode",
            Enabled = true
        };

        // ------------------------------------------------

        public bool Run()
        {
            PrepareData();
            var loss_value = RunEagerMode();
            return loss_value < 0.0628;
        }

        // ------------------------------------------------

        private float RunEagerMode()
        {
            var num_hidden = 8;
            var display_step = 1000;
            var learning_rate = 0.01f;
            var stddev = 1 / Math.Sqrt(2);
            var labels = tf.constant(label);
            var features = tf.constant(data);

            var hidden_weights = tf.Variable(tf.random.truncated_normal((2, num_hidden), seed: 1, stddev: (float)stddev));

            // ---------------------
            // Shape [4, num_hidden]

            var hidden_activations = tf.nn.relu(tf.matmul(features, hidden_weights));

            var output_weights = tf.Variable(tf.truncated_normal((num_hidden, 1),
                                                                 seed: 17,
                                                                 stddev: (float)(1 / Math.Sqrt(num_hidden))
            ));

            var optimizer = keras.optimizers.SGD(learning_rate);

            // -------------------------------------------
            // Run training for the given number of steps.

            foreach(var step in range(1, num_steps + 1))
            {
                using var g = tf.GradientTape();

                // ------------
                // Shape [4, 1]

                var logits = tf.matmul(hidden_activations, output_weights);

                // ---------
                // Shape [4]

                var predictions = tf.tanh(tf.squeeze(logits));
                var loss = tf.reduce_mean(tf.square(predictions - tf.cast(labels, tf.float32)), name: "loss");

                // ---------------------
                // should stop recording
                // Compute gradients.

                var gradients = g.gradient(loss, output_weights);

                // -----------------------------------
                // Update W and b following gradients.

                optimizer.apply_gradients((gradients, output_weights));

                if(step % display_step == 0)
                {
                    print($"step: {step}, loss: {loss.numpy()}");
                }
            }

            return 0;
        }

        // ------------------------------------------------

        public override void PrepareData()
        {
            data = new float[,]
            {
            {1, 0 },
            {1, 1 },
            {0, 0 },
            {0, 1 }
            };

            label = new float[,]
            {
            {1 },
            {0 },
            {0 },
            {1 }
            };
        }
    }
}
