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

namespace JWTensorflowNET.BasicModels
{
    // ----------------------------------------------------
    /// <summary>
    ///     LinearRegressionEager Description
    /// </summary>

    public class LinearRegressionEager : absSciSharpBase, IJWTensorBase
    {
        int training_steps = 1000;

        // Parameters
        float learning_rate = 0.01f;
        int display_step = 100;

        NDArray train_X, train_Y;
        int n_samples;

        // ------------------------------------------------

        public BaseConfig InitConfig() => Config = new BaseConfig
        {
            Name = "Linear Regression (Eager)",
            Enabled = true,
            IsImportingGraph = false
        };

        // ------------------------------------------------

        public bool Run()
        {
            tf.enable_eager_execution();

            // Training Data

            PrepareData();

            Train();

            return true;
        }

        // ------------------------------------------------

        public override void Train()
        {
            // -----------------
            // Set model weights 
            // We can set a fixed init value in order to debug
            // var rnd1 = rng.randn<float>();
            // var rnd2 = rng.randn<float>();

            var W = tf.Variable(-0.06f, name: "weight");
            var b = tf.Variable(-0.73f, name: "bias");
            var optimizer = keras.optimizers.SGD(learning_rate);

            // Run training for the given number of steps.
            foreach(var step in range(1, training_steps + 1))
            {
                // ----------------------------------------------
                // Run the optimization to update W and b values.
                // Wrap computation inside a GradientTape for automatic differentiation.

                using var g = tf.GradientTape();
                var pred = W * train_X + b;

                // ------------------
                // Mean square error.

                var sub = pred - train_Y;
                var p = tf.pow(sub, 2);
                var s = tf.reduce_sum(p);
                var loss = s / (2 * n_samples);

                // ---------------------
                // should stop recording
                // Compute gradients.

                var gradients = g.gradient(loss, (W, b));

                // -----------------------------------
                // Update W and b following gradients.

                optimizer.apply_gradients(zip(gradients, (W, b)));

                if(step % display_step == 0)
                {
                    pred = W * train_X + b;
                    loss = tf.reduce_sum(tf.pow(pred - train_Y, 2)) / (2 * n_samples);
                    print($"step: {step}, loss: {loss.numpy()}, W: {W.numpy()}, b: {b.numpy()}");
                }
            }
        }

        // ------------------------------------------------

        public override void PrepareData()
        {
            train_X = np.array(3.3f,   4.4f,   5.5f,   6.71f,
                               6.93f,  4.168f, 9.779f, 6.182f,
                               7.59f,  2.167f, 7.042f, 10.791f,
                               5.313f, 7.997f, 5.654f, 9.27f,
                               3.1f);

            train_Y = np.array(1.7f,   2.76f,  2.09f,  3.19f,
                               1.694f, 1.573f, 3.366f, 2.596f,
                               2.53f,  1.221f, 2.827f, 3.465f,
                               1.65f,  2.904f, 2.42f,  2.94f,
                               1.3f);

            n_samples = (int)train_X.shape[0];
        }
    }
}
