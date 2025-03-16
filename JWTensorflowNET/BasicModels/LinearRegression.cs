#region © 2025 Joe Arrowood (JoeWare)
//
// All rights reserved. Reproduction or transmission in whole or in part, in
// any form or by any means, electronic, mechanical, or otherwise, is prohibited
// without the prior written consent of the copyright owner.
//
#endregion

using JWTensorflowNET.ReqResp;

using Tensorflow;
using Tensorflow.NumPy;

using static Tensorflow.Binding;

namespace JWTensorflowNET.BasicModels
{
    // ----------------------------------------------------
    /// <summary>
    ///     LinearRegression Description
    /// </summary>

    public class LinearRegression : absSciSharpBase, IJWTensorBase
    {
        public int training_epochs = 1000;

        // ----------
        // Parameters

        float learning_rate = 0.01f;
        int display_step = 50;

        NDArray train_X, train_Y;
        int n_samples;

        // ------------------------------------------------

        public BaseConfig InitConfig() => Config = new BaseConfig
        {
            Enabled = true,
            IsImportingGraph = false,
            Name = "Linear Regression (Graph)",
        };

        // ------------------------------------------------

        public bool Run(IReq? req = null)
        {
            tf.compat.v1.disable_eager_execution();

            // -------------
            // Training Data

            PrepareData();

            // --------------
            // tf Graph Input

            var X = tf.placeholder(tf.float32, shape: (-1, 3));
            var Y = tf.placeholder(tf.float32);

            // -----------------
            // Set model weights 

            var W = tf.Variable(tf.random.normal(new Shape(3, 1)), name: "weights");
            var b = tf.Variable(-0.73f, name: "bias");

            // ------------------------
            // Construct a linear model

            var pred = tf.add(tf.matmul(X, W), b);

            // ------------------
            // Mean squared error

            var cost = tf.reduce_sum(tf.pow(pred - Y, 2.0f)) / (2.0f * n_samples);

            // ----------------
            // Gradient descent
            // Note, minimize() knows to modify W and b
            // because Variable objects are trainable=True by default

            //var optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost);

            var grads_and_vars = tf.train.GradientDescentOptimizer(learning_rate).compute_gradients(cost);
            //var clipped_grads = new List<(Tensor, RefVariable)>();
            
            foreach(var (grad, var) in grads_and_vars)
            {
                clipped_grads.Add(((Tensor, RefVariable))(tf.clip_by_value(grad, -1.0f, 1.0f), var));
            }

            var optimizer = tf.train.GradientDescentOptimizer(learning_rate).apply_gradients(clipped_grads);

            // ----------------------------------------------------------
            // Initialize the variables (i.e. assign their default value)

            var init = tf.global_variables_initializer();

            // --------------
            // Start training

            using var sess = tf.Session();

            // -------------------
            // Run the initializer

            sess.run(init);

            // ---------------------
            // Fit all training data

            for(int epoch = 0; epoch < training_epochs; epoch++)
            {
                var train_X_jagged = train_X.numpy();

                foreach(var row in train_X.numpy())
                {
                    var x_array = np.array(new float[] { row[0], row[1], row[2] });
                    var x_reshaped = x_array.reshape(new Shape(1, 3));
                    sess.run(optimizer, (X, x_reshaped), (Y, train_Y.numpy()[0]));
                }

                // ---------------------------
                // Display logs per epoch step

                if((epoch + 1) % display_step == 0)
                {
                    var c = sess.run(cost, (X, train_X), (Y, train_Y));
                }
            }

            var training_cost = sess.run(cost, (X, train_X), (Y, train_Y));

            // ---------------
            // Testing example

            var test_X = np.array(new float[,]
            {
                { 6.83f,  55f,  3f },
                { 4.668f, 40f,  2f },
                { 8.9f,   70f,  5f },
                { 7.91f,  65f,  4f },
                { 5.7f,   45f,  2f },
                { 8.7f,   85f,  6f },
                { 3.1f,   30f,  1f },
                { 2.1f,   20f,  1f }
            });

            var test_Y = np.array(60.83f, 40.66f, 80.90f, 70.91f, 50.70f, 80.70f, 30.10f, 20.10f);

            var test_X_reshaped = np.reshape(test_X, new Shape(-1, 3));
            var testing_cost = sess.run(tf.reduce_sum(tf.pow(pred - Y, 2.0f)) / (2.0f * test_X.shape[0]), (X, test_X_reshaped), (Y, test_Y));

            var diff = Math.Abs((float)training_cost - (float)testing_cost);

            return diff < 0.01;
        }

        // ------------------------------------------------

        public override void PrepareData()
        {
            //  Volume,   CPU,  Disk I/O

            train_X = np.array(new float[,]
            {
                { 3.3f,    50f,  2f },
                { 4.4f,    40f,  3f },
                { 5.5f,    60f,  5f },
                { 6.71f,   30f,  1f },
                { 6.93f,   80f,  6f },
                { 4.168f,  55f,  3f },
                { 9.779f,  70f,  7f },
                { 6.182f,  45f,  2f },
                { 7.59f,   65f,  4f },
                { 2.167f,  35f,  2f },
                { 7.042f,  85f,  6f },
                { 10.791f, 90f,  8f },
                { 5.313f,  25f,  1f },
                { 7.997f,  75f,  7f },
                { 5.654f,  60f,  3f },
                { 9.27f,   95f,  9f },
                { 3.1f,    20f,  2f }
            });

            // -------
            // Runtime

            train_Y = np.array(30.30f, 40.40f, 50.50f, 60.71f, 60.97f,
                               40.16f, 90.77f, 60.18f, 70.59f, 20.16F,
                               70.04F, 100.79F, 50.31F, 70.99F, 50.65F,
                               90.27F, 30.10F);

            n_samples = (int)train_X.shape[0];
        }
    }
}
