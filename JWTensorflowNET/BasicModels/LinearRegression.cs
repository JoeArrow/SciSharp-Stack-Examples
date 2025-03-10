#region © 2025 Joe Arrowood (JoeWare)
//
// All rights reserved. Reproduction or transmission in whole or in part, in
// any form or by any means, electronic, mechanical, or otherwise, is prohibited
// without the prior written consent of the copyright owner.
//
#endregion

using JWTensorflowNET.ReqResp;

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

        NDArray train_Input, train_Output;
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

            PrepareData(req);

            // --------------
            // tf Graph Input

            var X = tf.placeholder(tf.float32);
            var Y = tf.placeholder(tf.float32);

            // -----------------
            // Set model weights 

            var b = tf.Variable(req.GetValue<float>("Bias"), name: "bias");
            var W = tf.Variable(req.GetValue<float>("Weight"), name: "weight");

            // ------------------------
            // Construct a linear model

            var pred = tf.add(tf.multiply(X, W), b);

            // ------------------
            // Mean squared error

            var cost = tf.reduce_sum(tf.pow(pred - Y, 2.0f)) / (2.0f * n_samples);

            // ----------------
            // Gradient descent
            // Note, minimize() knows to modify W and b
            // because Variable objects are trainable=True by default

            var optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost);

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
                foreach(var (x, y) in zip<float>(train_Input, train_Output))
                {
                    sess.run(optimizer, (X, x), (Y, y));
                }

                // ---------------------------
                // Display logs per epoch step

                if((epoch + 1) % display_step == 0)
                {
                    var c = sess.run(cost, (X, train_Input), (Y, train_Output));
                }
            }

            var training_cost = sess.run(cost, (X, train_Input), (Y, train_Output));

            // ---------------
            // Testing example

            var test_X = np.array(6.83f, 4.668f, 8.9f, 7.91f, 5.7f, 8.7f, 3.1f, 2.1f);
            var test_Y = np.array(1.84f, 2.273f, 3.2f, 2.831f, 2.92f, 3.24f, 1.35f, 1.03f);
            var testing_cost = sess.run(tf.reduce_sum(tf.pow(pred - Y, 2.0f)) / (2.0f * test_X.shape[0]), (X, test_X), (Y, test_Y));

            var diff = Math.Abs((float)training_cost - (float)testing_cost);

            return diff < 0.01;
        }

        // ------------------------------------------------

        public override void PrepareData(IReq req)
        {
            train_Input = np.array(req.GetValue<float[]>("Input"));
            train_Output = np.array(req.GetValue<float[]>("Output"));

            n_samples = (int)train_Input.shape[0];
        }
    }
}
