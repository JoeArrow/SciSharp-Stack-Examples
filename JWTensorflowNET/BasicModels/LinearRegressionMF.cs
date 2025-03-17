#region © 2025 Joe Arrowood (JoeWare)
//
// All rights reserved. Reproduction or transmission in whole or in part, in
// any form or by any means, electronic, mechanical, or otherwise, is prohibited
// without the prior written consent of the copyright owner.
//
#endregion


using System;
using System.Collections.Generic;
using System.Linq;

using JWTensorflowNET.ReqResp;

using Tensorflow;
using Tensorflow.NumPy;

using static Tensorflow.Binding;

namespace JWTensorflowNET.BasicModels
{
    // ----------------------------------------------------
    /// <summary>
    ///     LinearRegressionMF Description
    /// </summary>

    public class LinearRegressionMF : IJWTensorBase
    {
        public BaseConfig Config { get; set; }
        private Session _session;
        private Tensor _X, _Y, _pred, _cost, _optimizer;
        private Tensor _W, _b;

        private const float learning_rate = 0.01f;
        private const int training_epochs = 1000;
        private const int display_step = 100;

        private NDArray train_X, train_Y;
        private int n_samples;

        // ------------------------------------------------

        public BaseConfig InitConfig() => Config = new BaseConfig
        {
            Enabled = true,
            IsImportingGraph = false,
            Name = "Multi-Feature Linear Regression"
        };

        // ------------------------------------------------

        public bool Run(IReq? req = null)
        {
            tf.compat.v1.disable_eager_execution();
            PrepareData(req);
            BuildModel();
            Train();
            return true;
        }

        // ------------------------------------------------

        public void BuildModel()
        {
            _X = tf.placeholder(tf.float32, shape: (-1, 1)); // WorkVolume input
            _Y = tf.placeholder(tf.float32); // RunTime output

            _W = tf.Variable(tf.random.uniform((1, 1)), name: "weight");
            _b = tf.Variable(tf.random.uniform((1, 1)), name: "bias");

            _pred = tf.add(tf.matmul(_X, _W), _b);
            _cost = tf.reduce_mean(tf.pow(_pred - _Y, 2));
            _optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(_cost);
        }

        // ------------------------------------------------

        public void Train()
        {
            using var sess = tf.Session();
            sess.run(tf.global_variables_initializer());

            for(int epoch = 0; epoch < training_epochs; epoch++)
            {
                sess.run(_optimizer, new FeedItem(_X, train_X), new FeedItem(_Y, train_Y));

                if((epoch + 1) % display_step == 0)
                {
                    float c = sess.run(_cost, new FeedItem(_X, train_X), new FeedItem(_Y, train_Y));
                    Console.WriteLine($"Epoch {epoch + 1}, Cost: {c}");
                }
            }

            Console.WriteLine("Training Completed!");
            _session = sess;
        }

        // ------------------------------------------------

        public void Predict()
        {
            Console.Write("Enter Job Name: ");
            string jobName = Console.ReadLine();

            Console.Write("Enter Work Volume: ");
            int workVolume = int.Parse(Console.ReadLine());

            var input = np.array(new float[,] { { workVolume } });
            var prediction = _session.run(_pred, new FeedItem(_X, input));

            Console.WriteLine($"Predicted runtime for {jobName}: {prediction[0]:F2} seconds");
        }

        // ------------------------------------------------

        public void Test() => Predict();

        // ------------------------------------------------

        public string FreezeModel()
        {
            throw new NotImplementedException("FreezeModel is not implemented yet.");
        }

        // ------------------------------------------------

        public Graph BuildGraph() => throw new NotImplementedException();

        // ------------------------------------------------

        public Graph ImportGraph() => throw new NotImplementedException();

        // ------------------------------------------------

        public void PrepareData(IReq req = null)
        {
            var data = new List<(string JobName, int WorkVolume, int RunTime)>
            {
                ("JobA", 100, 30),
                ("JobB", 200, 50),
                ("JobC", 150, 40),
                ("JobD", 300, 80),
                ("JobE", 250, 65),
                ("JobF", 180, 45)
            };

            train_X = np.array(data.Select(d => new float[] { d.WorkVolume }).ToArray());
            train_Y = np.array(data.Select(d => (float)d.RunTime).ToArray());

            n_samples = (int)train_X.shape[0];
        }
    }
}
