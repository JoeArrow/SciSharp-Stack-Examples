#region © 2025 Joe Arrowood (JoeWare)
//
// All rights reserved. Reproduction or transmission in whole or in part, in
// any form or by any means, electronic, mechanical, or otherwise, is prohibited
// without the prior written consent of the copyright owner.
//
#endregion

using System.Diagnostics;

using JWTensorflowNET.ReqResp;

using Tensorflow;

using static Tensorflow.Binding;

namespace JWTensorflowNET.NeuralNetworks
{
    // ----------------------------------------------------
    /// <summary>
    ///     FullyConnectedInQueue Description
    /// </summary>

    public class FullyConnectedInQueue : absSciSharpBase, IJWTensorBase
    {
        Tensor input = null;
        Tensor y_true = null;
        Tensor loss_op = null;
        Tensor accuracy = null;
        Operation train_op = null;
        Tensor x_inputs_data = null;
        Tensor y_inputs_data = null;

        // ------------------------------------------------

        public BaseConfig InitConfig() => Config = new BaseConfig
        {
            Enabled = false,
            IsImportingGraph = false,
            Name = "Fully Connected Neural Network In Queue",
        };

        // ------------------------------------------------

        public override Graph BuildGraph()
        {
            var g = tf.get_default_graph();

            // ---------------------------------
            // We build our small model: a basic
            // two layers neural net with ReLU

            tf_with(tf.variable_scope("queue"), delegate
            {
                // -----------------
                // enqueue 5 batches

                var q = tf.FIFOQueue(capacity: 5, dtype: tf.float32);

                // -------------------------------------------
                // We use the "enqueue" operation so 1 element
                // of the queue is the full batch

                var enqueue_op = q.enqueue(x_inputs_data);
            });

            return g;
        }

        // ------------------------------------------------

        public override void PrepareData()
        {
            // --------------------------------------------------------
            // batches of 128 samples, each containing 1024 data points

            x_inputs_data = tf.random.normal(new[] { 128, 1024 }, mean: 0, stddev: 1);

            // --------------------------------
            // We will try to predict this law:
            // predict 1 if the sum of the elements is positive and 0 otherwise

            y_inputs_data = tf.cast(tf.reduce_sum(x_inputs_data, axis: 1, keepdims: true) > 0, tf.int32);
        }

        // ------------------------------------------------

        public bool Run(IReq? req = null)
        {
            PrepareData();
            BuildGraph();
            Train();
            return true;
        }

        // ------------------------------------------------

        public override void Train()
        {
            var sw = new Stopwatch();
            sw.Start();

            var sess = tf.Session();

            // --------------
            // init variables

            sess.run(tf.global_variables_initializer());

            // ----------------------------------
            // check the accuracy before training

            var (x_input, y_input) = sess.run((x_inputs_data, y_inputs_data));
            sess.run(accuracy, (input, x_input), (y_true, y_input));

            // --------
            // training

            foreach(var i in range(5000))
            {
                // --------------------------------------
                // by sampling some input data (fetching)

                (x_input, y_input) = sess.run((x_inputs_data, y_inputs_data));
                var (_, loss) = sess.run((train_op, loss_op), (input, x_input), (y_true, y_input));

                // ---------------------------
                // We regularly check the loss

                if(i % 500 == 0)
                {
                    print($"iter:{i} - loss:{loss}");
                }
            }

            // ------------------------------------
            // Finally, we check our final accuracy

            (x_input, y_input) = sess.run((x_inputs_data, y_inputs_data));
            sess.run(accuracy, (input, x_input), (y_true, y_input));

            print($"Time taken: {sw.Elapsed.TotalSeconds}s");
        }
    }
}
