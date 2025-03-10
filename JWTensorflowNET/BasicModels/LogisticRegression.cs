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
using Tensorflow.NumPy;

using static Tensorflow.Binding;

namespace JWTensorflowNET.BasicModels
{
    // ----------------------------------------------------
    /// <summary>
    ///     LogisticRegression Description
    /// </summary>

    public class LogisticRegression : absSciSharpBase, IJWTensorBase
    {
        public int training_epochs = 10;
        public int? train_size = null;
        public int validation_size = 5000;
        public int? test_size = null;
        public int batch_size = 100;

        private float learning_rate = 0.01f;
        private int display_step = 1;
        float accuracy = 0f;

        Datasets<MnistDataSet> mnist;

        // ------------------------------------------------

        public BaseConfig InitConfig() => Config = new BaseConfig
        {
            Name = "Logistic Regression (Graph)",
            Enabled = true,
            IsImportingGraph = false
        };

        // ------------------------------------------------

        public bool Run(IReq? req = null)
        {
            PrepareData(req);
            tf.compat.v1.disable_eager_execution();
            Train();

            return accuracy > 0.9;
        }

        // ------------------------------------------------

        public override void PrepareData(IReq req)
        {
            var loader = new MnistModelLoader();
            mnist = loader.LoadAsync(".resources/mnist", oneHot: true, trainSize: train_size, validationSize: validation_size, testSize: test_size, showProgressInConsole: true).Result;
        }

        // ------------------------------------------------

        public override void Train()
        {
            // --------------
            // tf Graph Input

            var x = tf.placeholder(tf.float32, (-1, 784)); // mnist data image of shape 28*28=784
            var y = tf.placeholder(tf.float32, (-1, 10)); // 0-9 digits recognition => 10 classes

            // -----------------
            // Set model weights

            var W = tf.Variable(tf.zeros((784, 10)));
            var b = tf.Variable(tf.zeros(10));

            // ---------------
            // Construct model

            var pred = tf.nn.softmax(tf.matmul(x, W) + b); // Softmax

            // ----------------------------------
            // Minimize error using cross entropy

            var cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices: 1));

            // ----------------
            // Gradient Descent

            var optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost);

            // ----------------------------------------------------------
            // Initialize the variables (i.e. assign their default value)

            var init = tf.global_variables_initializer();

            var total_batch = mnist.Train.NumOfExamples / batch_size;

            var sw = new Stopwatch();

            using var sess = tf.Session();
            
            // -------------------
            // Run the initializer

            sess.run(init);

            // --------------
            // Training cycle

            foreach(var epoch in range(training_epochs))
            {
                sw.Start();
                var avg_cost = 0.0f;

                // ---------------------
                // Loop over all batches

                foreach(var i in range(total_batch))
                {
                    var start = i * batch_size;
                    var end = (i + 1) * batch_size;
                    var (batch_xs, batch_ys) = mnist.GetNextBatch(mnist.Train.Data, mnist.Train.Labels, start, end);
                    
                    // --------------------------------------------------------------
                    // Run optimization op (backprop) and cost op (to get loss value)

                    (_, float c) = sess.run((optimizer, cost),
                        (x, batch_xs),
                        (y, batch_ys));

                    // --------------------
                    // Compute average loss

                    avg_cost += c / total_batch;
                }

                sw.Stop();

                // ---------------------------
                // Display logs per epoch step

                if((epoch + 1) % display_step == 0)
                    print($"Epoch: {(epoch + 1):D4} Cost: {avg_cost:G9} Elapsed: {sw.ElapsedMilliseconds}ms");

                sw.Reset();
            }

            print("Optimization Finished!");
            //SaveModel(sess);

            // Test model
            
            var correct_prediction = tf.equal(tf.math.argmax(pred, 1), tf.math.argmax(y, 1));
            
            // Calculate accuracy

            var acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32));
            accuracy = acc.eval(sess, (x, mnist.Test.Data), (y, mnist.Test.Labels));
            print($"Accuracy: {accuracy:F4}");
        }

        // ------------------------------------------------

        public void SaveModel(Session sess)
        {
            var saver = tf.train.Saver();
            saver.save(sess, ".resources/logistic_regression/model.ckpt");
            tf.train.write_graph(sess.graph, ".resources/logistic_regression", "model.pbtxt", as_text: true);

            FreezeGraph.freeze_graph(input_graph: ".resources/logistic_regression/model.pbtxt",
                                     input_saver: "",
                                     input_binary: false,
                                     input_checkpoint: ".resources/logistic_regression/model.ckpt",
                                     output_node_names: "Softmax",
                                     restore_op_name: "save/restore_all",
                                     filename_tensor_name: "save/Const:0",
                                     output_graph: ".resources/logistic_regression/model.pb",
                                     clear_devices: true,
                                     initializer_nodes: "");
        }

        // ------------------------------------------------

        public override void Predict()
        {
            var graph = new Graph().as_default();
            using var sess = tf.Session(graph);
            graph.Import(Path.Join(".resources/logistic_regression", "model.pb"));

            // -------------------
            // restoring the model
            // var saver = tf.train.import_meta_graph("logistic_regression/tensorflowModel.ckpt.meta");
            // saver.restore(sess, tf.train.latest_checkpoint('logistic_regression'));
            
            var pred = graph.OperationByName("Softmax");
            var output = pred.outputs[0];

            var x = graph.OperationByName("Placeholder");
            var input = x.outputs[0];

            // -------
            // predict

            var (batch_xs, batch_ys) = mnist.Train.GetNextBatch(10);
            var results = sess.run(output, new FeedItem(input, batch_xs[np.arange(1)]));

            if((bool)(np.argmax(results[0]) == np.argmax(batch_ys[0])))
            {
                print("predicted OK!");
            }
            else
            {
                throw new ValueError("predict error, should be 90% accuracy");
            }
        }
    }
}
