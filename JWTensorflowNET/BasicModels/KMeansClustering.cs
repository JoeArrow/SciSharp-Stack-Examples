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
    ///     KMeansClustering Description
    /// </summary>

    public class KMeansClustering : absSciSharpBase, IJWTensorBase
    {
        public int? test_size = null;
        public int? train_size = null;
        public int validation_size = 5000;

        // -------------------------------
        // The number of samples per batch

        public int batch_size = 1024;

        Datasets<MnistDataSet> mnist;
        NDArray full_data_x;

        int k = 25;           // The number of clusters
        int num_steps = 20;   // Total steps to train
        int num_classes = 10; // The 10 digits

        float accuray_test = 0f;

        // ------------------------------------------------

        public BaseConfig InitConfig() => Config = new BaseConfig
        {
            Enabled = true,
            IsImportingGraph = true,
            Name = "K-means Clustering",
        };

        // ------------------------------------------------

        public bool Run(IReq? req = null)
        {
            PrepareData();
            var graph = ImportGraph();

            using(var sess = tf.Session(graph))
            {
                Train(sess);
            }

            return accuray_test > (req == null ? 0.70 : req.GetValue<float>("Threshold"));
        }

        // ------------------------------------------------

        public override void PrepareData()
        {
            var loader = new MnistModelLoader();

            var setting = new ModelLoadSetting
            {
                OneHot = true,
                TestSize = test_size,
                TrainSize = train_size,
                ShowProgressInConsole = true,
                TrainDir = ".resources/mnist",
                ValidationSize = validation_size,
            };

            mnist = loader.LoadAsync(setting).Result;

            full_data_x = mnist.Train.Data;

            // ------------------------
            // download graph meta data

            string url = "https://raw.githubusercontent.com/SciSharp/TensorFlow.NET/master/graph/kmeans.meta";
            loader.DownloadAsync(url, ".resources/graph", "kmeans.meta").Wait();
        }

        // ------------------------------------------------

        public override Graph ImportGraph()
        {
            var graph = tf.Graph().as_default();

            using(tf.init_scope())  // Ensures TensorFlow initializes properly
            {
                tf.train.import_meta_graph(".resources/graph/kmeans.meta");
            }

            return graph;
        }

        // ------------------------------------------------

        public void Train(Session sess)
        {
            var graph = sess.graph;

            // ------------
            // Input images

            Tensor X = graph.get_operation_by_name("Placeholder");

            // --------------------------------------------------------
            // Labels (for assigning a label to a centroid and testing)

            Tensor Y = graph.get_operation_by_name("Placeholder_1");

            var init_vars = tf.global_variables_initializer();
            Tensor init_op = graph.get_operation_by_name("cond/Merge");
            var train_op = graph.get_operation_by_name("group_deps");
            Tensor avg_distance = graph.get_operation_by_name("Mean");
            Tensor cluster_idx = graph.get_operation_by_name("Squeeze_1");
            NDArray[] result = null;

            sess.run(init_vars, new FeedItem(X, full_data_x));
            sess.run(init_op, new FeedItem(X, full_data_x));

            // --------
            // Training

            var sw = new Stopwatch();

            foreach(var i in range(1, num_steps + 1))
            {
                sw.Restart();
                result = sess.run(new ITensorOrOperation[] { train_op, avg_distance, cluster_idx }, new FeedItem(X, full_data_x));
                sw.Stop();

                if(i % 4 == 0 || i == 1)
                {
                    print($"Step {i}, Avg Distance: {result[1]} Elapse: {sw.ElapsedMilliseconds}ms");
                }
            }

            var idx = result[2].ToArray<Int64>();

            // -------------------------------
            // Assign a label to each centroid
            // Count total number of labels per centroid, using the label of each training
            // sample to their closest centroid (given by 'idx')

            var counts = np.zeros((k, num_classes), np.float32);

            sw.Start();

            foreach(var i in range(idx.Length))
            {
                var x = mnist.Train.Labels[i];
                counts[(int)idx[i], np.argmax(x)] += 1;
            }

            sw.Stop();
            print($"Assign a label to each centroid took {sw.ElapsedMilliseconds}ms");

            // ----------------------------------------------
            // Assign the most frequent label to the centroid

            var labels_map_array = np.argmax(counts, 1);
            var labels_map = tf.convert_to_tensor(labels_map_array);

            // --------------
            // Evaluation ops
            // Lookup: centroid_id -> label

            var cluster_label = tf.nn.embedding_lookup(labels_map, cluster_idx);

            // ----------------
            // Compute accuracy

            var correct_prediction = tf.equal(cluster_label, tf.cast(tf.math.argmax(Y, 1), tf.int32));
            var cast = tf.cast(correct_prediction, tf.float32);
            var accuracy_op = tf.reduce_mean(cast);

            // ----------
            // Test Model

            var (test_x, test_y) = (mnist.Test.Data, mnist.Test.Labels);

            accuray_test = sess.run(accuracy_op, new FeedItem(X, test_x), new FeedItem(Y, test_y));

            print($"Test Accuracy: {accuray_test}");
        }
    }
}
