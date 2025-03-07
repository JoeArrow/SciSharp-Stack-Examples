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
    ///     NearestNeighbor Description
    /// </summary>

    public class NearestNeighbor : absSciSharpBase, IJWTensorBase
    {
        NDArray Xtr; 
        NDArray Ytr; 
        NDArray Xte;
        NDArray Yte;
        public int? TestSize = null;
        Datasets<MnistDataSet> mnist;
        public int? TrainSize = null;
        public int ValidationSize = 5000;

        // ------------------------------------------------

        public BaseConfig InitConfig() => Config = new BaseConfig
        {
            Name = "Nearest Neighbor",
            Enabled = true,
            IsImportingGraph = false
        };

        // ------------------------------------------------

        public bool Run(IReq? req = null)
        {
            tf.compat.v1.disable_eager_execution();

            // --------------
            // tf Graph Input

            var xtr = tf.placeholder(tf.float32, (-1, 784));
            var xte = tf.placeholder(tf.float32, 784);

            // ----------------------------------------------
            // Nearest Neighbor calculation using L1 Distance
            // Calculate L1 Distance

            var distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), reduction_indices: 1);

            // Prediction: Get min distance index (Nearest neighbor)

            var pred = tf.arg_min(distance, 0);

            float accuracy = 0f;

            // ----------------------------------------------------------
            // Initialize the variables (i.e. assign their default value)

            var init = tf.global_variables_initializer();

            using(var sess = tf.Session())
            {
                // -------------------
                // Run the initializer

                sess.run(init);

                PrepareData();

                foreach(int i in range((int)Xte.shape[0]))
                {
                    // --------------------
                    // Get nearest neighbor

                    long nn_index = sess.run(pred, (xtr, Xtr), (xte, Xte[i]));

                    // -----------------------------------------------------------------
                    // Get nearest neighbor class label and compare it to its true label

                    int index = (int)nn_index;

                    if(i % 10 == 0 || i == 0)
                    {
                        print($"Test {i} Prediction: {np.argmax(Ytr[index])} True Class: {np.argmax(Yte[i])}");
                    }

                    // ------------------
                    // Calculate accuracy

                    if(np.argmax(Ytr[index]) == np.argmax(Yte[i]))
                    {
                        accuracy += 1f / Xte.shape[0];
                    }
                }

                print($"Accuracy: {accuracy}");
            }

            return accuracy > 0.8;
        }

        // ------------------------------------------------

        public override void PrepareData()
        {
            var loader = new MnistModelLoader();
            mnist = loader.LoadAsync(".resources/mnist", oneHot: true, trainSize: TrainSize, validationSize: ValidationSize, testSize: TestSize, showProgressInConsole: true).Result;
            
            // ------------------------------------
            // In this example, we limit mnist data

            (Xtr, Ytr) = mnist.Train.GetNextBatch(TrainSize == null ? 5000 : TrainSize.Value / 100); // 5000 for training (nn candidates)
            (Xte, Yte) = mnist.Test.GetNextBatch(TestSize == null ? 200 : TestSize.Value / 100); // 200 for testing
        }
    }
}
