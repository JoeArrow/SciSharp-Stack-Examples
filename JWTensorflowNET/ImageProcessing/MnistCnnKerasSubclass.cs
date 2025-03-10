#region © 2025 Joe Arrowood (JoeWare)
//
// All rights reserved. Reproduction or transmission in whole or in part, in
// any form or by any means, electronic, mechanical, or otherwise, is prohibited
// without the prior written consent of the copyright owner.
//
#endregion

using JWTensorflowNET.ReqResp;

using Tensorflow;
using Tensorflow.Keras.Engine;
using Tensorflow.NumPy;

using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace JWTensorflowNET.ImageProcessing
{
    // ----------------------------------------------------
    /// <summary>
    ///     MnistCnnKerasSubclass Description
    /// </summary>

    public class MnistCnnKerasSubclass : absSciSharpBase, IJWTensorBase
    {
        // -------------------------
        // MNIST dataset parameters.

        int num_classes = 10;

        // --------------------
        // Training parameters.

        float learning_rate = 0.001f;
        int training_steps = 100;
        int batch_size = 128;
        int display_step = 10;

        float accuracy_test = 0.0f;

        IDatasetV2 train_data;
        NDArray x_test, y_test, x_train, y_train;

        // ------------------------------------------------

        public BaseConfig InitConfig() => Config = new BaseConfig
        {
            Name = "MNIST CNN (Keras Subclass)",
            Enabled = true,
            IsImportingGraph = false
        };

        // ------------------------------------------------

        public bool Run(IReq? req = null)
        {
            tf.enable_eager_execution();

            PrepareData(req);

            Train();

            // Test();

            return accuracy_test > 0.85;
        }

        // ------------------------------------------------

        public override void Train()
        {
            // ---------------------------
            // Build neural network model.

            var conv_net = new ConvNet(new ConvNetArgs
            {
                NumClasses = num_classes
            });

            // ---------------
            // ADAM optimizer. 

            var optimizer = keras.optimizers.Adam(learning_rate);

            // -------------------------------------------
            // Run training for the given number of steps.

            foreach(var (step, (batch_x, batch_y)) in enumerate(train_data, 1))
            {
                // ----------------------------------------------
                // Run the optimization to update W and b values.

                run_optimization(conv_net, optimizer, batch_x, batch_y);

                if(step % display_step == 0)
                {
                    var pred = conv_net.Apply(batch_x);
                    var loss = cross_entropy_loss(pred, batch_y);
                    var acc = accuracy(pred, batch_y);
                    print($"step: {step}, loss: {(float)loss}, accuracy: {(float)acc}");
                }
            }

            // -----------------------------
            // Test model on validation set.

            {
                x_test = x_test["::100"];
                y_test = y_test["::100"];
                var pred = conv_net.Apply(x_test);
                accuracy_test = (float)accuracy(pred, y_test);
                print($"Test Accuracy: {accuracy_test}");
            }

            conv_net.save_weights("weights.h5");
        }

        // ------------------------------------------------

        public override void Test()
        {
            var conv_net = new ConvNet(new ConvNetArgs
            {
                NumClasses = num_classes
            });

            // --------------------------
            // Test model on testing set.

            {
                x_test = x_test["::100"];
                y_test = y_test["::100"];
            }
        }

        // ------------------------------------------------

        void run_optimization(ConvNet conv_net, IOptimizer optimizer, Tensor x, Tensor y)
        {
            using var g = tf.GradientTape();
            var pred = conv_net.Apply(x, training: true);
            var loss = cross_entropy_loss(pred, y);

            // ------------------
            // Compute gradients.

            var gradients = g.gradient(loss, conv_net.TrainableVariables);

            // -----------------------------------
            // Update W and b following gradients.

            optimizer.apply_gradients(zip(gradients, conv_net.TrainableVariables.Select(x => x as ResourceVariable)));
        }

        // ------------------------------------------------

        Tensor cross_entropy_loss(Tensor x, Tensor y)
        {
            // -------------------------------------------------------
            // Convert labels to int 64 for tf cross-entropy function.

            y = tf.cast(y, tf.int64);

            // --------------------------------------------------
            // Apply softmax to logits and compute cross-entropy.

            var loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels: y, logits: x);

            // ------------------------------
            // Average loss across the batch.

            return tf.reduce_mean(loss);
        }

        // ------------------------------------------------

        Tensor accuracy(Tensor y_pred, Tensor y_true)
        {
            // -----------------------------------------
            // # Predicted class is the index of highest
            // score in prediction vector (i.e. argmax).

            var correct_prediction = tf.equal(tf.math.argmax(y_pred, 1), tf.cast(y_true, tf.int64));
            return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis: -1);
        }

        // ------------------------------------------------

        public override void PrepareData(IReq req)
        {
            ((x_train, y_train), (x_test, y_test)) = keras.datasets.mnist.load_data();

            // -----------------------------------------------
            // Normalize images value from [0, 255] to [0, 1].

            (x_train, x_test) = (x_train / 255.0f, x_test / 255.0f);

            train_data = tf.data.Dataset.from_tensor_slices(x_train, y_train);
            train_data = train_data.repeat()
                .shuffle(5000)
                .batch(batch_size)
                .prefetch(1)
                .take(training_steps);
        }
    }
}
