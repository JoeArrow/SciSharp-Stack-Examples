#region © 2025 Joe Arrowood (JoeWare)
//
// All rights reserved. Reproduction or transmission in whole or in part, in
// any form or by any means, electronic, mechanical, or otherwise, is prohibited
// without the prior written consent of the copyright owner.
//
#endregion

using Tensorflow;
using Tensorflow.Keras;
using Tensorflow.Keras.Engine;
using Tensorflow.Common.Types;

using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace JWTensorflowNET.ImageProcessing
{
    // ----------------------------------------------------
    /// <summary>
    ///     LSTMModel Description
    /// </summary>

    internal class LSTMModel : Model
    {
        ILayer lstm;
        ILayer output;
        IOptimizer optimizer;

        // ------------------------------------------------

        public LSTMModel(LSTMModelArgs args)
            : base(args)
        {
            optimizer = keras.optimizers.Adam(args.LearningRate);

            var layers = keras.layers;
            lstm = layers.LSTM(args.NumUnits);
            output = layers.Dense(args.NumClasses);
        }

        // ------------------------------------------------

        protected override Tensors Call(Tensors inputs, Tensors state = null, bool? training = null, IOptionalArgs? optional_args = null)
        {
            // LSTM layer.

            inputs = lstm.Apply(inputs);

            // ---------------------------
            // Output layer (num_classes).

            inputs = output.Apply(inputs);

            if(!training.Value)
            {
                inputs = tf.nn.softmax(inputs);
            }
            
            return inputs;
        }

        // ------------------------------------------------
        /// <summary>
        ///     Optimization process.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>

        public void Optimize(Tensor x, Tensor y)
        {
            using var g = tf.GradientTape();

            // -------------
            // Forward pass.

            var pred = Apply(x, training: true);

            // -------------
            // Compute loss.

            var loss = CrossEntropyLoss(pred, y);

            // ------------------
            // Compute gradients.

            var gradients = g.gradient(loss, TrainableVariables);

            // -----------------------------------
            // Update W and b following gradients.

            optimizer.apply_gradients(zip(gradients, TrainableVariables.Select(x => x as ResourceVariable)));
        }

        // ------------------------------------------------
        /// <summary>
        ///     Cross-Entropy Loss.
        ///     Note that this will apply 'softmax' to the logits.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <returns></returns>

        public Tensor CrossEntropyLoss(Tensor x, Tensor y)
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
        /// <summary>
        ///     Accuracy metric.
        /// </summary>
        /// <param name="yPred"></param>
        /// <param name="yTrue"></param>
        /// <returns></returns>

        public Tensor Accuracy(Tensor yPred, Tensor yTrue)
        {
            // ---------------------------------------
            // Predicted class is the index of highest
            // score in prediction vector (i.e. argmax).
             
            var correct_prediction = tf.equal(tf.math.argmax(yPred, 1), tf.cast(yTrue, tf.int64));
            return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis: -1);
        }

        // ------------------------------------------------

        public Tensor Predict(Tensor x)
        {
            return Apply(x, training: true);
        }
    }
}
