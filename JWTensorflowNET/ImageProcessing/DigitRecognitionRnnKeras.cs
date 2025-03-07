#region © 2025 Joe Arrowood (JoeWare)
//
// All rights reserved. Reproduction or transmission in whole or in part, in
// any form or by any means, electronic, mechanical, or otherwise, is prohibited
// without the prior written consent of the copyright owner.
//
#endregion

using Tensorflow;
using Tensorflow.NumPy;

using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace JWTensorflowNET.ImageProcessing
{
    // ----------------------------------------------------
    /// <summary>
    ///     DigitRecognitionRnnKeras Description
    /// </summary>

    public class DigitRecognitionRnnKeras : absSciSharpBase, IJWTensorBase
    {
        float accuracy = 0f;
        int batch_size = 32;
        int display_step = 100;
        int training_steps = 1000;

        LSTMModel lstm_net;
        IDatasetV2 train_data;
        NDArray x_test, y_test, x_train, y_train;

        // ------------------------------------------------

        public BaseConfig InitConfig() => Config = new BaseConfig
        {
            Name = "MNIST RNN (Keras)",
            Enabled = false,
            IsImportingGraph = false
        };

        // ------------------------------------------------

        public bool Run()
        {
            tf.enable_eager_execution();

            PrepareData();
            BuildModel();
            Train();
            Test();

            return accuracy > 0.95;
        }

        // ------------------------------------------------
        /// <summary>
        ///     Build LSTM model.
        /// </summary>

        public override void BuildModel()
        {
            lstm_net = new LSTMModel(new LSTMModelArgs
            {
                NumUnits = 32,
                NumClasses = 10,
                LearningRate = 0.001f
            });
        }

        // ------------------------------------------------

        public override void Train()
        {
            // Run training for the given number of steps.

            foreach(var (step, (batch_x, batch_y)) in enumerate(train_data, 1))
            {
                // Run the optimization to update W and b values.

                lstm_net.Optimize(batch_x, batch_y);

                if(step % display_step == 0)
                {
                    var pred = lstm_net.Predict(batch_x);
                    var loss = lstm_net.CrossEntropyLoss(pred, batch_y);
                    var acc = lstm_net.Accuracy(pred, batch_y);
                    print($"step: {step}, loss: {(float)loss}, accuracy: {(float)acc}");
                }
            }
        }

        // ------------------------------------------------

        public override void Test() {  }

        // ------------------------------------------------

        public override void PrepareData()
        {
            ((x_train, y_train), (x_test, y_test)) = keras.datasets.mnist.load_data();
            // Convert to float32.
            // (x_train, x_test) = (np.array(x_train, np.float32), np.array(x_test, np.float32));
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
