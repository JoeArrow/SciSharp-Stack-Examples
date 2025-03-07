#region © 2025 Joe Arrowood (JoeWare)
//
// All rights reserved. Reproduction or transmission in whole or in part, in
// any form or by any means, electronic, mechanical, or otherwise, is prohibited
// without the prior written consent of the copyright owner.
//
#endregion

using Tensorflow.Keras.Engine;
using Tensorflow.NumPy;

using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace JWTensorflowNET.ImageProcessing
{
    // ----------------------------------------------------
    /// <summary>
    ///     MnistFnnKerasFunctional Description
    /// </summary>

    public class MnistFnnKerasFunctional : absSciSharpBase, IJWTensorBase
    {
        IModel model;
        NDArray x_train, y_train, x_test, y_test;

        // ------------------------------------------------

        public BaseConfig InitConfig() => Config = new BaseConfig
        {
            Name = "MNIST FNN (Keras Functional)",
            Enabled = true
        };

        // ------------------------------------------------

        public bool Run()
        {
            tf.enable_eager_execution();

            PrepareData();
            BuildModel();
            Train();

            return true;
        }

        // ------------------------------------------------

        public override void PrepareData()
        {
            (x_train, y_train, x_test, y_test) = keras.datasets.mnist.load_data();
            x_train = x_train.reshape((60000, 784)) / 255f;
            x_test = x_test.reshape((10000, 784)) / 255f;
        }

        // ------------------------------------------------

        public override void BuildModel()
        {
            // -----------
            // input layer

            var inputs = keras.Input(shape: 784);

            // ---------------
            // 1st dense layer

            var outputs = layers.Dense(64, activation: keras.activations.Relu).Apply(inputs);

            // ---------------
            // 2nd dense layer

            outputs = layers.Dense(64, activation: keras.activations.Relu).Apply(outputs);

            // ------------
            // output layer

            outputs = layers.Dense(10).Apply(outputs);

            // -----------------
            // build keras model

            model = keras.Model(inputs, outputs, name: "mnist_model");

            // ------------------
            // show model summary

            model.summary();

            // --------------------------------------------------
            // compile keras model into tensorflow's static graph

            model.compile(loss: keras.losses.SparseCategoricalCrossentropy(from_logits: true),
                          optimizer: keras.optimizers.RMSprop(),
                          metrics: new[] { "accuracy" });
        }

        // ------------------------------------------------

        public override void Train()
        {
            // ---------------------------------------
            // train model by feeding data and labels.

            model.fit(x_train, y_train, batch_size: 64, epochs: 2, validation_split: 0.2f);

            // -----------------
            // evluate the model

            model.evaluate(x_test, y_test, verbose: 2);

            // ------------------------
            // save and serialize model

            model.save("mnist_model");
        }
    }
}
