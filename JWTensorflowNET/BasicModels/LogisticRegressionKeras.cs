#region © 2025 Joe Arrowood (JoeWare)
//
// All rights reserved. Reproduction or transmission in whole or in part, in
// any form or by any means, electronic, mechanical, or otherwise, is prohibited
// without the prior written consent of the copyright owner.
//
#endregion

using Tensorflow.Keras;
using Tensorflow.Keras.Engine;

using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace JWTensorflowNET.BasicModels
{
    // ----------------------------------------------------
    /// <summary>
    ///     LogisticRegressionKeras Description
    /// </summary>

    public class LogisticRegressionKeras : absSciSharpBase, IJWTensorBase
    {
        ICallback result;

        // ------------------------------------------------

        public BaseConfig InitConfig() => Config = new BaseConfig
        {
            Name = "Logistic Regression (Keras)",
            Enabled = true,
            IsImportingGraph = false
        };

        // ------------------------------------------------

        public bool Run()
        {
            tf.enable_eager_execution();

            // -------------------
            // Prepare MNIST data.

            var ((x_train, y_train), (x_test, y_test)) = keras.datasets.mnist.load_data();

            // -----------------------------------------------
            // Normalize images value from [0, 255] to [0, 1].

            (x_train, x_test) = (x_train / 255f, x_test / 255f);

            var model = keras.Sequential(new List<ILayer>
            {
                // -----------------------------------------------------
                // Flatten images to 1-D vector of 784 features (28*28).

                keras.layers.Flatten(),
                keras.layers.Dense(10, activation: "softmax")
            });

            // ---------------------------------------------
            // Compile the model, specifying that SGD should
            // be used to train and the cross entropy loss function
            // should be used. Also keep track of accuracy throughout training.

            model.compile(optimizer: keras.optimizers.SGD(0.01f),
                          loss: keras.losses.SparseCategoricalCrossentropy(from_logits: true),
                          metrics: new[] { "accuracy" });

            result = model.fit(x_train, y_train, epochs: 5);

            // model.evaluate(x_test, np.argmax(y_test));

            var predicted = model.predict(x_test);

            return true;
        }
    }
}
