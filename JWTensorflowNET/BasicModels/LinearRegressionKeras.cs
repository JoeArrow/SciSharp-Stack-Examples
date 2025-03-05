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

namespace JWTensorflowNET.BasicModels
{
    // ----------------------------------------------------
    /// <summary>
    ///     LinearRegressionKeras Description
    /// </summary>

    public class LinearRegressionKeras : absSciSharpBase, IJWTensorBase
    {
        NDArray train_X, train_Y;
        ICallback? result;

        // ------------------------------------------------

        public BaseConfig InitConfig() => Config = new BaseConfig
        {
            Name = "Linear Regression (Keras)",
            Enabled = true,
            IsImportingGraph = false
        };

        // ------------------------------------------------

        public bool Run()
        {
            tf.enable_eager_execution();

            PrepareData();

            BuildModel();

            return true;
        }

        // ------------------------------------------------

        public override void BuildModel()
        {
            var inputs = keras.Input(shape: 1);
            var outputs = layers.Dense(1).Apply(inputs);
            var model = keras.Model(inputs, outputs);

            model.summary();

            model.compile(loss: keras.losses.MeanSquaredError(),
                          optimizer: keras.optimizers.SGD(0.005f),
                          metrics: new[] { "acc" });

            result = model.fit(train_X, train_Y, epochs: 10);

            var weights = model.TrainableVariables;
            print($"weight: {weights[0].numpy()}, bias: {weights[1].numpy()}");
        }

        // ------------------------------------------------

        public override void PrepareData()
        {
            train_X = np.array(3.3f,   4.4f,   5.5f,   6.71f,
                               6.93f,  4.168f, 9.779f, 6.182f,
                               7.59f,  2.167f, 7.042f, 10.791f,
                               5.313f, 7.997f, 5.654f, 9.27f,
                               3.1f);

            train_Y = np.array(1.7f,   2.76f,  2.09f,  3.19f,
                               1.694f, 1.573f, 3.366f, 2.596f,
                               2.53f,  1.221f, 2.827f, 3.465f,
                               1.65f,  2.904f, 2.42f,  2.94f,
                               1.3f);
        }
    }
}
