#region © 2025 Joe Arrowood (JoeWare)
//
// All rights reserved. Reproduction or transmission in whole or in part, in
// any form or by any means, electronic, mechanical, or otherwise, is prohibited
// without the prior written consent of the copyright owner.
//
#endregion

using JWTensorflowNET.ReqResp;

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

        public bool Run(IReq? req = null)
        {
            tf.enable_eager_execution();

            PrepareData(req);

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

        public override void PrepareData(IReq req)
        {
            train_X = np.array(req.GetValue<float[]>("Input"));
            train_Y = np.array(req.GetValue<float[]>("Output"));
        }
    }
}
