#region © 2025 Joe Arrowood (JoeWare)
//
// All rights reserved. Reproduction or transmission in whole or in part, in
// any form or by any means, electronic, mechanical, or otherwise, is prohibited
// without the prior written consent of the copyright owner.
//
#endregion

using JWTensorflowNET.ReqResp;

using SciSharp.Models;
using SciSharp.Models.ImageClassification;

using System.Diagnostics;

using Tensorflow;
using Tensorflow.NumPy;

using static Tensorflow.Binding;

namespace JWTensorflowNET.ImageProcessing
{
    // ----------------------------------------------------
    /// <summary>
    ///     DigitRecognitionCNN Description
    /// </summary>

    public class DigitRecognitionCNN : absSciSharpBase, IJWTensorBase
    {
        Datasets<MnistDataSet> mnist;

        float accuracy_test = 0f;

        NDArray x_train, y_train;
        NDArray x_valid, y_valid;
        NDArray x_test, y_test;

        // ------------------------------------------------

        public BaseConfig InitConfig() => Config = new BaseConfig
        {
            Name = "MNIST CNN (Graph)",
            Enabled = true
        };

        // ------------------------------------------------

        public bool Run(IReq? req = null)
        {
            PrepareData(req);
            Train();
            Test();
            Predict();

            return accuracy_test > 0.95;
        }

        // ------------------------------------------------

        public override void Train()
        {
            // using wizard to train model
            var wizard = new ModelWizard();
            var task = wizard.AddImageClassificationTask<CNN>(new TaskOptions
            {
                InputShape = (28, 28, 1),
                NumberOfClass = 10,
            });
            task.SetModelArgs(new ConvArgs
            {
                NumberOfNeurons = 128
            });
            task.Train(new TrainingOptions
            {
                Epochs = 5,
                TrainingData = new FeatureAndLabel(x_train, y_train),
                ValidationData = new FeatureAndLabel(x_valid, y_valid)
            });
        }

        // ------------------------------------------------

        public override void Test()
        {
            var wizard = new ModelWizard();
            var task = wizard.AddImageClassificationTask<CNN>(new TaskOptions
            {
                ModelPath = @"image_classification_cnn_v1\saved_model.pb"
            });
            var result = task.Test(new TestingOptions
            {
                TestingData = new FeatureAndLabel(x_test, y_test)
            });
            accuracy_test = result.Accuracy;
        }

        // ------------------------------------------------

        public override void Predict()
        {
            // predict image
            var wizard = new ModelWizard();
            var task = wizard.AddImageClassificationTask<CNN>(new TaskOptions
            {
                LabelPath = @"image_classification_cnn_v1\labels.txt",
                ModelPath = @"image_classification_cnn_v1\saved_model.pb"
            });

            var input = x_test["0:1"];
            var result = task.Predict(input);
            long output = np.argmax(y_test[0]);
            Debug.Assert(result.Label == output.ToString());

            input = x_test["1:2"];
            result = task.Predict(input);
            output = np.argmax(y_test[1]);
            Debug.Assert(result.Label == output.ToString());
        }

        // ------------------------------------------------

        public override void PrepareData(IReq req)
        {
            Directory.CreateDirectory("image_classification_cnn_v1");
            var loader = new MnistModelLoader();
            mnist = loader.LoadAsync(".resources/mnist", oneHot: true, showProgressInConsole: true).Result;

            (x_train, y_train) = Reformat(mnist.Train.Data, mnist.Train.Labels);
            (x_valid, y_valid) = Reformat(mnist.Validation.Data, mnist.Validation.Labels);
            (x_test, y_test) = Reformat(mnist.Test.Data, mnist.Test.Labels);

            print("Size of:");
            print($"- Training-set:\t\t{len(mnist.Train.Data)}");
            print($"- Validation-set:\t{len(mnist.Validation.Data)}");

            // ---------------
            // generate labels

            var labels = range(0, 10).Select(x => x.ToString());
            File.WriteAllLines(@"image_classification_cnn_v1\labels.txt", labels);
        }

        // ------------------------------------------------
        /// <summary>
        ///     Reformats the data to the format 
        ///     acceptable for convolutional layers
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <returns></returns>

        private (NDArray, NDArray) Reformat(NDArray x, NDArray y)
        {
            var (unique_y, _) = np.unique(np.argmax(y, 1));
            var (img_size, num_ch, num_class) = ((int)np.sqrt((float)x.shape[1]).astype(np.int32), 1, len(unique_y));
            var dataset = x.reshape((x.shape[0], img_size, img_size, num_ch)).astype(np.float32);

            return (dataset, y);
        }
    }
}
