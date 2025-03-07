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

using Tensorflow.Keras.Utils;

namespace JWTensorflowNET.ImageProcessing
{
    // ----------------------------------------------------
    /// <summary>
    ///     TransferLearningWithInceptionV3 Description
    /// </summary>

    public class TransferLearningWithInceptionV3 : absSciSharpBase, IJWTensorBase
    {
        float accuracy;

        // ------------------------------------------------

        public BaseConfig InitConfig() => Config = new BaseConfig
        {
            Name = "Transfer Learning With InceptionV3 (Graph)",
            Enabled = true
        };

        // ------------------------------------------------

        public bool Run(IReq? req = null)
        {
            PrepareData();
            Train();
            Test();
            Predict();

            return accuracy > 0.75f;
        }

        // ------------------------------------------------

        public override void PrepareData()
        {
            // ----------------------------
            // get a set of images to teach
            // the network about the new classes

            string fileName = "flower_photos.tgz";
            string dataDir = "image_classification_v1";
            string url = $"http://download.tensorflow.org/example_images/{fileName}";
            Web.Download(url, dataDir, fileName);
            Compress.ExtractTGZ(Path.Join(dataDir, fileName), dataDir);
        }

        // ------------------------------------------------

        public override void Train()
        {
            // ---------------------------
            // using wizard to train model

            var wizard = new ModelWizard();

            var task = wizard.AddImageClassificationTask<TransferLearning>(new TaskOptions
            {
                DataDir = @"image_classification_v1\flower_photos",
            });

            task.Train(new TrainingOptions
            {
                TrainingSteps = 100
            });
        }

        // ------------------------------------------------
        /// <summary>
        /// Prediction
        /// labels mapping, it's from output_lables.txt
        /// 0 - daisy
        /// 1 - dandelion
        /// 2 - roses
        /// 3 - sunflowers
        /// 4 - tulips
        /// </summary>

        public override void Predict()
        {
            // -------------
            // predict image
           
            var wizard = new ModelWizard();
           
            var task = wizard.AddImageClassificationTask<TransferLearning>(new TaskOptions
            {
                ModelPath = @"image_classification_v1\saved_model.pb"
            });

            var imgPath = Path.Join("image_classification_v1", "flower_photos", "daisy", "5547758_eea9edfd54_n.jpg");
            var input = ImageUtil.ReadImageFromFile(imgPath);
            var result = task.Predict(input);
        }

        // ------------------------------------------------

        public override void Test()
        {
            var wizard = new ModelWizard();
            
            var task = wizard.AddImageClassificationTask<TransferLearning>(new TaskOptions
            {
                DataDir = @"image_classification_v1\flower_photos",
                ModelPath = @"image_classification_v1\saved_model.pb"
            });

            var result = task.Test(new TestingOptions { });
            accuracy = result.Accuracy;
        }
    }
}
