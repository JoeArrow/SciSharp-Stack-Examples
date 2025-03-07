#region © 2025 Joe Arrowood (JoeWare)
//
// All rights reserved. Reproduction or transmission in whole or in part, in
// any form or by any means, electronic, mechanical, or otherwise, is prohibited
// without the prior written consent of the copyright owner.
//
#endregion

using JWTensorflowNET.ReqResp;

using SciSharp.Models;
using SciSharp.Models.ObjectDetection;

namespace JWTensorflowNET.ObjectDetection
{
    // ----------------------------------------------------
    /// <summary>
    ///     MnistInYOLOv3 Description
    /// </summary>

    public class MnistInYOLOv3 : absSciSharpBase, IJWTensorBase
    {
        YoloConfig cfg;
        float accuracy_test = 0f;
        YoloDataset trainingData, testingData;

        // ------------------------------------------------

        public BaseConfig InitConfig() => Config = new BaseConfig
        {
            Name = "MNIST in YOLOv3",
            Enabled = true
        };

        // ------------------------------------------------

        public bool Run(IReq? req = null)
        {
            cfg = new YoloConfig("YOLOv3");
            (trainingData, testingData) = PrepareData();
            // Train();
            Test();
            return true;
        }

        // ------------------------------------------------

        public override void Train()
        {
            // ---------------------------
            // using wizard to train model

            var wizard = new ModelWizard();

            var task = wizard.AddObjectDetectionTask<YOLOv3>(new TaskOptions
            {
                InputShape = (28, 28, 1),
                NumberOfClass = 10,
            });

            task.SetModelArgs(cfg);

            task.Train(new YoloTrainingOptions
            {
                TrainingData = trainingData,
                TestingData = testingData
            });
        }

        // ------------------------------------------------

        public override void Test()
        {
            var wizard = new ModelWizard();

            var task = wizard.AddObjectDetectionTask<YOLOv3>(new TaskOptions
            {
                ModelPath = @"./YOLOv3/yolov3.h5"
            });

            task.SetModelArgs(cfg);

            var result = task.Test(new TestingOptions
            {

            });

            accuracy_test = result.Accuracy;
        }

        // ------------------------------------------------

        public (YoloDataset, YoloDataset) PrepareData()
        {
            string dataDir = Path.Combine("YOLOv3", "data");
            Directory.CreateDirectory(dataDir);

            var trainset = new YoloDataset("train", cfg);
            var testset = new YoloDataset("test", cfg);
            return (trainset, testset);
        }
    }
}
