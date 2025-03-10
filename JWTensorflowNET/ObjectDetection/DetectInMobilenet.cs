#region © 2025 Joe Arrowood (JoeWare)
//
// All rights reserved. Reproduction or transmission in whole or in part, in
// any form or by any means, electronic, mechanical, or otherwise, is prohibited
// without the prior written consent of the copyright owner.
//
#endregion

using System;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.IO;
using System.Linq;

using Tensorflow;
using Tensorflow.Keras.Utils;
using Tensorflow.NumPy;

using JWTensorflowNET.Utility;

using static Tensorflow.Binding;
using JWTensorflowNET.ReqResp;

namespace JWTensorflowNET.ObjectDetection
{
    // ----------------------------------------------------
    /// <summary>
    ///     DetectInMobilenet Description
    /// </summary>

    public class DetectInMobilenet : absSciSharpBase, IJWTensorBase
    {
        public float MIN_SCORE = 0.5f;

        string imageDir = "images";
        string pbFile = "frozen_inference_graph.pb";
        string modelDir = "ssd_mobilenet_v1_coco_2018_01_28";

        // ------------------------------------------------

        public BaseConfig InitConfig() => Config = new BaseConfig
        {
            Enabled = true,
            IsImportingGraph = true,
            Name = "Object Detection in MobileNet (Graph)",
        };

        // ------------------------------------------------

        public bool Run(IReq? req = null)
        {
            tf.compat.v1.disable_eager_execution();

            PrepareData(req);

            Predict();

            return true;
        }

        // ------------------------------------------------

        public override Graph ImportGraph()
        {
            var graph = new Graph().as_default();
            graph.Import(Path.Join(modelDir, pbFile));

            return graph;
        }

        // ------------------------------------------------

        public override void Predict()
        {
            // read in the input image
            var imgArr = ReadTensorFromImageFile(Path.Join(imageDir, "input.jpg"));

            var graph = Config.IsImportingGraph ? ImportGraph() : BuildGraph();

            var sess = tf.Session(graph);
            Tensor tensorNum = graph.OperationByName("num_detections");
            Tensor tensorBoxes = graph.OperationByName("detection_boxes");
            Tensor tensorScores = graph.OperationByName("detection_scores");
            Tensor tensorClasses = graph.OperationByName("detection_classes");
            Tensor imgTensor = graph.OperationByName("image_tensor");
            Tensor[] outTensorArr = new Tensor[] { tensorNum, tensorBoxes, tensorScores, tensorClasses };

            var results = sess.run(outTensorArr, new FeedItem(imgTensor, imgArr));

            buildOutputImage(results);
        }

        // ------------------------------------------------

        public override void PrepareData(IReq req)
        {
            // get model file
            string url = "http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz";
            Web.Download(url, modelDir, "ssd_mobilenet_v1_coco.tar.gz");

            Compress.ExtractTGZ(Path.Join(modelDir, "ssd_mobilenet_v1_coco.tar.gz"), "./");

            // download sample picture
            url = $"https://github.com/tensorflow/models/raw/master/research/object_detection/test_images/image2.jpg";
            Web.Download(url, imageDir, "input.jpg");

            // download the pbtxt file
            url = $"https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/mscoco_label_map.pbtxt";
            Web.Download(url, modelDir, "mscoco_label_map.pbtxt");
        }

        // ------------------------------------------------

        private NDArray ReadTensorFromImageFile(string file_name)
        {
            var graph = tf.Graph().as_default();

            var file_reader = tf.io.read_file(file_name, "file_reader");
            var decodeJpeg = tf.image.decode_jpeg(file_reader, channels: 3, name: "DecodeJpeg");
            var casted = tf.cast(decodeJpeg, TF_DataType.TF_UINT8);
            var dims_expander = tf.expand_dims(casted, 0);

            var sess = tf.Session(graph);
            return sess.run(dims_expander);
        }

        // ------------------------------------------------

        private void buildOutputImage(NDArray[] resultArr)
        {
            // ---------------
            // get pbtxt items

            PbtxtItems pbTxtItems = PbtxtParser.ParsePbtxtFile(Path.Join(modelDir, "mscoco_label_map.pbtxt"));

            // ----------
            // get bitmap

            Bitmap bitmap = new Bitmap(Path.Join(imageDir, "input.jpg"));

            var scores = resultArr[2].ToArray<float>();
            var boxes = resultArr[1].ToArray<float>();
            var id = np.squeeze(resultArr[3]).ToArray<float>();

            for(int i = 0; i < scores.Length; i++)
            {
                var score = scores[i];

                if(score > MIN_SCORE)
                {
                    var top = boxes[i * 4] * bitmap.Height;
                    var left = boxes[i * 4 + 1] * bitmap.Width;
                    var bottom = boxes[i * 4 + 2] * bitmap.Height;
                    var right = boxes[i * 4 + 3] * bitmap.Width;

                    Rectangle rect = new Rectangle()
                    {
                        X = (int)left,
                        Y = (int)top,
                        Width = (int)(right - left),
                        Height = (int)(bottom - top)
                    };

                    var name = pbTxtItems.items.Where(w => w.id == id[i]).Select(s => s.display_name).FirstOrDefault();

                    drawObjectOnBitmap(bitmap, rect, score, name);
                }
            }

            var path = Path.Join(imageDir, "output.jpg");
            bitmap.Save(path);
            Console.WriteLine($"Processed image is saved as {path}");
        }

        // ------------------------------------------------

        private void drawObjectOnBitmap(Bitmap bmp, Rectangle rect, float score, string name)
        {
            using(Graphics graphic = Graphics.FromImage(bmp))
            {
                graphic.SmoothingMode = SmoothingMode.AntiAlias;

                using(Pen pen = new Pen(Color.Red, 2))
                {
                    graphic.DrawRectangle(pen, rect);

                    Point p = new Point(rect.Right + 5, rect.Top + 5);
                    string text = string.Format("{0}:{1}%", name, (int)(score * 100));
                    graphic.DrawString(text, new Font("Verdana", 8), Brushes.Red, p);
                }
            }
        }
    }
}
