#region © 2025 Joe Arrowood (JoeWare)
//
// All rights reserved. Reproduction or transmission in whole or in part, in
// any form or by any means, electronic, mechanical, or otherwise, is prohibited
// without the prior written consent of the copyright owner.
//
#endregion

using System.Text.Json;

using JWTensorflowNET.ReqResp;
using JWTensorflowNET.BasicModels;

namespace LinearRegressionKeras_Test
{
    // ----------------------------------------------------
    /// <summary>
    ///     Summary description for ArrowUnitTestXML1
    /// </summary>

    [TestClass]
    public class LinearRegressionKeras_Test
    {
        public LinearRegressionKeras_Test() { }

        // ------------------------------------------------

        [TestMethod]
        [DataRow(true, -0.73f, -0.06f,
                 "[3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167, 7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1]",
                 "[1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221, 2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3]")]
        public void Run_LinearRegressionKeras(bool expected, float bias, float weight, string inputJson, string outputJson)
        {
            // -------
            // Arrange

            var input = JsonSerializer.Deserialize<float[]>(inputJson) ?? new float[] { 0.0f };
            var output = JsonSerializer.Deserialize<float[]>(outputJson) ?? new float[] { 0.0f };

            var req = new Req();

            var sut = new LinearRegressionKeras();

            req.SetValue<float>("Bias", bias);
            req.SetValue<float>("Weight", weight);
            req.SetValue<float[]>("Input", input);
            req.SetValue<float[]>("Output", output);

            // ---
            // Act

            var resp = sut.Run(req);

            // ------
            // Assert

            Assert.AreEqual(expected, resp, "Linear Regression model did not return expected results.");
        }

        // ------------------------------------------------

        [TestMethod]
        public void InitConfig_LinearRegressionKeras()
        {
            // -------
            // Arrange

            var sut = new LinearRegressionKeras();

            // ---
            // Act

            var resp = sut.InitConfig();

            // ------
            // Assert

            Assert.IsNotNull(resp);
        }
    }
}