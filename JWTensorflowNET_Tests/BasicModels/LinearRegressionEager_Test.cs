#region © 2025 Joe Arrowood (JoeWare)
//
// All rights reserved. Reproduction or transmission in whole or in part, in
// any form or by any means, electronic, mechanical, or otherwise, is prohibited
// without the prior written consent of the copyright owner.
//
#endregion

using JWTensorflowNET.BasicModels;
using JWTensorflowNET.ReqResp;

using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace LinearRegressionEager_Test
{
    // ----------------------------------------------------
    /// <summary>
    ///     Summary description for ArrowUnitTestXML1
    /// </summary>

    [TestClass]
    public class LinearRegressionEager_Test
    {
        public LinearRegressionEager_Test() { }

        // ------------------------------------------------

        [TestMethod]
        [DynamicData(nameof(LinearRegressionEagerTestData), DynamicDataSourceType.Method)]
        public void Run_LinearRegressionEager(bool expected, float bias, float weight,
                                              float[] input, float[] output)
        {
            // -------
            // Arrange

            var req = new Req();
            var sut = new LinearRegressionEager();

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
        public void InitConfig_LinearRegressionEager()
        {
            // -------
            // Arrange

            var sut = new LinearRegressionEager();

            // ---
            // Act

            var resp = sut.InitConfig();

            // ------
            // Assert

            Assert.IsNotNull(resp);
        }

        // ================================================
        // Data provider method
        // ================================================

        public static IEnumerable<object[]> LinearRegressionEagerTestData()
        {
            yield return new object[]
            {
                true,
                -0.73f,
                -0.06f,
                
                // --------
                // Features
                
                new float[] {3.3f, 4.4f, 5.5f, 6.71f, 6.93f, 4.168f, 9.779f, 6.182f, 7.59f, 2.167f, 7.042f, 10.791f, 5.313f, 7.997f, 5.654f, 9.27f, 3.1f},

                // -------
                // Results

                new float[] {1.7f, 2.76f, 2.09f, 3.19f, 1.694f, 1.573f, 3.366f, 2.596f, 2.53f, 1.221f, 2.827f, 3.465f, 1.65f, 2.904f, 2.42f, 2.94f, 1.3f},
            };
        }
    }
}