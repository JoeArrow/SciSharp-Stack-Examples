#region © 2025 Joe Arrowood (JoeWare)
//
// All rights reserved. Reproduction or transmission in whole or in part, in
// any form or by any means, electronic, mechanical, or otherwise, is prohibited
// without the prior written consent of the copyright owner.
//
#endregion

using JWTensorflowNET.ReqResp;
using JWTensorflowNET.BasicModels;

using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace LinearRegression_Test
{
    // ----------------------------------------------------
    /// <summary>
    ///     Summary description for ArrowUnitTestXML1
    /// </summary>

    [TestClass]
    public class LinearRegression_Test
    {
        public LinearRegression_Test() { }

        // ------------------------------------------------

        [TestMethod]
        [DataRow(true)]
        public void Run_LinearRegression(bool expected)
        {
            // -------
            // Arrange

            var sut = new LinearRegression();

            var input = new float[] {3.3f,   4.4f,   5.5f,    6.71f,
                                     6.93f,  4.168f, 9.779f,  6.182f,
                                     7.59f,  2.167f, 7.042f, 10.791f,
                                     5.313f, 7.997f, 5.654f,  9.27f,
                                     3.1f};

            var output = new float[] {1.7f,   2.76f,  2.09f,  3.19f,
                                      1.694f, 1.573f, 3.366f, 2.596f,
                                      2.53f,  1.221f, 2.827f, 3.465f,
                                      1.65f,  2.904f, 2.42f,  2.94f,
                                      1.3f};
            var req = new Req();

            req.SetValue<float>("Bias", -0.73f);
            req.SetValue<float>("Weight", -0.06f);
            req.SetValue<float[]>("Input", input);
            req.SetValue<float[]>("Output", output);

            // ---
            // Act

            var actual = sut.Run(req);

            // ------
            // Assert

            Assert.AreEqual(expected, actual, "Linear Regression model did not return expected results.");
        }
    }
}