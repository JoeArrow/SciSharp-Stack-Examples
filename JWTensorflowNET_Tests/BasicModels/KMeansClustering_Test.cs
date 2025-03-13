#region © 2025 Joe Arrowood (JoeWare)
//
// All rights reserved. Reproduction or transmission in whole or in part, in
// any form or by any means, electronic, mechanical, or otherwise, is prohibited
// without the prior written consent of the copyright owner.
//
#endregion

using JWTensorflowNET.ReqResp;
using JWTensorflowNET.BasicModels;

namespace KMeansClustering_Tests
{
    // ----------------------------------------------------
    /// <summary>
    ///     Summary description for ArrowUnitTestXML1
    /// </summary>

    [TestClass]
    public class KMeansClustering_Test
    {
        public KMeansClustering_Test() { }

        // ------------------------------------------------

        [TestMethod]
        [DataRow(0.70f)]
        [DataRow(-0.1f)]
        [DoNotParallelize]
        public void Run_KMeansClustering(float threshold)
        {
            // -------
            // Arrange

            var req = new Req();

            req.SetValue<float>("Threshold", threshold);

            if(threshold < 0) req = null;

            var sut = new KMeansClustering();

            // ---
            // Act

            var resp = sut.Run(req);

            // ------
            // Assert

            Assert.IsTrue(resp);
        }

        // ------------------------------------------------

        //[TestMethod]
        public void Run_KMeansClustering_NoReq()
        {
            // -------
            // Arrange

            var sut = new KMeansClustering();

            // ---
            // Act

            var resp = sut.Run();

            // ------
            // Assert

            Assert.IsTrue(resp);
        }

        // ------------------------------------------------

        [TestMethod]
        public void InitConfig_KMeansClustering()
        {
            // -------
            // Arrange

            var sut = new KMeansClustering();

            // ---
            // Act

            var resp = sut.InitConfig();

            // ------
            // Assert

            Assert.IsNotNull(resp);
        }
    }
}