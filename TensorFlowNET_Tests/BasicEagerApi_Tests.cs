#region © 2025 Aflac.
//
// All rights reserved. Reproduction or transmission in whole or in part, in
// any form or by any means, electronic, mechanical, or otherwise, is prohibited
// without the prior written consent of the copyright owner.
//
#endregion

using TensorFlowNET.Examples.ReqResp;
using TensorFlowNET.Examples.Testable;

namespace BasicEagerApi_Tests
{
    // ----------------------------------------------------
    /// <summary>
    ///     Summary description for ArrowUnitTestXML1
    /// </summary>

    [TestClass]
    public class BasicEagerApi_Tests
    {
        public BasicEagerApi_Tests() { }

        // ------------------------------------------------

        [TestMethod]
        [DataRow(2, 5)]
        [DataRow(12, 15)]
        [DataRow(102, 105)]
        public void Run_BasicEagerApi_T(int a, int b)
        {
            // -------
            // Arrange

            var req = new Run_Req();
            req.SetValue("a", a);
            req.SetValue("b", b);

            var sut = new BasicEagerApi_T();
            sut.InitConfig();

            // ---
            // Act

            var resp = sut.Run(req);

            // ------
            // Assert

            Assert.IsTrue(resp);
        }
    }
}