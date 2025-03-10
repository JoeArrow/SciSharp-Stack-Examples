#region © 2025 Joe Arrowood (JoeWare)
//
// All rights reserved. Reproduction or transmission in whole or in part, in
// any form or by any means, electronic, mechanical, or otherwise, is prohibited
// without the prior written consent of the copyright owner.
//
#endregion

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

            // ---
            // Act

            var actual = sut.Run(null);

            // ------
            // Assert

            Assert.AreEqual(expected, actual, "Linear Regression model did not return expected results.");
        }
    }
}