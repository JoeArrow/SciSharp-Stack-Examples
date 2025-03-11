#region © 2025 Joe Arrowood (JoeWare)
//
// All rights reserved. Reproduction or transmission in whole or in part, in
// any form or by any means, electronic, mechanical, or otherwise, is prohibited
// without the prior written consent of the copyright owner.
//
#endregion

using JWTensorflowNET.ReqResp;

namespace Req_Test
{
    // ----------------------------------------------------
    /// <summary>
    ///     Summary description for ArrowUnitTestXML1
    /// </summary>

    [TestClass]
    public class Req_Test
    {
        public Req_Test() { }

        // ------------------------------------------------

        [TestMethod]
        [DataRow("Item1", "Value1")]
        [DataRow("Item2", "Value2")]
        public void SetGetValues_Req(string name, string value)
        {
            // -------
            // Arrange

            var sut = new Req();

            // ---
            // Act

            sut.SetValue<string>(name, value);
            sut.SetValue<string>(name, value);
            var val = sut.GetValue<string>(name);

            // ------
            // Assert

            Assert.AreEqual(value, val);
        }
    }
}