#region © 2025 Aflac.
//
// All rights reserved. Reproduction or transmission in whole or in part, in
// any form or by any means, electronic, mechanical, or otherwise, is prohibited
// without the prior written consent of the copyright owner.
//
#endregion

using System.Collections.Generic;
using System.Linq;

namespace TensorFlowNET.Examples.ReqResp
{
    public class Run_Req : IReq
    {
        private List<(string, int)> _properties = new List<(string, int)>();

        // ------------------------------------------------

        public T GetValue<T>(string key)
        {
            var value = _properties.FirstOrDefault(p => p.Item1 == key).Item2;
            return (T)(object)value;
        }

        // ------------------------------------------------

        public void SetValue<T>(string key, T val)
        {
            var index = _properties.FindIndex(p => p.Item1 == key);

            if(index != -1)
            {
                _properties[index] = (key, (int)(object)val);
            }
            else
            {
                _properties.Add((key, (int)(object)val));   
            }
        }
    }
}