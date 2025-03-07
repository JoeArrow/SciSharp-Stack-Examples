#region © 2025 Joe Arrowood (JoeWare)
//
// All rights reserved. Reproduction or transmission in whole or in part, in
// any form or by any means, electronic, mechanical, or otherwise, is prohibited
// without the prior written consent of the copyright owner.
//
#endregion

using Newtonsoft.Json;

namespace JWTensorflowNET.Utility
{
    // ------------------------------------------------

    public class PbtxtParser
    {
        public static PbtxtItems ParsePbtxtFile(string filePath)
        {
            string line;
            string newText = "{\"items\":[";

            using(System.IO.StreamReader reader = new System.IO.StreamReader(filePath))
            {

                while((line = reader.ReadLine()) != null)
                {
                    string newline = string.Empty;

                    if(line.Contains("{"))
                    {
                        newline = line.Replace("item", "").Trim();
                        //newText += line.Insert(line.IndexOf("=") + 1, "\"") + "\",";
                        newText += newline;
                    }
                    else if(line.Contains("}"))
                    {
                        newText = newText.Remove(newText.Length - 1);
                        newText += line;
                        newText += ",";
                    }
                    else
                    {
                        newline = line.Replace(":", "\":").Trim();
                        newline = "\"" + newline;// newline.Insert(0, "\"");
                        newline += ",";

                        newText += newline;
                    }
                }

                newText = newText.Remove(newText.Length - 1);
                newText += "]}";

                reader.Close();
            }

            PbtxtItems items = JsonConvert.DeserializeObject<PbtxtItems>(newText);

            return items;
        }
    }
}
