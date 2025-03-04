/*
BERT base model (uncased)
    Pretrained model on English language using a masked language modeling (MLM) objective. 
    It was introduced in this paper and first released in this repository. 
    This model is uncased: it does not make a difference between english and English.

    BERT is a transformers model pretrained on a large corpus of English data in a self-supervised fashion. 
    This means it was pretrained on the raw texts only, with no humans labeling them in any way  
    (which is why it can use lots of publicly available data) with an automatic process to generate inputs and labels from those texts.
*/

using System.IO;
using System.Net;
using Tensorflow;
using System.Collections.Generic;

namespace BERT
{
    internal interface IMDBDataPreProcessor
    {
        public static (int[,], int[]) ProcessData(string path, int max_len, int label = 0)
        {
            string url = "https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt";
            
            string vocab_file = "./vocab.txt";
            { 
                using (WebClient client = new WebClient())
                {
                    byte[] fileData = client.DownloadData(url); 

                    using (Stream stream = File.Create(vocab_file))
                    {
                        stream.Write(fileData, 0, fileData.Length);
                    }
                }
            }

            BertTokenizer tokenizer = new BertTokenizer(vocab_file);
            DirectoryInfo root = new DirectoryInfo(path);
            FileInfo[] files = root.GetFiles();
            List<List<int>> res = new List<List<int>>();
            List<int> res_y = new List<int>();
           
            foreach (var item in files)
            {
                
                string str = File.ReadAllText(item.ToString());
                str = str.Replace("<br /><br />", " ");
                var ids = tokenizer.convert_tokens_to_ids(tokenizer._tokenize(str));
                ids = tokenizer.build_inputs_with_special_tokens(ids);
                
                if (ids.Count < max_len)
                {
                    var tmp = new List<int>(new int[max_len - ids.Count]);
                    for (int i = 0; i < tmp.Count; i++) tmp[i] = 0;
                    ids.extend(tmp);
                }
                else 
                { 
                    ids = ids.GetRange(0, max_len); ids[ids.Count - 1] = 102; 
                }

                res_y.Add(label);
                res.Add(ids);
            }

            var res_array = new int[res.Count, max_len];

            for(int i = 0; i < res.Count; i++)
            {
                for(int j = 0; j < max_len; j++) res_array[i, j] = res[i][j];
            }

            return (res_array, res_y.ToArray());
        }
    }
}
