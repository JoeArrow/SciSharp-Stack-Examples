
namespace TensorFlowNET.Examples.ReqResp
{
    public interface IReq
    {
        T GetValue<T>(string name);
        void SetValue<T>(string name, T val);
    }
}
