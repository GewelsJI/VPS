//#include<torch/torch.h>
#include<torch/extension.h>
#include"utils.h"
#include"timer.h"
#include"reference.h"

void get_sizes(const torch::Tensor& t,int *B,int *T,int *C,int *H,int *W){
	*B=t.size(0);
	*T=t.size(1);
	*C=t.size(2);
	*H=t.size(3);
	*W=t.size(4);
}

void _sa_weight_forward_cuda(const torch::Tensor& query,const torch::Tensor& key,torch::Tensor& weight,int B,int T,int C,int H,int W,int radius,int dilation);
void _sa_map_forward_cuda(const torch::Tensor& weight,const torch::Tensor& proj,torch::Tensor& out,int B,int T,int C,int H,int W,int radius,int dilation);
void _sa_weight_backward_cuda(const torch::Tensor& dw,const torch::Tensor& query,
		const torch::Tensor& key,torch::Tensor& dquery,torch::Tensor& dkey,
		int B,int T,int C,int H,int W,int radius,int dilation);
void _sa_map_backward_cuda(const torch::Tensor& dout,const torch::Tensor& weight,
		const torch::Tensor& proj,torch::Tensor& dweight,torch::Tensor& dproj,
		int B,int T,int C,int H,int W,int radius,int dilaiton);


//forward declarations-------python pass information here
void sa_weight_forward(const torch::Tensor& query,const torch::Tensor& key,torch::Tensor& weight,int radius,int dilation){
	int B,T,C,H,W;
	get_sizes(query,&B,&T,&C,&H,&W);
	//GpuTimer timer;
	//timer.Start();
	_sa_weight_forward_cuda(query,key,weight,B,T,C,H,W,radius,dilation);
	//timer.Stop();
	//cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());
	//printf("Your code ran in: %f msecs.\n", timer.Elapsed());
}

void sa_map_forward(const torch::Tensor& weight,const torch::Tensor& proj,torch::Tensor& out,int radius,int dilation){
	int B,T,C,H,W;
	get_sizes(proj,&B,&T,&C,&H,&W);
	//GpuTimer timer;
	//timer.Start();
	_sa_map_forward_cuda(weight,proj,out,B,T,C,H,W,radius,dilation);
	//timer.Stop();
	//cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());
	//printf("Your code ran in: %f msecs.\n", timer.Elapsed());
}

void sa_weight_backward(const torch::Tensor& dw,const torch::Tensor& query,const torch::Tensor& key,torch::Tensor& dquery,torch::Tensor& dkey,int radius,int dilation){
	int B,T,C,H,W;
	get_sizes(query,&B,&T,&C,&H,&W);
	//GpuTimer timer;
	//timer.Start();
	_sa_weight_backward_cuda(dw,query,key,dquery,dkey,B,T,C,H,W,radius,dilation);
	//timer.Stop();
	//cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());
	//printf("Your code ran in: %f msecs.\n", timer.Elapsed());
}

void sa_map_backward(const torch::Tensor& dout,const torch::Tensor& weight,const torch::Tensor& proj,torch::Tensor& dweight,torch::Tensor& dproj,int radius,int dilation){
	int B,T,C,H,W;
	get_sizes(proj,&B,&T,&C,&H,&W);
	//GpuTimer timer;
	//timer.Start();
	_sa_map_backward_cuda(dout,weight,proj,dweight,dproj,B,T,C,H,W,radius,dilation);
	//timer.Stop();
	//cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());
	//printf("Your code ran in: %f msecs.\n", timer.Elapsed());
}

void sa_weight_forward_ref(const torch::Tensor& query,const torch::Tensor& key,torch::Tensor& weight,int radius,int dilation){
	int B,T,C,H,W;
	get_sizes(query,&B,&T,&C,&H,&W);
	sa_weight_forward_Ref(query,key,weight,B,T,C,H,W,radius,dilation);
}

void sa_weight_backward_ref(const torch::Tensor& dw,const torch::Tensor& query,const torch::Tensor& key,torch::Tensor& dquery,torch::Tensor& dkey,int radius,int dilation){
	int B,T,C,H,W;
	get_sizes(query,&B,&T,&C,&H,&W);
	sa_weight_backward_query_Ref(dw,query,key,dquery,B,T,C,H,W,radius,dilation);
	sa_weight_backward_key_Ref(dw,query,key,dkey,B,T,C,H,W,radius,dilation);
}

void sa_map_forward_ref(const torch::Tensor& weight,const torch::Tensor& proj,torch::Tensor& out,int radius,int dilation){
	int B,T,C,H,W;
	get_sizes(proj,&B,&T,&C,&H,&W);
	sa_map_forward_Ref(weight,proj,out,B,T,C,H,W,radius,dilation);
}

void sa_map_backward_ref(const torch::Tensor& dout,const torch::Tensor& weight,const torch::Tensor& proj,torch::Tensor& dweight,torch::Tensor& dproj,int radius,int dilation){
	int B,T,C,H,W;
	get_sizes(proj,&B,&T,&C,&H,&W);
	sa_map_backward_weight_Ref(dout,weight,proj,dweight,B,T,C,H,W,radius,dilation);
	sa_map_backward_proj_Ref(dout,weight,proj,dproj,B,T,C,H,W,radius,dilation);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("weight_forward", &sa_weight_forward, "weight forward (CUDA)");
	m.def("weight_backward", &sa_weight_backward, "weight backward (CUDA)");
	m.def("map_forward", &sa_map_forward, "map forward (CUDA)");
	m.def("map_backward", &sa_map_backward, "map backward (CUDA)");
	m.def("weight_forward_ref", &sa_weight_forward_ref, "weight forward ref (CUDA)");
	m.def("weight_backward_ref", &sa_weight_backward_ref, "weight backward ref (CUDA)");
	m.def("map_forward_ref", &sa_map_forward_ref, "map forward ref (CUDA)");
	m.def("map_backward_ref", &sa_map_backward_ref, "map backward ref (CUDA)");
}
/*
int main() {
	//torch::Tensor weight=torch::ones({2,5,5*9,28,42}, at::kFloat).to(torch::Device(torch::kCUDA, 0));
	//torch::Tensor query=torch::ones({2,5,8,28,42}, at::kFloat).to(torch::Device(torch::kCUDA, 0));
	//torch::Tensor key=torch::ones({2,5,8,28,42}, at::kFloat).to(torch::Device(torch::kCUDA, 0));
	//sa_weight_forward(query,key,weight);
	/*
	torch::Tensor weight=torch::ones({1,5,5*9,28,42}, at::kFloat).to(torch::Device(torch::kCUDA, 0));
	torch::Tensor proj=torch::ones({1,5,8,28,42}, at::kFloat).to(torch::Device(torch::kCUDA, 0));
	torch::Tensor out=torch::zeros({1,5,8,28,42}, at::kFloat).to(torch::Device(torch::kCUDA, 0));
	sa_map_forward(weight,proj,out);

	/*
	torch::Tensor dw=torch::ones({1,5,5*9,28,42}, at::kFloat).to(torch::Device(torch::kCUDA, 0));
	torch::Tensor query=torch::ones({1,5,8,28,42}, at::kFloat).to(torch::Device(torch::kCUDA, 0));
	torch::Tensor key=torch::ones({1,5,8,28,42}, at::kFloat).to(torch::Device(torch::kCUDA, 0));
	torch::Tensor dquery=torch::zeros({1,5,8,28,42}, at::kFloat).to(torch::Device(torch::kCUDA, 0));
	torch::Tensor dkey=torch::zeros({1,5,8,28,42}, at::kFloat).to(torch::Device(torch::kCUDA, 0));
	sa_weight_backward(dw,query,key,dquery,dkey);

	torch::Tensor dout=torch::ones({1,5,8,28,42}, at::kFloat).to(torch::Device(torch::kCUDA, 0));

	torch::Tensor weight=torch::ones({1,5,5*9,28,42}, at::kFloat).to(torch::Device(torch::kCUDA, 0));
	torch::Tensor proj=torch::ones({1,5,8,28,42}, at::kFloat).to(torch::Device(torch::kCUDA, 0));
	torch::Tensor dweight=torch::zeros({1,5,5*9,28,42}, at::kFloat).to(torch::Device(torch::kCUDA, 0));
	torch::Tensor dproj=torch::zeros({1,5,8,28,42}, at::kFloat).to(torch::Device(torch::kCUDA, 0));
	sa_map_backward(dout,weight,proj,dweight,dproj);
	std::cout<<dweight[0][0][0][0][0];
	return 0;
}*/

