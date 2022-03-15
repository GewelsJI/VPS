#ifndef REFERENCE_H_
#define REFERENCE_H_

void sa_weight_forward_Ref(const torch::Tensor& query,const torch::Tensor& key,torch::Tensor& weight,int B,int T,int C,int H,int W,int radius,int dilation);

void sa_weight_backward_query_Ref(const torch::Tensor& dweight,const torch::Tensor& query,
		const torch::Tensor& key,torch::Tensor& dquery,int B,int T,int C,int H,int W,int radius,int dilation);

void sa_weight_backward_key_Ref(const torch::Tensor& dweight,const torch::Tensor& query,
		const torch::Tensor& key,torch::Tensor& dkey,int B,int T,int C,int H,int W,int radius,int dilaiton);

void sa_map_forward_Ref(const torch::Tensor& weight,const torch::Tensor& proj,torch::Tensor& out,int B,int T,int C,int H,int W,int radius,int dilation);

void sa_map_backward_weight_Ref(const torch::Tensor& dout,const torch::Tensor& weight,
		const torch::Tensor& proj,torch::Tensor& dweight,int B,int T,int C,int H,int W,int radius,int dilation);

void sa_map_backward_proj_Ref(const torch::Tensor& dout,const torch::Tensor& weight,
		const torch::Tensor& proj,torch::Tensor& dproj,int B,int T,int C,int H,int W,int radius,int dilation);

#endif /* REFERENCE_H_ */
