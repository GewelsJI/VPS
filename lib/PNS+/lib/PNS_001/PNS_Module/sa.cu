//#include <torch/torch.h>
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>

#define TensorAccessor5D torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits,int32_t>
/*
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
  static __inline__ __device__ double atomicAdd(double *address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    if (val==0.0)
      return __longlong_as_double(old);
    do {
      assumed = old;
      old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val +__longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
  }
#endif
*/
template <typename scalar_t>
__global__
void sa_weight_forward_kernel(
	const TensorAccessor5D query,
	const TensorAccessor5D key,
	TensorAccessor5D weight,int B,int T,int C,int H,int W,int radius,int dilation){
	int w = blockIdx.x * blockDim.x + threadIdx.x;//col
	int h = blockIdx.y * blockDim.y + threadIdx.y;//row
	int time = blockIdx.z;//time
	int diameter=2*radius+1;

	//query B*T*C*H*W
	//key B*T*C*H*W
	//weight B*T*9T*H*W
	if(w<W&&h<H&&time<T){
		for(int batch=0;batch<B;++batch){
			for(int cal_time=0;cal_time<T;++cal_time){
				for(int dh=-radius*dilation;dh<=radius*dilation;dh+=dilation){
					for(int dw=-radius*dilation;dw<=radius*dilation;dw+=dilation){
						scalar_t sum=0.0;
						if(h+dh<H&&h+dh>=0&&w+dw<W&&w+dw>=0){
							for(int c=0;c<C;++c){
								scalar_t q=query[batch][time][c][h][w];
								scalar_t k=key[batch][cal_time][c][h+dh][w+dw];
								sum+=q*k;
							}
						}
						weight[batch][time][cal_time*diameter*diameter+(dh/dilation+radius)*(2*radius+1)+(dw/dilation+radius)][h][w]=sum;
					}
				}
			}
		}
	}
}

template <typename scalar_t>
__global__
void sa_map_forward_kernel(
	const TensorAccessor5D weight,
	const TensorAccessor5D proj,
	TensorAccessor5D out,int B,int T,int C,int H,int W,int radius,int dilation){
	int w = blockIdx.x * blockDim.x + threadIdx.x;//col
	int h = blockIdx.y * blockDim.y + threadIdx.y;//row
	int time = blockIdx.z;//time
	int diameter=2*radius+1;

	//weight B*T*9T*H*W
	//proj B*T*C*H*W
	//out B*T*C*H*W
	if(w<W&&h<H&&time<T){
		for(int batch=0;batch<B;++batch){
			for(int c=0;c<C;++c){
				scalar_t sum=0.0;
				for(int cal_time=0;cal_time<T;++cal_time){
					for(int dh=-radius*dilation;dh<=radius*dilation;dh+=dilation){
						for(int dw=-radius*dilation;dw<=radius*dilation;dw+=dilation){
							if(h+dh<H&&h+dh>=0&&w+dw<W&&w+dw>=0){
								scalar_t weight_temp=weight[batch][time][cal_time*diameter*diameter+(dh/dilation+radius)*(2*radius+1)+(dw/dilation+radius)][h][w];
								scalar_t proj_value=proj[batch][cal_time][c][h+dh][w+dw];
								sum+=weight_temp*proj_value;
							}
						}
					}
				}
				out[batch][time][c][h][w]=sum;
			}
		}
	}
}

template <typename scalar_t>
__global__
void sa_weight_backward_kernel_query(
	const TensorAccessor5D dweight,
	const TensorAccessor5D key,
	TensorAccessor5D dquery,int B,int T,int C,int H,int W,int radius,int dilation){
	int w = blockIdx.x * blockDim.x + threadIdx.x;//col
	int h = blockIdx.y * blockDim.y + threadIdx.y;//row
	int time = blockIdx.z;//time
	int diameter=2*radius+1;

	//weight B*T*9T*H*W
	//proj B*T*C*H*W
	//out B*T*C*H*W
	if(w<W&&h<H&&time<T){
		for(int batch=0;batch<B;++batch){
			for(int c=0;c<C;++c){
				scalar_t sum=0.0;
				for(int cal_time=0;cal_time<T;++cal_time){
					for(int dh=-radius*dilation;dh<=radius*dilation;dh+=dilation){
						for(int dw=-radius*dilation;dw<=radius*dilation;dw+=dilation){
							if(h+dh<H&&h+dh>=0&&w+dw<W&&w+dw>=0){
								scalar_t _dweight=dweight[batch][time][cal_time*diameter*diameter+(dh/dilation+radius)*(2*radius+1)+(dw/dilation+radius)][h][w];
								scalar_t _key=key[batch][cal_time][c][h+dh][w+dw];
								sum+=_dweight*_key;
							}
						}
					}
				}
				dquery[batch][time][c][h][w]=sum;
			}
		}
	}
}

template <typename scalar_t>
__global__
void sa_weight_backward_kernel_key(
	const TensorAccessor5D dweight,
	const TensorAccessor5D query,
	TensorAccessor5D dkey,int B,int T,int C,int H,int W,int radius,int dilation){
	int w = blockIdx.x * blockDim.x + threadIdx.x;//col
	int h = blockIdx.y * blockDim.y + threadIdx.y;//row
	int time = blockIdx.z;//time
	int diameter=2*radius+1;

	//weight B*T*9T*H*W
	//proj B*T*C*H*W
	//out B*T*C*H*W
	if(w<W&&h<H&&time<T){
		for(int batch=0;batch<B;++batch){
			for(int c=0;c<C;++c){
				for(int cal_time=0;cal_time<T;++cal_time){
					for(int dh=-radius*dilation;dh<=radius*dilation;dh+=dilation){
						for(int dw=-radius*dilation;dw<=radius*dilation;dw+=dilation){
							if(h+dh<H&&h+dh>=0&&w+dw<W&&w+dw>=0){
								scalar_t _dweight=dweight[batch][time][cal_time*diameter*diameter+(dh/dilation+radius)*(2*radius+1)+(dw/dilation+radius)][h][w];
								scalar_t _query=query[batch][time][c][h][w];
								atomicAdd(&dkey[batch][cal_time][c][h+dh][w+dw],_dweight*_query);
							}
						}
					}
				}
			}
		}
	}
}

template <typename scalar_t>
__global__
void sa_map_backward_kernel_weight(
	const TensorAccessor5D dout,
	const TensorAccessor5D proj,
	TensorAccessor5D dweight,int B,int T,int C,int H,int W,int radius,int dilation){
	int w = blockIdx.x * blockDim.x + threadIdx.x;//col
	int h = blockIdx.y * blockDim.y + threadIdx.y;//row
	int time = blockIdx.z;//time
	int diameter=2*radius+1;

	//weight B*T*9T*H*W
	//proj B*T*C*H*W
	//out B*T*C*H*W
	if(w<W&&h<H&&time<T){
		for(int batch=0;batch<B;++batch){
			for(int cal_time=0;cal_time<T;++cal_time){
				for(int dh=-radius*dilation;dh<=radius*dilation;dh+=dilation){
					for(int dw=-radius*dilation;dw<=radius*dilation;dw+=dilation){
						scalar_t sum=0.0;
						for(int c=0;c<C;++c){
							if(h+dh<H&&h+dh>=0&&w+dw<W&&w+dw>=0){
								scalar_t _proj=proj[batch][cal_time][c][h+dh][w+dw];
								scalar_t _dout=dout[batch][time][c][h][w];
								sum+=_dout*_proj;
							}
						}
						dweight[batch][time][cal_time*diameter*diameter+(dh/dilation+radius)*(2*radius+1)+(dw/dilation+radius)][h][w]=sum;
					}
				}
			}
		}
	}
}

template <typename scalar_t>
__global__
void sa_map_backward_kernel_proj(
	const TensorAccessor5D dout,
	const TensorAccessor5D weight,
	TensorAccessor5D dproj,int B,int T,int C,int H,int W,int radius,int dilation){
	int w = blockIdx.x * blockDim.x + threadIdx.x;//col
	int h = blockIdx.y * blockDim.y + threadIdx.y;//row
	int time = blockIdx.z;//time
	int diameter=2*radius+1;
	//weight B*T*9T*H*W
	//proj B*T*C*H*W
	//out B*T*C*H*W
	if(w<W&&h<H&&time<T){
		for(int batch=0;batch<B;++batch){
			for(int c=0;c<C;++c){
				for(int cal_time=0;cal_time<T;++cal_time){
					for(int dh=-radius*dilation;dh<=radius*dilation;dh+=dilation){
						for(int dw=-radius*dilation;dw<=radius*dilation;dw+=dilation){
							if(h+dh<H&&h+dh>=0&&w+dw<W&&w+dw>=0){
								scalar_t weight_temp=weight[batch][time][cal_time*diameter*diameter+(dh/dilation+radius)*(2*radius+1)+(dw/dilation+radius)][h][w];
								scalar_t _dout=dout[batch][time][c][h][w];
								atomicAdd(&dproj[batch][cal_time][c][h+dh][w+dw],_dout*weight_temp);
							}
						}
					}
				}
			}
		}
	}
}

void _sa_weight_forward_cuda(const torch::Tensor& query,const torch::Tensor& key,torch::Tensor& weight,int B,int T,int C,int H,int W,int radius,int dilation){
	dim3 threads(16,16);
	dim3 blocks((W+threads.x-1)/threads.x,(H+threads.y-1)/threads.y,T);

	AT_DISPATCH_FLOATING_TYPES(weight.scalar_type(), "sa_weight_forward_cuda", ([&] {
		sa_weight_forward_kernel<scalar_t><<<blocks, threads>>>(
			query.packed_accessor<scalar_t,5,torch::RestrictPtrTraits,int32_t>(),
			key.packed_accessor<scalar_t,5,torch::RestrictPtrTraits,int32_t>(),
			weight.packed_accessor<scalar_t,5,torch::RestrictPtrTraits,int32_t>(),B,T,C,H,W,radius,dilation);
	  }));
}

void _sa_map_forward_cuda(const torch::Tensor& weight,const torch::Tensor& proj,torch::Tensor& out,int B,int T,int C,int H,int W,int radius,int dilation){
	dim3 threads(16,16);
	dim3 blocks((W+threads.x-1)/threads.x,(H+threads.y-1)/threads.y,T);
	AT_DISPATCH_FLOATING_TYPES(weight.scalar_type(), "sa_map_forward_cuda", ([&] {
		sa_map_forward_kernel<scalar_t><<<blocks, threads>>>(
		weight.packed_accessor<scalar_t,5,torch::RestrictPtrTraits,int32_t>(),
		proj.packed_accessor<scalar_t,5,torch::RestrictPtrTraits,int32_t>(),
		out.packed_accessor<scalar_t,5,torch::RestrictPtrTraits,int32_t>(),B,T,C,H,W,radius,dilation);
	}));
}

void _sa_weight_backward_cuda(const torch::Tensor& dw,const torch::Tensor& query,
		const torch::Tensor& key,torch::Tensor& dquery,torch::Tensor& dkey,
		int B,int T,int C,int H,int W,int radius,int dilation){
	dim3 threads(16,16);
	dim3 blocks((W+threads.x-1)/threads.x,(H+threads.y-1)/threads.y,T);
	AT_DISPATCH_FLOATING_TYPES(dw.scalar_type(), "sa_weight_backward_cuda", ([&] {
		const TensorAccessor5D dw_acc=dw.packed_accessor<scalar_t,5,torch::RestrictPtrTraits,int32_t>();
		const TensorAccessor5D query_acc=query.packed_accessor<scalar_t,5,torch::RestrictPtrTraits,int32_t>();
		const TensorAccessor5D key_acc=key.packed_accessor<scalar_t,5,torch::RestrictPtrTraits,int32_t>();
		TensorAccessor5D dquery_acc=dquery.packed_accessor<scalar_t,5,torch::RestrictPtrTraits,int32_t>();
		TensorAccessor5D dkey_acc=dkey.packed_accessor<scalar_t,5,torch::RestrictPtrTraits,int32_t>();
		sa_weight_backward_kernel_query<scalar_t><<<blocks, threads>>>(dw_acc,key_acc,dquery_acc,B,T,C,H,W,radius,dilation);
		sa_weight_backward_kernel_key<scalar_t><<<blocks, threads>>>(dw_acc,query_acc,dkey_acc,B,T,C,H,W,radius,dilation);
	}));
}

void _sa_map_backward_cuda(const torch::Tensor& dout, const torch::Tensor& weight,
		const torch::Tensor& proj,torch::Tensor& dweight,torch::Tensor& dproj,
		int B,int T,int C,int H,int W,int radius,int dilation){
	dim3 threads(16,16);
	dim3 blocks((W+threads.x-1)/threads.x,(H+threads.y-1)/threads.y,T);

	AT_DISPATCH_FLOATING_TYPES(dout.scalar_type(), "sa_map_backward_cuda", ([&] {
		const TensorAccessor5D dout_acc=dout.packed_accessor<scalar_t,5,torch::RestrictPtrTraits,int32_t>();
		const TensorAccessor5D weight_acc=weight.packed_accessor<scalar_t,5,torch::RestrictPtrTraits,int32_t>();
		const TensorAccessor5D proj_acc=proj.packed_accessor<scalar_t,5,torch::RestrictPtrTraits,int32_t>();
		TensorAccessor5D dweight_acc=dweight.packed_accessor<scalar_t,5,torch::RestrictPtrTraits,int32_t>();
		TensorAccessor5D dproj_acc=dproj.packed_accessor<scalar_t,5,torch::RestrictPtrTraits,int32_t>();
		sa_map_backward_kernel_weight<scalar_t><<<blocks, threads>>>(dout_acc,proj_acc,dweight_acc,B,T,C,H,W,radius,dilation);
		sa_map_backward_kernel_proj<scalar_t><<<blocks, threads>>>(dout_acc,weight_acc,dproj_acc,B,T,C,H,W,radius,dilation);
	}));
}
