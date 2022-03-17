#include<iostream>
//#include<torch/torch.h>
#include <torch/extension.h>
#include<vector>

//串行比对
void sa_weight_forward_Ref(const torch::Tensor& query,const torch::Tensor& key,torch::Tensor& weight,int B,int T,int C,int H,int W,int radius,int dilation){
	int diameter=2*radius+1;

	for(int batch=0;batch<B;batch++){
		for(int time=0;time<T;time++){
			for(int h=0;h<H;h++){
				for(int w=0;w<W;w++){
					//batch time h w
					for(int cal_time=0;cal_time<T;cal_time++){
						for(int dh=-radius*dilation;dh<=radius*dilation;dh+=dilation){
							for(int dw=-radius*dilation;dw<=radius*dilation;dw+=dilation){
								//reference_position batch cal_time h+dh w+dw
								//float sum=0.0;
								weight[batch][time][cal_time*diameter*diameter+(dh/dilation+radius)*(2*radius+1)+(dw/dilation+radius)][h][w]=0;
								if(dh+h<H&&dw+w<W&&dh+h>=0&&dw+w>=0){
									for(int c=0;c<C;c++){
										//sum+=query[batch][time][c][h][w]*key[batch][cal_time][c][h+dh][w+dw];
										weight[batch][time][cal_time*diameter*diameter+(dh/dilation+radius)*(2*radius+1)+(dw/dilation+radius)][h][w]+=query[batch][time][c][h][w]*key[batch][cal_time][c][h+dh][w+dw];
									}
								}
								//weight[batch][time][cal_time*9+(dh+1)*3+(dw+1)][h][w]=sum;
							}
						}
					}
				}
			}
		}
	}
}

void sa_weight_backward_query_Ref(const torch::Tensor& dweight,const torch::Tensor& query,
		const torch::Tensor& key,torch::Tensor& dquery,int B,int T,int C,int H,int W,int radius,int dilation){
	int diameter=2*radius+1;

	for(int batch=0;batch<B;batch++){
		for(int time=0;time<T;time++){
			for(int h=0;h<H;h++){
				for(int w=0;w<W;w++){
					//batch time h w
					for(int c=0;c<C;c++){
						//double sum=0.0;
						dquery[batch][time][c][h][w]=0;
						//batch time c h w 的梯度来自于45个位置
						for(int cal_time=0;cal_time<T;cal_time++){
							for(int dh=-radius*dilation;dh<=radius*dilation;dh+=dilation){
								for(int dw=-radius*dilation;dw<=radius*dilation;dw+=dilation){
									if(h+dh>=0&&h+dh<H&&w+dw>=0&&w+dw<W){
										//sum+=key[batch][cal_time][c][h+dh][w+dw]*dweight[batch][time][cal_time*9+(dh+1)*3+(dw+1)][h][w];
										dquery[batch][time][c][h][w]+=key[batch][cal_time][c][h+dh][w+dw]*dweight[batch][time][cal_time*diameter*diameter+(dh/dilation+radius)*(2*radius+1)+(dw/dilation+radius)][h][w];
									}
								}
							}
						}
						//dquery[batch][time][c][h][w]=sum;
					}
				}
			}
		}
	}
}

void sa_weight_backward_key_Ref(const torch::Tensor& dweight,const torch::Tensor& query,
		const torch::Tensor& key,torch::Tensor& dkey,int B,int T,int C,int H,int W,int radius,int dilation){
	int diameter=2*radius+1;

	for(int batch=0;batch<B;batch++){
		for(int time=0;time<T;time++){
			for(int h=0;h<H;h++){
				for(int w=0;w<W;w++){
					//batch time h w
					for(int c=0;c<C;c++){
						//d_key的梯度累加
						for(int cal_time=0;cal_time<T;cal_time++){
							for(int dh=-radius*dilation;dh<=radius*dilation;dh+=dilation){
								for(int dw=-radius*dilation;dw<=radius*dilation;dw+=dilation){
									if(h+dh>=0&&h+dh<H&&w+dw>=0&&w+dw<W){
										dkey[batch][cal_time][c][h+dh][w+dw]+=query[batch][time][c][h][w]*dweight[batch][time][cal_time*diameter*diameter+(dh/dilation+radius)*(2*radius+1)+(dw/dilation+radius)][h][w];
									}
								}
							}
						}
					}
				}
			}
		}
	}
}

void sa_map_forward_Ref(const torch::Tensor& weight,const torch::Tensor& proj,torch::Tensor& out,int B,int T,int C,int H,int W,int radius,int dilation){
	int diameter=2*radius+1;

	for(int batch=0;batch<B;batch++){
		for(int time=0;time<T;time++){
			for(int h=0;h<H;h++){
				for(int w=0;w<W;w++){
					//batch time h w
					for(int c=0;c<C;c++){
						//float sum=0.0;
						out[batch][time][c][h][w]=0;
						//batch time c h w
						for(int cal_time=0;cal_time<T;cal_time++){
							for(int dh=-radius*dilation;dh<=radius*dilation;dh+=dilation){
								for(int dw=-radius*dilation;dw<=radius*dilation;dw+=dilation){
									if(h+dh>=0&&h+dh<H&&w+dw>=0&&w+dw<W){
										//sum+=proj[batch][cal_time][c][h+dh][w+dw]*weight[batch][time][cal_time*9+(dh+1)*3+(dw+1)][h][w];
										out[batch][time][c][h][w]+=proj[batch][cal_time][c][h+dh][w+dw]*weight[batch][time][cal_time*diameter*diameter+(dh/dilation+radius)*(2*radius+1)+(dw/dilation+radius)][h][w];
									}
								}
							}
						}
						//out[batch][time][c][h][w]=sum;
					}
				}
			}
		}
	}
}

void sa_map_backward_weight_Ref(const torch::Tensor& dout,const torch::Tensor& weight,
		const torch::Tensor& proj,torch::Tensor& dweight,int B,int T,int C,int H,int W,int radius,int dilation){
	int diameter=2*radius+1;

	for(int batch=0;batch<B;batch++){
		for(int time=0;time<T;time++){
			for(int h=0;h<H;h++){
				for(int w=0;w<W;w++){
					//batch time h w
					for(int cal_time=0;cal_time<T;cal_time++){
						for(int dh=-radius*dilation;dh<=radius*dilation;dh+=dilation){
							for(int dw=-radius*dilation;dw<=radius*dilation;dw+=dilation){
								//reference_position batch cal_time h+dh w+dw
								//float sum=0.0;
								dweight[batch][time][cal_time*diameter*diameter+(dh/dilation+radius)*(2*radius+1)+(dw/dilation+radius)][h][w]=0;
								if(dh+h<H&&dw+w<W&&dh+h>=0&&dw+w>=0){
									for(int c=0;c<C;c++){
										//sum+=dout[batch][time][c][h][w]*proj[batch][cal_time][c][h+dh][w+dw];
										dweight[batch][time][cal_time*diameter*diameter+(dh/dilation+radius)*(2*radius+1)+(dw/dilation+radius)][h][w]+=dout[batch][time][c][h][w]*proj[batch][cal_time][c][h+dh][w+dw];
									}
								}
								//dweight[batch][time][cal_time*9+(dh+1)*3+(dw+1)][h][w]=sum;
							}
						}
					}
				}
			}
		}
	}
}

void sa_map_backward_proj_Ref(const torch::Tensor& dout,const torch::Tensor& weight,
		const torch::Tensor& proj,torch::Tensor& dproj,int B,int T,int C,int H,int W,int radius,int dilation){
	int diameter=2*radius+1;

	for(int batch=0;batch<B;batch++){
		for(int time=0;time<T;time++){
			for(int h=0;h<H;h++){
				for(int w=0;w<W;w++){
					//batch time h w
					for(int c=0;c<C;c++){
						//d_key的梯度累加
						for(int cal_time=0;cal_time<T;cal_time++){
							for(int dh=-radius*dilation;dh<=radius*dilation;dh+=dilation){
								for(int dw=-radius*dilation;dw<=radius*dilation;dw+=dilation){
									if(h+dh>=0&&h+dh<H&&w+dw>=0&&w+dw<W){
										dproj[batch][cal_time][c][h+dh][w+dw]+=dout[batch][time][c][h][w]*weight[batch][time][cal_time*diameter*diameter+(dh/dilation+radius)*(2*radius+1)+(dw/dilation+radius)][h][w];
									}
								}
							}
						}
					}
				}
			}
		}
	}
}
