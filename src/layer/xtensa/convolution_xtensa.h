#ifndef LAYER_CONVOLUTION_XTENSA_H
#define LAYER_CONVOLUTION_XTENSA_H

#include "convolution.h"

#include "esp_nn.h"


namespace ncnn {

class Convolution_xtensa : virtual public Convolution
{
public:
    Convolution_xtensa();

    virtual int load_param(const ParamDict& pd);

    virtual int load_model(const ModelBin& mb);

    virtual int create_pipeline(const Option& opt);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

protected:
    void make_padding(const Mat& bottom_blob, Mat& bottom_blob_bordered, const Option& opt) const;
    void make_padding(const Mat& bottom_blob, Mat& bottom_blob_bordered, int kernel_w, int kernel_h, const Option& opt) const;

#if NCNN_INT8
    int forward_int8(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
#endif

public:
};

} // namespace ncnn

#endif // LAYER_CONVOLUTION_VULKAN_H
