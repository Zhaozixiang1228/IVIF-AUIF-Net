# AUIF-Net
 Codes for Efficient and Interpretable Infrared and Visible Image Fusion Via Algorithm Unrolling

- [*[Paper]*](https://ieeexplore.ieee.org/document/9416456)
- [*[ArXiv]*](https://arxiv.org/abs/2003.09210v1)

## Citation

*[Zixiang Zhao](https://zhaozixiang1228.github.io/), [Shuang Xu](https://xsxjtu.github.io/), [Jiangshe Zhang](http://gr.xjtu.edu.cn/web/jszhang), [Chengyang Liang](), [Chunxia Zhang](https://scholar.google.com/citations?user=b5KG5awAAAAJ&hl=zh-CN) and [Junmin Liu](https://scholar.google.com/citations?user=C9lKEu8AAAAJ&hl=zh-CN), "Efficient and Model-Based Infrared and Visible Image Fusion via Algorithm Unrolling," in IEEE Transactions on Circuits and Systems for Video Technology, doi: 10.1109/TCSVT.2021.3075745, https://ieeexplore.ieee.org/document/9416456.*

```
@ARTICLE{9416456,
  author={Zhao, Zixiang and Xu, Shuang and Zhang, Jiangshe and Liang, Chengyang and Zhang, Chunxia and Liu, Junmin},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Efficient and Model-Based Infrared and Visible Image Fusion via Algorithm Unrolling}, 
  year={2021},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TCSVT.2021.3075745}}
```

## Abstract

Infrared and visible image fusion (IVIF) expects to obtain images that retain thermal radiation information from infrared images and texture details from visible images. In this paper, a model-based convolutional neural network (CNN) model, referred to as Algorithm Unrolling Image Fusion (AUIF), is proposed to overcome the shortcomings of traditional CNN-based IVIF models. The proposed AUIF model starts with the iterative formulas of two traditional optimization models, which are established to accomplish two-scale decomposition, i.e., separating low-frequency base information and high-frequency detail information from source images. Then the algorithm unrolling is implemented where each iteration is mapped to a CNN layer and each optimization model is transformed into a trainable neural network. Compared with the general network architectures, the proposed framework combines the model-based prior information and is designed more reasonably. After the unrolling operation, our model contains two decomposers (encoders) and an additional reconstructor (decoder). In the training phase, this network is trained to reconstruct the input image. While in the test phase, the base (or detail) decomposed feature maps of infrared/visible images are merged respectively by an extra fusion layer, and then the decoder outputs the fusion image. Qualitative and quantitative comparisons demonstrate the superiority of our model, which can robustly generate fusion images containing highlight targets and legible details, exceeding the state-of-the-art methods. Furthermore, our network has fewer weights and faster speed.

## Usage

### Training
A pretrained model is available in ```'./Models/TCSVT_Encoder_Base.model'```, ```'./Models/TCSVT_Encoder_Base.model'```, and ```'./Models/TCSVT_Decoder.model'```. We train it on FLIR (180 image pairs) in ```'./Datasets/Train_data_FLIR'```. In the training phase, all images are resize to 128x128 and are transformed to gray pictures.

If you want to re-train this net, you should run ```'train.py'```.

### Testing
The test images used in the paper have been stored in ```'./Test_result/TNO_TCSVT'```, ```'./Test_result/NIR_TCSVT'``` and ```'./Test_result/FLIR_TCSVT'```, respectively.

For other test images, run ```'test.py'``` and find the results in ```'./Test_result/'```.

## DIDFuse

### Illustration of our DIDFuse model.

<img src="image//Framework.png" width="100%" align=center />

### Qualitative fusion results.

<img src="image//Qualitative.png" width="90%" align=center />

### Quantitative  fusion results.

<img src="image//Quantitative.png" width="90%" align=center />

## Related Work

- Zixiang Zhao, Shuang Xu, Chunxia Zhang, Junmin Liu, Jiangshe Zhang and Pengfei Li, *DIDFuse: Deep Image Decomposition for Infrared and Visible Image Fusion.* IJCAI 2020: 970-976, https://www.ijcai.org/Proceedings/2020/135.

- Zixiang Zhao, Shuang Xu, Chunxia Zhang, Junmin Liu, Jiangshe Zhang, *Bayesian fusion for infrared and visible images, Signal Processing,* Volume 177, 2020, 107734, ISSN 0165-1684, https://doi.org/10.1016/j.sigpro.2020.107734.


