# CO3Net: Coordinate-Aware Contrastive Competitive Neural Network for Palmprint Recognition

This repository is a PyTorch implementation of CO3Net (accepted by IEEE Transactions on Instrumentation and Measurement)

#### Abstract
Palmprint recognition achieves high discrimination for identity verification. Compared with handcrafted local texture descriptors, convolutional neural networks (CNNs) can spontaneously learn optimal discriminative features without any prior knowledge. To further enhance the features' representation and discrimination, we propose a coordinate-aware contrastive competitive neural network (CO$_3$Net) for palmprint recognition. To extract the multi-scale textures, CO$_3$Net consists of three parallel learnable Gabor filters (LGF)-based texture extraction branches that learn the discriminative and robust ordering features. Due to the heterogeneity of palmprints, the effects of different textures on the final recognition performance are inconsistent, and dynamically focusing on the textures is beneficial to the performance improvement. Then, CO$_3$Net introduces the attention modules to explore the spatial information, and selects more robust and discriminative textures. Specifically, coordinate attention is embedded into CO$_3$Net to adaptively focus on the important textures from the positional information. Since it is difficult for the cross-entropy loss to build a compact intra-class and separate inter-class feature space, the contrastive loss is employed to jointly optimize the network. CO$_3$Net is validated on four public datasets, and the results demonstrate the remarkable recognition performance of the proposed CO$_3$Net compared to other state-of-the-art methods.


#### Citation
If our work is valuable to you, please cite our work:
```
@article{yang2022mtcc,
  title={CO3Net: Coordinate-Aware Contrastive Competitive Neural Network for Palmprint Recognition},
  author={Yang, Ziyuan and Xia, Wenjun and Qiao, Yifan and Lu, Zexin and Zhang, Bob and Leng, Lu and Zhang, Yi},
  journal={IEEE Transactions on Instrumentation and Measurement},
  year={2023},
  publisher={IEEE}
}
```

#### Requirements
Our codes were implemented by ```PyTorch 1.10``` and ```11.3``` CUDA version. If you wanna try our method, please first install necessary packages as follows:

```
pip install requirements.txt
```

#### Data Preprocessing
To help readers to reproduce our method, we also release our training and testing lists (including PolyU, Tongji, IITD, Multi-Spectrum datasets). If you wanna try our method in other datasets, you need to generate training and testing texts as follows:

```
python ./data/genText.py
```

#### Training
After you prepare the training and testing texts, then you can directly run our training code as follows:

```
python train.py --id_num xxxx --train_set_file xxxx --test_set_file xxxx --des_path xxxx --path_rst xxxx
```

* batch_size: training batch

#### Acknowledgments
Thanks to my all cooperators, they contributed so much to this work.

#### Reference
We refer to the following repositories:
* https://github.com/JonnyLewis/compnet
* https://github.com/houqb/CoordAttention
