# DataMining-HW3

# Option 2: Acceleratinng code using NVIDIA Rapids + BigQueryML

## Steps

### Motivation

Traditional machine learning algorithms usually run on the CPU rather than GPU. While CPU is versatile and generally is capable enough to handle almost any tasks, GPU is extremely good when it comes to massively parallelized tasks. We will be able to train faster (usually) when using a GPU rather than a CPU.

For this to happen, however, we need **NVIDIA rapids**.

In essence, the collection of packages presented in NVIDIA rapids help users to utilize GPU for traditional machine learning algorithms

### Installation
#### NVIDIA Rapids
First, make sure you have yourself a CUDA compatible NVIDIA GPU.

Next, head over to [Rapids AI](https://docs.rapids.ai/install)

![image](https://github.com/HieuVuong001/DataMining-HW3/assets/60205090/8f3afdd8-e1a8-4efa-948c-b8ee027f4d0e)

Make your selection, and using your favorite package installer to install the require packages.

**NOTE**: Whatever you choose for `ENV.CUDA` during your selection, head over to [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) to get the same version.

If you have multiple cuda versions on your machine:

You can use this command
```
# To select specific cuda version
sudo update-alternatives --config cuda
```

#### BigQuery ML

Head over to [BigQuery ML Introduction](https://cloud.google.com/bigquery/docs/bqml-introduction) for a quick refresh and details on how to setup.

### Training

Details of training (using `cuML` from NVIDIA Rapids) can be found in the code secion of the repository.
Since `cuML` shares the same API interface with that of `sklearn`, the difference ended up being just importing the same package from different libraries.

For BigQueryML, the process can be boiled down to several steps:
- Authenticate
- Upload table into BigQuery
- Train and evaluate model using Google's BigQuery ML

Overall, it is a relatively painless process. Big plus: everything is on the cloud!

### Result

![accuracy](https://github.com/HieuVuong001/DataMining-HW3/assets/60205090/6084fa85-5265-44f9-9af2-ed6782678acf)

![train_time](https://github.com/HieuVuong001/DataMining-HW3/assets/60205090/fe86d321-be88-4e92-8f29-b397f37cfaba)

**NOTE**: Training process is random and thus result yield would also be random. However, the general trend should be the same.
**NOTE**: Google's BigQueryML doesn't allow number of trees to be 1000, and thus was not included into the graph.

**Observation**: We can see that using GPU speeds up the training process by a lot. Consequently, inferencing would be faster as well.

**Observation**: The running time difference between training locally and over the cloud (Google BigQuery) could be due to many factors. My guess is that BigQueryML spends a lot of time setting up the training process rather than actually training the data. Overall, though, Google's BigQueryML is more suited to train Big Data of much larger size, so this example doesn't do it justice. 

**Observation**: Moving data from CPU to GPU costs time. Sometimes, it's the most costly process in the whole training pipeline. PyTorch and Tensorflow often optimize this process for us, but accelerating code ourselves we could easily run into this problem. Example: I tried to speed up XGBoost (not included in NVIDIA Rapids) using GPU for hours but it ended up being about 30%-40% slower than CPU.


# References
[cuML official doc](https://docs.rapids.ai/api/cuml/stable/)


