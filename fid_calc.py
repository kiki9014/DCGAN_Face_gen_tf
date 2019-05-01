import os
import numpy as np
import tensorflow as tf
import datasets.data as dataset
import pickle as pkl
import scipy.linalg as linalg


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = '0,1'
init = tf.global_variables_initializer()
sess = tf.Session(config=config)

graph_path = './inception/inception_v3_fr.pb'
pool_layer_name = "FID_Inception_Net/InceptionV3/Logits/AvgPool_1a_8x8/AvgPool:0"
input_name = 'FID_Inception_Net/input:0'
return_el_name = 'InceptionV3/Logits/AvgPool_1a_8x8/AvgPool:0'
    
with tf.gfile.FastGFile(graph_path, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    pool_layer = tf.import_graph_def(graph_def, name="FID_Inception_Net", return_elements=[return_el_name])[0]
    

ops = pool_layer.graph.get_operations()

for op_idx, op in enumerate(ops):
    for o in op.outputs:
        shape = o.get_shape()
        if shape._dims != []:
            shape = [s.value for s in shape]
            new_shape = []
            for j, s in enumerate(shape):
                if s == 1 and j == 0:
                    new_shape.append(None)
                else:
                    new_shape.append(s)
            o.__dict__['_shape_val'] = tf.TensorShape(new_shape)
         
    
sess.run(init)
data_dir = './data/FFHQ/thumbnails128x128/'
img_list = [img for img in os.listdir(data_dir) if img.split(".")[-1] in ["png", "jpg", "bmp"]]

# total 70000 image used
batch_size = 250
n_batch = 280

n_sample = batch_size * n_batch

pred_arr = np.empty((n_sample,2048))

for i in range(n_batch):
    start = i*batch_size
    end = start + batch_size
    img_list_batch = img_list[start:end]
    batch = dataset.read_data_batch(data_dir, img_list_batch, (64,64), 108)
    pred = sess.run(pool_layer, {input_name:batch})
    pred_arr[start:end] = pred.reshape(batch_size,-1)

print(pred_arr.shape)

mu = np.mean(pred_arr, axis=0)
sigma = np.cov(pred_arr, rowvar=False)

with open(os.path.join(data_dir,"stats.pkl"), 'wb') as f:
    pkl.dump({"mu" : mu, "sigma" : sigma}, f)