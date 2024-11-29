import h5py  
  
file_path = 'F:/实用机器学习/大作业/代码/animal_classifier_model.h5'  
f = h5py.File(file_path, 'r')  
  
print("Objects in the HDF5 file:")  
for name in f:  
    print(name)  
    obj = f[name]  
    if isinstance(obj, h5py.Dataset):  
        # 这是一个数据集，我们可以打印其形状  
        print(obj.name, obj.shape)  
    elif isinstance(obj, h5py.Group):  
        # 这是一个组，我们需要进一步迭代其中的内容  
        for dataset_name in obj:  
            dataset = obj[dataset_name]  
            if isinstance(dataset, h5py.Dataset):  
                # 确保我们处理的是数据集，然后打印其形状  
                print(dataset.name, dataset.shape)  
            # 如果需要，这里可以添加进一步的操作来查看数据集内容  
  
f.close()