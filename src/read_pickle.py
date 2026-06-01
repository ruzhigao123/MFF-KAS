import pickle

dse_KERNELS =['doitgen', 'bicg', 'heat-3d', 'jacobi-2d']
if __name__ == '__main__':
    # kernel = 'aes'
    # with open(join(best_result_path , f'{kernel}.pickle'), 'rb') as f:
    #     data = pickle.load(f)
    #     print(list(data))
    res_1 = []
    dict1 = {}
    for kernel in dse_KERNELS:
        with open(f'/home/wslcccc/passion/LLM4DSE/plot/ref/{kernel}.pickle', 'rb') as f:
            data = pickle.load(f)
            data_list = []
            for i in data:
                data_1, _ = i
                data_list.append(data_1)
            print(data_list)
            dict1[kernel] = data_list
    for kernel in dse_KERNELS:
        with open(f'/home/wslcccc/passion/LLM4DSE/plot/GPT4.1/best/{kernel}.pickle', 'rb') as f:
            data = pickle.load(f)
            data_list = []
            for i in data.values():
                data_list.append(i)
            print(data_list[-1])
            dict1[kernel] = data_list