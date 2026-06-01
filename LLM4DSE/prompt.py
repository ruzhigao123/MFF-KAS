from networkx.classes import neighbors

from src.config import FLAGS, temperature
from src.saver import saver
from datetime import datetime
from src.utils import MLP, load, get_save_path, argsort, get_root_path, get_src_path, \
    _get_y_with_target, _get_y
from src.programl_data import print_data_stats, _check_any_in_str, NON_OPT_PRAGMAS, WITH_VAR_PRAGMAS, \
    _in_between, _encode_edge_dict, _encode_edge_torch, _encode_X_torch, create_edge_index
from src.model import Net
from src.parameter import DesignSpace, DesignPoint, DesignParameter, get_default_point, topo_sort_param_ids, \
    compile_design_space, gen_key_from_design_point
from src.config_ds import build_config
from src.result import Result
from CoGNN.model_parse import GumbelArgs, EnvArgs, ActionNetArgs, ActivationType
import json
import os
from LLM4DSE.LLM import llm_process_ec
from math import ceil, inf, exp, log10
from os.path import join
import time
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from typing import Deque, Dict, List, Optional, Set, Union, Generator, Any
import sys
import copy
import itertools
import networkx as nx
from collections import OrderedDict
from glob import glob
import pickle
from torch.nn import Sequential, Linear, ReLU
from typing import NamedTuple, Any, Callable
# openal的聊天模型
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from random import uniform, randint, random
from dse import GNNModel, Explorer

import random
import tsplib95
import time
import csv
# langchain用于部署大语言模型
# PromptTemplate是一个语言模型的prompt模版，ChatPromptTemplate是用于chat model的prompt模版
# HumanMessagePromptTemplate是用户发出的prompt
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
# 用于启动AI行为的消息
from langchain.schema.messages import SystemMessage

# 大语言模型chain
from langchain.chains import LLMChain
# 拿到openai的返回结果
from langchain_community.callbacks.manager import get_openai_callback
from src.config import FLAGS
# 拿到date和time
from datetime import datetime

# 测试不同prompt方式生成新解的能力

# few-shot
content1 = '''|----Description of problem and solution properties----|
                You are addressing a multi-objective optimization problem. Your task is to find all the Pareto design configurations(solutions), that no other design configuration 
                in the search space has simultaneously less area and less latency than the Pareto configurations. Each design configuration is associated with two conflicting 
                objective values: area and latency. The following are the possible values for pragma and the method for returning the solution count as well as the format of the solution that will be provided.
                pragmas_possible_values = {pragma_1:possible_values_1,...,pragma_k:possible_values_k}
                solution = <start>{pragma_1:values_1_1,...,pragma_k:values_k_1}<end>
                result_number = N
                |----In-context examples----|
                The examples of pragmas_possible_values and solution are as follows:
                pragmas_possible_values = {'__PIPE__L1': ['off', '', 'flatten'], '__PIPE__L2': ['off', '', 'flatten'], '__TILE__L2': [1, 2, 4, 8, 13]}
                solution_1 = <start>{'__PIPE__L1': '', '__PIPE__L2': '', '__TILE__L2': 1}<end>
                solution_2 = <start>{'__PIPE__L1': '', '__PIPE__L2': 'off', '__TILE__L2': 13}<end>
                solution_3 = <start>{'__PIPE__L1': '', '__PIPE__L2': 'off', '__TILE__L2': 4}<end>
                Examples of incorrectly generated solutions：
                solution_1 = <start>{'__(PIPE__L1': '', '__(PIPE__L2': '', '__(TILE__L2': 1}<end>
                solution_2 = <start>{'__PIPE__L1': 'off', '__(PIPE__L2': '', '__TILE__L2': 4}<end>
                Solutions containing invalid identifiers like <\start> and <\end> will be deemed invalid.
                |----Task instructions----|
                Please follow the instruction step-by-step to generate new design configurations:
                First, if the content of current_population is empty, initialize the method based on the values of pragmas_possible_values and result_number.
                Otherwise, generate potentially better configurations based on the current population.
                Finally, return the generated solution and the number of generated solutions is limited to the value of result_number.
                Please display the complete returned results without any omissions, and show each solution separately.
                You do not need to provide any thought process or analysis process. 
                And the provided solution must be enclosed with <start> and <end> before and after.'''

# CoT
content2 = '''|----Description of problem and solution properties----|
                You are addressing a multi-objective optimization problem. Your task is to find all the Pareto design configurations(solutions), that no other design configuration 
                in the search space has simultaneously less area and less latency than the Pareto configurations. Each design configuration is associated with two conflicting 
                objective values: area and latency. The following are the possible values for pragma and the method for returning the solution count as well as the format of the solution that will be provided.
                pragmas_possible_values = {pragma_1:possible_values_1,...,pragma_k:possible_values_k}
                solution = <start>{pragma_1:values_1_1,...,pragma_k:values_k_1}<end>
                result_number = N
                |----In-context examples----|
                The examples of pragmas_possible_values and solution are as follows:
                pragmas_possible_values = {'__PIPE__L1': ['off', '', 'flatten'], '__PIPE__L2': ['off', '', 'flatten'], '__TILE__L2': [1, 2, 4, 8, 13]}
                solution_1 = <start>{'__PIPE__L1': '', '__PIPE__L2': '', '__TILE__L2': 1}<end>
                solution_2 = <start>{'__PIPE__L1': '', '__PIPE__L2': 'off', '__TILE__L2': 13}<end>
                solution_3 = <start>{'__PIPE__L1': '', '__PIPE__L2': 'off', '__TILE__L2': 4}<end>
                Examples of incorrectly generated solutions：
                solution_1 = <start>{'__(PIPE__L1': '', '__(PIPE__L2': '', '__(TILE__L2': 1}<end>
                solution_2 = <start>{'__PIPE__L1': 'off', '__(PIPE__L2': '', '__TILE__L2': 4}<end>
                Solutions containing invalid identifiers like <\start> and <\end> will be deemed invalid.
                |----Task instructions----|
                Please follow the instruction step-by-step to generate new design configurations:
                First, if the content of current_population is empty, initialize the method based on the values of pragmas_possible_values and result_number.
                Otherwise, generate potentially better configurations based on the current population.
                Finally, return the generated solution and the number of generated solutions is limited to the value of result_number.
                And the provided solution must be enclosed with <start> and <end> before and after.
                |----Example process of generating new solutions----|
                pragmas_possible_values = {'__PIPE__L1': ['off', '', 'flatten'], '__PIPE__L2': ['off', '', 'flatten'], '__TILE__L2': [1, 2, 4, 8, 13]}
                current_solutions = [{'__PIPE__L1': '', '__PIPE__L2': 'off', '__TILE__L2': 13}, {'__PIPE__L1': '', '__PIPE__L2': '', '__TILE__L2': 1}]
                result_number = 5
                First, check current_solutions to determine whether current_solutions is empty. 
                There are two solutions in the "current_solutions" here. 
                Then, make a selection based on the possible values corresponding to each pragmas in pragmas_possible_values. 
                For example, '__PIPE__L1' has three options, and the '__PIPE__L1' values of these two solutions are both ' '. 
                Therefore, the solutions we generate can modify the value of '__PIPE_L1'. 
                Finally, the generated results will be returned in the following form:
                1.<start>{'__PIPE__L1': 'off', '__PIPE__L2': 'off', '__TILE__L2': 13}<end>
                2.<start>{'__PIPE__L1': 'flatten', '__PIPE__L2': '', '__TILE__L2': 1}<end>
                3.<start>{'__PIPE__L1': '', '__PIPE__L2': '', '__TILE__L2': 2}<end>
                4.<start>{'__PIPE__L1': '', '__PIPE__L2': 'flatten', '__TILE__L2': 4}<end>
                5.<start>{'__PIPE__L1': 'flatten', '__PIPE__L2': 'off', '__TILE__L2': 8}<end>
                '''

#OPRO
content3 = '''|----Description of problem and solution properties----|
                You are addressing a multi-objective optimization problem. Your task is to find all the Pareto design configurations(solutions), that no other design configuration 
                in the search space has simultaneously less area and less latency than the Pareto configurations. Each design configuration is associated with two conflicting 
                objective values: area and latency. The following are the possible values for pragma and the method for returning the solution count as well as the format of the solution that will be provided.
                Fitness is the fitness of the current population.
                pragmas_possible_values = {pragma_1:possible_values_1,...,pragma_k:possible_values_k}
                solution = <start>{pragma_1:values_1_1,...,pragma_k:values_k_1}<end>
                result_number = N
                fitness = [fitness_1,fitness_2,...,fitness_n]
                |----In-context examples----|
                The examples of pragmas_possible_values and solution are as follows:
                pragmas_possible_values = {'__PIPE__L1': ['off', '', 'flatten'], '__PIPE__L2': ['off', '', 'flatten'], '__TILE__L2': [1, 2, 4, 8, 13]}
                solution_1 = <start>{'__PIPE__L1': '', '__PIPE__L2': '', '__TILE__L2': 1}<end>
                solution_2 = <start>{'__PIPE__L1': '', '__PIPE__L2': 'off', '__TILE__L2': 13}<end>
                solution_3 = <start>{'__PIPE__L1': '', '__PIPE__L2': 'off', '__TILE__L2': 4}<end>
                Examples of incorrectly generated solutions：
                solution_1 = <start>{'__(PIPE__L1': '', '__(PIPE__L2': '', '__(TILE__L2': 1}<end>
                solution_2 = <start>{'__PIPE__L1': 'off', '__(PIPE__L2': '', '__TILE__L2': 4}<end>
                Solutions containing invalid identifiers like <\start> and <\end> will be deemed invalid.
                |----Task instructions----|
                Please follow the instruction step-by-step to generate new design configurations:
                First, if the content of current_population is empty, initialize the method based on the values of pragmas_possible_values and result_number.
                Otherwise, Generate potentially better configurations based on the current population and corresponding fitness.
                Finally, return the generated solution and the number of generated solutions is limited to the value of result_number.
                Please display the complete returned results without any omissions, and show each solution separately.
                You do not need to provide any thought process or analysis process. 
                And the provided solution must be enclosed with <start> and <end> before and after.'''

# best prompt solution
content4 = '''|----Description of problem and solution properties----|
                You are addressing a multi-objective optimization problem. Your task is to find all the Pareto design configurations(solutions), that no other design configuration 
                in the search space has simultaneously less area and less latency than the Pareto configurations. Each design configuration is associated with two conflicting 
                objective values: area and latency. The following are the possible values for pragma and the method for returning the solution count as well as the format of the solution that will be provided.
                Fitness is the fitness of the current population.
                pragmas_possible_values = {pragma_1:possible_values_1,...,pragma_k:possible_values_k}
                solution = <start>{pragma_1:values_1_1,...,pragma_k:values_k_1}<end>
                result_number = N
                fitness = [fitness_1,fitness_2,...,fitness_n]
                |----In-context examples----|
                The examples of pragmas_possible_values and solution are as follows:
                pragmas_possible_values = {'__PIPE__L1': ['off', '', 'flatten'], '__PIPE__L2': ['off', '', 'flatten'], '__TILE__L2': [1, 2, 4, 8, 13]}
                solution_1 = <start>{'__PIPE__L1': '', '__PIPE__L2': '', '__TILE__L2': 1}<end>
                solution_2 = <start>{'__PIPE__L1': '', '__PIPE__L2': 'off', '__TILE__L2': 13}<end>
                solution_3 = <start>{'__PIPE__L1': '', '__PIPE__L2': 'off', '__TILE__L2': 4}<end>
                Examples of incorrectly generated solutions：
                solution_1 = <start>{'__(PIPE__L1': '', '__(PIPE__L2': '', '__(TILE__L2': 1}<end>
                solution_2 = <start>{'__PIPE__L1': 'off', '__(PIPE__L2': '', '__TILE__L2': 4}<end>
                Solutions containing invalid identifiers like <\start> and <\end> will be deemed invalid.
                |----Task instructions----|
                Please follow the instruction step-by-step to generate new design configurations:
                First, if the content of current_population is empty, initialize the method based on the values of pragmas_possible_values and result_number.
                Otherwise, Generate potentially better configurations based on the current population and corresponding fitness.
                Finally, return the generated solution and the number of generated solutions is limited to the value of result_number.
                Please display the complete returned results without any omissions, and show each solution separately.
                List the generated solutions together without adding other headings.
                And the provided solution must be enclosed with <start> and <end> before and after.
                |----Example process of generating new solutions----|
                pragmas_possible_values = {'__PIPE__L1': ['off', '', 'flatten'], '__PIPE__L2': ['off', '', 'flatten'], '__TILE__L2': [1, 2, 4, 8, 13]}
                current_solutions = [{'__PIPE__L1': '', '__PIPE__L2': 'off', '__TILE__L2': 13}, {'__PIPE__L1': '', '__PIPE__L2': '', '__TILE__L2': 1}]
                fitness = [5.9814, 6.1545]
                result_number = 5
                First, check current_solutions to determine whether current_solutions is empty. 
                There are two solutions in the "current_solutions" here. 
                Then, make a selection based on the possible values corresponding to each pragmas in pragmas_possible_values. 
                For example, '__PIPE__L1' has three options, and the '__PIPE__L1' values of these two solutions are both ' '. 
                Therefore, the solutions we generate can modify the value of '__PIPE_L1'. 
                In addition, try to understand the relationship between the different pragmas values in fitness and solutions.
                Finally, the generated results will be returned in the following form:
                1.<start>{'__PIPE__L1': 'off', '__PIPE__L2': 'off', '__TILE__L2': 1}<end>
                2.<start>{'__PIPE__L1': 'flatten', '__PIPE__L2': '', '__TILE__L2': 1}<end>
                3.<start>{'__PIPE__L1': '', '__PIPE__L2': '', '__TILE__L2': 1}<end>
                4.<start>{'__PIPE__L1': '', '__PIPE__L2': 'flatten', '__TILE__L2': 1}<end>
                5.<start>{'__PIPE__L1': 'flatten', '__PIPE__L2': 'off', '__TILE__L2': 1}<end>
                '''


def llm_process(llm, tokens, secs, fitness, current_population, pragmas_possible_value, result_number, mode):
    if mode == 'few-shot':
        system_message = SystemMessage(content=content1)
        input_message = HumanMessagePromptTemplate.from_template('''
                    current_population: {current_population}
                    pragmas_possible_value: {pragmas_possible_value}
                    result_number: {result_number}
                    ''')
    elif mode == 'CoT':
        system_message = SystemMessage(content=content2)
        input_message = HumanMessagePromptTemplate.from_template('''
                    current_population: {current_population}
                    pragmas_possible_value: {pragmas_possible_value}
                    result_number: {result_number}
                    ''')
    elif mode == 'OPRO':
        system_message = SystemMessage(content=content3)
        input_message = HumanMessagePromptTemplate.from_template('''
                    current_population: {current_population}
                    fitness: {fitness}
                    pragmas_possible_value: {pragmas_possible_value}
                    result_number: {result_number}
                    ''')
    else:
        system_message = SystemMessage(content=content4)
        input_message = HumanMessagePromptTemplate.from_template('''
                    current_population: {current_population}
                    fitness: {fitness}
                    pragmas_possible_value: {pragmas_possible_value}
                    result_number: {result_number}
                    ''')

    if llm == "deepseek-ai/deepseek-r1":
        template = ChatPromptTemplate.from_messages(
            [{"role":"system", "content":system_message},
            {"role":"user", "content": input_message}]
        )
    else:
        template = ChatPromptTemplate.from_messages(
            [system_message, input_message]
        )

    with get_openai_callback() as cb:
        chain = LLMChain(llm=llm, prompt=template)
        tokens += cb.total_tokens
        if tokens >= 90000:
            time.sleep(62 - secs)
        if mode == 'few-shot' and 'CoT':
            res = chain.run(current_population=current_population, pragmas_possible_value=pragmas_possible_value, result_number=result_number)
        else:
            res = chain.run(fitness=fitness, current_population=current_population, pragmas_possible_value=pragmas_possible_value, result_number=result_number)
    return res, tokens



class PromptExplorer(Explorer):
    def __init__(self, path_kernel: str, kernel_name: str, path_graph: str, run_dse: bool = True,
                 prune_invalid=FLAGS.prune_class, point: DesignPoint = None):
        """Constructor.

        Args:
            ds: Design space.
        """
        # 用Exhaustive方法去DSE
        super(PromptExplorer, self).__init__(path_kernel, kernel_name, path_graph, run_dse, prune_invalid)
        self.batch_size = 1
        self.log.info('Done init')
        if self.run_dse:
            self.run()
            # 拿到最好结果的属性和属性值
            attrs = vars(self.best_result)
            self.log.info('Best Results Found:')
            i = 1
            with open(join(f'/home/wslcccc/passion/LLM4DSE/plot/{llm_model}/{mode}', f'{kernel_name}.pickle'), 'wb') as handle:
                pickle.dump(self.best_save_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.flush()
            # 输出最好结果
            for _, result in sorted(self.best_results_dict.items()):
                attrs = vars(result)
                self.log.info(f'Design {i}')
                self.log.info(', '.join("%s: %s" % item for item in attrs.items()))
                i += 1
        # 不运行dse，直接返回结果
        else:
            results = self.get_results([point])
            attrs = vars(results[0])
            self.log.info(', '.join("%s: %s" % item for item in attrs.items()))

    def load_llm_process(self, fitness, current_population, pragmas_possible_value, result_number, mode):
        tokens = 0
        llm = ChatOpenAI(model=FLAGS.llm_model, temperature=FLAGS.temperature, openai_api_key=FLAGS.api_key,
                         openai_api_base=FLAGS.api_base, request_timeout=2000, streaming=True, callbacks=[StreamingStdOutCallbackHandler()])
        start_time = datetime.now()
        secs = (datetime.now() - start_time).total_seconds()
        if secs >= 60:
            secs = 60
            tokens = 0
        res, tokens = llm_process(llm=llm, tokens=tokens, secs=secs, fitness=fitness, current_population=current_population,
                                     pragmas_possible_value=pragmas_possible_value,
                                     result_number=result_number,mode=mode)
        return res, tokens
    def res_to_config(self, res, logger):
        logger.info('starting to transfer res to config')
        res = res.split('\n')
        # res = [i for i in res if i.find('<start>') >= 0 and i.find('<end>') >= 0]
        # res = [eval(i.split('<start>')[1].split('<end>')[0]) for i in res]
        res = [i for i in res if i.find('<start>') != -1 and i.find('<end>') != -1]
        res = [eval((('{' + i.split('{')[1]).split('}')[0]) + '}') for i in res]
        return res

    def run(self) -> None:
        if self.ds_size <= 500:
            result_number = 10
            stop_cond = ceil(0.5 * self.ds_size)
        elif 500 < self.ds_size <= 10000:
            result_number = 20
            stop_cond = ceil(0.3 * self.ds_size)
        elif 10000 < self.ds_size <= 100000:
            result_number = 30
            stop_cond = ceil(0.05 * self.ds_size)
        elif 100000 < self.ds_size <= 1e8:
            result_number = 40
            stop_cond = ceil(0.0005 * self.ds_size)
        elif 1e8 < self.ds_size <= 1e9:
            result_number = 50
            stop_cond = ceil(0.000005 * self.ds_size)
        else:
            result_number = 60
            stop_cond = ceil(0.0000005 * self.ds_size)

        timer = time.time()
        stop_flag = 0
        defaults_dict, config_options, config_cond = self.get_config_dafault_options()
        best_population = []
        current_population = []
        # evolutionary algorithm body
        while (time.time() - timer) < self.timeout and stop_flag <= stop_cond:
            if len(current_population):
                results = self.get_results(current_population)
                for r in results:
                    stop_flag += 1
                    self.explored_point += 1
                    fitness = [i.quality for i in self.best_results_dict.values()]
                    self.best_save_results[self.explored_point] = [i.quality for i in self.best_results_dict.values()]
                    if isinstance(r, Result):
                        attrs = vars(r)
                        flag = self.update_best(r)
                population_list = [i.point for i in self.best_results_dict.values()]
                fitness = [i.quality for i in self.best_results_dict.values()]
                best_population = []
                for i in population_list:
                    for j in i.keys():
                        temp = i[j]
                        if torch.is_tensor(temp):
                            i[j] = int(temp)
                    best_population.append(i)
                if mode == 'few-shot' and 'CoT':
                    res, tokens = self.load_llm_process(fitness, current_population, config_options, result_number, mode)
                else:
                    res, tokens = self.load_llm_process(fitness, best_population, config_options, result_number,
                                                        mode)
            else:
                fitness = []
                if mode == 'few-shot' and 'CoT':
                    res, tokens = self.load_llm_process(fitness, current_population, config_options, result_number, mode)
                else:
                    res, tokens = self.load_llm_process(fitness, best_population, config_options, result_number, mode)
            self.log.info(f'The LLM model successfully returns the result！！！')
            config = self.res_to_config(res, self.log)
            self.log.info(f'The LLM generates {len(config)} design points')
            current_population += config
        self.log.info(f'Explored {self.explored_point} points')

from src.config import FLAGS
from src.train import train_main, inference
from src.saver import saver
import os.path as osp
from src.utils import get_root_path
from os.path import join
import src.config
from src.programl_data import get_data_list, MyOwnDataset

if __name__ == '__main__':
    mode = 'best' # 'CoT', 'OPRO', 'best', 'few-shot'
    llm_model = 'GPT4.1' # GPT4.1, GPT3.5_turbo
    Kernels = ['bicg']
    # Kernels = ['doitgen', 'bicg', 'heat-3d', 'jacobi-2d']
    for k in Kernels:
        saver.info('#################################################################')
        saver.info(f'Starting DSE for {k}')
        path = join(get_root_path(), 'dse_database', 'poly', 'config')
        path_graph = join(get_root_path(), 'dse_database', 'programl', 'poly', 'processed')
        PromptExplorer(path, k, path_graph, run_dse = True)
        saver.info('#################################################################')
        saver.info(f'')
    saver.close()