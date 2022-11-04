import os
import copy
import numpy as np
from FJSP_Data import FJSP_Data


def read_data(path):
    if os.path.exists(path):
        with open(path, 'r') as data:

            lines = data.readlines()
            data_lists = []
            for line in lines:
                data_lists.append(get_list(line))
            origial_data = copy.deepcopy(data_lists)
            lists = [np.array(list).flatten() for list in data_lists]
            origial_data = np.array(lists, dtype=object)
            '''
            下面的整段代码功能可不需要自定义函数get_list和get_num，减少重复调用函数时间，实现如下：
            lines = data.readlines()
            origial_data = []
            data_lists = []
            for line in lines:
                list = []
                list.extend(map(int,line.split()))
                data_lists.append(list)
                origial_data.append(np.array(list))
            '''

            jobs_num = data_lists[0][0]
            machines_num = data_lists[0][1]
            max_machine_per_operation = data_lists[0][2]
            data_lists.pop(0)
            jobs_id = [i for i in range(jobs_num)]
            jobs_operations = np.zeros(
                shape=(1, jobs_num), dtype=np.int64).flatten()
            for i in range(jobs_num):
                jobs_operations[i] = data_lists[i][0]
            jobs_operations_detail = np.zeros(
                shape=(jobs_num, jobs_operations.max(), machines_num), dtype=np.int64)
            for i in range(len(data_lists)):
                position = 1
                list_length = len(data_lists[i])
                operation_number = 0
                while position < list_length:
                    machine_can_operation_num = data_lists[i][position]
                    position += 1
                    for j in range(machine_can_operation_num):
                        machine_number_can_operation = data_lists[i][position]-1
                        position += 1
                        jobs_operations_detail[i][operation_number][machine_number_can_operation] = data_lists[i][position]
                        position += 1
                    operation_number += 1
            
            candidate_machine = np.zeros(shape=(
                jobs_num, jobs_operations.max(), machines_num), dtype=np.int64)
            candidate_machine_index = np.zeros(
                shape=(jobs_num, jobs_operations.max()), dtype=np.int64)
            candidate_machine_time = np.zeros(shape=(
                jobs_num, jobs_operations.max(), machines_num), dtype=np.int64)
            for job_num in range(jobs_num):
                # candidate_machine_j改变，则相应地candidate_machine也改变
                candidate_machine_j = candidate_machine[job_num]
                candidate_machine_index_j = candidate_machine_index[job_num]
                candidate_machine_time_j = candidate_machine_time[job_num]
                for operation_num in range(jobs_operations[job_num]):
                    candidate_machine_o = np.where(
                        jobs_operations_detail[job_num][operation_num] != 0)[0]
                    
                    candidate_machine_index_j[operation_num] = candidate_machine_o.shape[0]
                    candidate_machine_j[operation_num][0:candidate_machine_index_j[operation_num]
                                                       ] = candidate_machine_o
                    candidate_machine_time_o = jobs_operations_detail[
                        job_num][operation_num][candidate_machine_o]
                    candidate_machine_time_j[operation_num][0:candidate_machine_index_j[operation_num]] = candidate_machine_time_o
        return FJSP_Data(path, jobs_num, machines_num, max_machine_per_operation, jobs_id, origial_data, jobs_operations, jobs_operations_detail, candidate_machine, candidate_machine_index, candidate_machine_time)


def get_list(line):
    list = []
    i = 0
    while i < len(line):
        try:
            if line[i].isdigit():
                j = i+1
                while(j < len(line) and line[j].isdigit()):
                    j = j+1
                num = get_num(line, i, j-1)
                i = j
                list.append(num)
            else:
                i = i+1
        except Exception as e:
            print("出现异常")
    return list


def get_num(line, i, j):
    num = 0
    while(i <= j):
        num = num*10 + int(line[i])
        i = i+1
    return num
