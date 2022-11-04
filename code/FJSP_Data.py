import numpy as np


class FJSP_Data:
    '''
    数据
    '''

    

    def __init__(self, path, jobs_num, machines_num, max_machine_per_operation, jobs_id, origial_data, jobs_operations, jobs_operations_detail, candidate_machine, candidate_machine_index, candidate_machine_time):
        self.path = path
        self.jobs_num = jobs_num
        self.machines_num = machines_num
        self.max_machine_per_operation = max_machine_per_operation
        self.jobs_id = jobs_id
        self.origial_data = origial_data
        self.jobs_operations = jobs_operations
        self.jobs_operations_detail = jobs_operations_detail
        self.candidate_machine = candidate_machine
        self.candidate_machine_index = candidate_machine_index
        self.candidate_machine_time = candidate_machine_time

    def display_info(self, show_origial_data=False):
        '''
        显示信息
        '''
        print()
        print('一共有{}个零件,{}台机器,每个步骤只能在{}台机器上进行加工。'.format(
            self.jobs_num, self.machines_num, self.max_machine_per_operation))
        if show_origial_data:
            print('原始数据为:', self.origial_data[0])
        print()
        print('零件的编号列表:', self.jobs_id)
        print(self.jobs_operations.shape)
        print('\n')
        for i in range(len(self.jobs_id)):
            id = self.jobs_id[i]
            print('零件{}有{}道工序:'.format(id, self.jobs_operations[i]))
            if show_origial_data:
                print('原始数据为:')
                print(self.origial_data[i+1])
            print('工序候选机器为:')
            print(self.candidate_machine[i])
            print('工序候选机器加工时长:')
            print(self.candidate_machine_time[i])
            print('工序候选机器索引为:')
            print(self.candidate_machine_index[i])
            print()


    path = None
    '''
    文件路径
    '''
    jobs_num = None
    '''
    工件数量
    '''
    machines_num = None
    '''
    机器数量
    '''
    jobs_id = None
    '''
    工件编号
    '''
    origial_data = None
    '''
    原始数据
    '''
    jobs_operations = None
    '''
    零件的工序
    '''
    jobs_operations_detail = None
    '''
    零件加工的详细信息
    '''

    candidate_machine = None
    '''
    候选机器列表
    '''

    candidate_machine_index = None
    '''
    候选机器列表索引
    '''

    candidate_machine_time = None
    '''
    候选机器加工时长
    '''