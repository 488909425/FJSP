import time
t0 = time.time()
from read_data import read_data
from population import population
import sys


def main(input, output):
    # while(True):
    begin_time = time.time()
    data_path = 'data_final.txt'
    data_path = input
    data = read_data(data_path)
    # data.display_info(True)

    peoples = population(data)

    begin_time = time.time()
    peoples.initial(500)
    peoples.GA(max_step=120, max_no_new_best=20,
               select_type='tournament_memory', tournament_M=4,
               memory_size=0.4,
               find_type='auto', V_C_ratio=0.2,
               crossover_MS_type='uniform', crossover_OS_type='POX',
               mutation_MS_type='random', mutation_OS_type='random',
               VNS_='not', VNS_type='both', VNS_ratio=0.02)

    final = peoples.print(peoples.best_MS, peoples.best_OS)
    final.append('运行时间: '+str(time.time()-begin_time)+'秒'+'\n')
    final.append('求解时间: '+str(peoples.best_score)+'秒'+'\n')
    final.append('收敛次数: '+str(peoples.best_step))
    with open(output, 'w', encoding='utf-8') as f:
        f.writelines(final)
    for line in final:
        print(line)
    # 工件或机器数量过多时不宜做甘特图展示
    #peoples.show_gantt_chart(peoples.best_MS, peoples.best_OS, figsize=(14, 4))
if __name__ == "__main__":
    #main(sys.argv[1], sys.argv[2])    #这种方式只能在脚本的命令行里设置，为python main.py data_final.txt output.txt
    main('data_chusai.txt', 'output.txt')
    print('处理+运行时间:',time.time()-t0)
