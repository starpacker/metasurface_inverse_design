import os
file_root = 'D:/subject/physics/AI4S/project1/fdtd'
import time
import matplotlib.pyplot as plt
import numpy as np
import json
import random
#种子 服务器为12 自己电脑为1273
random.seed(1273)
import glob
import datetime
from ipdb import set_trace

def get_Now_YYMMDD():
    # 获取当天年月日 example: 20221111
    dateNow = datetime.date.today()
    return ''+ str(dateNow.year)+ str(dateNow.month) + str(dateNow.day)

class FDTDModel:
    # 生成用于FDTD计算的对象
    # 命名， stru其实是pra矩阵，注意
    def __init__(self,
                 stru,
                 stru_name='pra.txt',
                 data_name="farfile_reflection.txt",
                 model_file="jones_model_400.fsp",
                 mother_script="cal_farfield_data.lsf", ):
        # 结构，模型，lsf脚本
        # para:
        self.stru = stru
        self.stru_name = os.path.join(file_root, stru_name)
        self.data_name = os.path.join(file_root, data_name)
        self.model = os.path.join(file_root, model_file)
        # 建议修改为model_file
        self.mother_scipt = os.path.join(file_root, mother_script)
        # 拼写错误 scipt  script

    def copy_and_modify_lsf(self):

        with open(self.mother_scipt, "r", encoding='utf-8') as f:
            lines = f.readlines()

        lsf_scipt = os.path.join(file_root, "script_copy.lsf")

        with open(lsf_scipt, "w", encoding='utf-8') as f1:
            f1.write("".join(lines))

        # print('copy_and_modify_lst is OK')
        return lsf_scipt

    # sxy add
    def add_quota(self, string):
        # para: string
        # return  " + string + "
        # example input:abc return "abc"
        # use: CMD文件路径有空格直接截断，加上引号正确读路径
        return '\"' + string + '\"'

    def run_lumerical(self, lsf_scipt):
        # para : lumerical script
        # Some system specific parameters.
        # Depends on the installation path of Lumerical fdtd
        fdtd_solutions = 'D:/Program Files/Lumerical/v241/bin/fdtd-solutions.exe'
        fsp_file = self.model
        lsf_file = lsf_scipt
        cmd = " ".join([self.add_quota(fdtd_solutions), fsp_file, " -nw -run ", lsf_file])
        print('cmd:' + cmd)
        os.system(cmd)
        print('run_lumerical is OK')
        return True

    # 调用FDTD进行计算
    def fdtd_cal(self):
        # fdtd sure the out-dated files are deleted
        if os.path.exists(self.stru_name):
            os.remove(self.stru_name)

        np.savetxt(self.stru_name, self.stru, fmt='%d')

        if os.path.exists(self.data_name):
            os.remove(self.data_name)
        # use matrix to generate stru.txt file for fdtd simulation

        lsf_scipt = self.copy_and_modify_lsf()

        self.run_lumerical(lsf_scipt)
        # print('fdtd_cal is  OK')
        return True

    def get_results(self):
        # 返回计算的数据结果\
        # print('get_results 1 is OK')
        self.fdtd_cal()
        print('get_results 2 is OK')
        time.sleep(1)
        while not os.path.exists(self.data_name):
            time.sleep(0.5)
        # print('get_results 3 is OK')
        data = np.loadtxt(self.data_name)
        # print('get_results final is OK')
        return data


class PatternRects:
    # 参数初始化
    def __init__(self, pra):
        # 参数初始化
        self.pra = pra
        self.pra1 = pra[0]
        self.pra2 = pra[1]
        self.pra3 = pra[2]
        self.pra4 = pra[3]
        self.pra5 = pra[4]
        self.pra6 = pra[5]
        self.pra7 = pra[6]
        self.pra8 = pra[7]
        self.pra9 = pra[8]
        self.pra10 = pra[9]

    # done
    def get_pattern(self):
        pattern = np.zeros((400, 400), dtype=int)
        for xi in range(pattern.shape[0]):
            for yj in range(pattern.shape[1]):
                if xi >= self.pra1 and xi <= self.pra1 + self.pra3:
                    if yj >= self.pra10 + self.pra9 + self.pra6 + self.pra5 and \
                            yj <= self.pra10 + self.pra9 + self.pra6 + self.pra5 + self.pra2:
                        pattern[xi, yj] = 1
                if xi >= self.pra1 and xi <= self.pra1 + self.pra4:
                    if yj >= self.pra10 + self.pra9 + self.pra6 and \
                            yj <= self.pra10 + self.pra9 + self.pra6 + self.pra5:
                        pattern[xi, yj] = 1
                if xi >= self.pra7 and xi <= self.pra7 + self.pra8:
                    if yj >= self.pra10 and yj <= self.pra10 + self.pra9:
                        pattern[xi, yj] = 1
        return np.transpose(pattern)

    # done
    def get_optical_response(self):
        stru = self.get_pattern()
        fdtd_model = FDTDModel(self.pra)
        optical_response = fdtd_model.get_results()
        return optical_response

    # 对结构进行描述
    def get_details(self, op_show=False):
        # Eij 表示 i偏振入射情况下的 j偏振的反射光
        op = self.get_optical_response()
        
        print("op shape:", op.shape)
        fre = op[:, 0]
        stru_lambda = op[:, 1]
        Exx_real = op[:, 2]
        Exx_imag = op[:, 3]
        Exy_real = op[:, 4]
        Exy_imag = op[:, 5]
        Eyx_real = op[:, 6]
        Eyx_imag = op[:, 7]
        Eyy_real = op[:, 8]
        Eyy_imag = op[:, 9]

        """
        eig_state_1 = E_data(1:nums,19)+ 1i*E_data(1:nums,20);
        eig_state_2 = E_data(1:nums,21)+ 1i*E_data(1:nums,22);

        r_lr = E_data(1:nums,11)+ 1i*E_data(1:nums,12);
        r_rl = E_data(1:nums,13)+ 1i*E_data(1:nums,14);
        r_rr = E_data(1:nums,15)+ 1i*E_data(1:nums,16);
        r_ll = E_data(1:nums,17)+ 1i*E_data(1:nums,18);

        """
        r_lr_real = op[:, 10]
        r_lr_imag = op[:, 11]
        r_rl_real = op[:, 12]
        r_rl_imag = op[:, 13]
        r_rr_real = op[:, 14]
        r_rr_imag = op[:, 15]
        r_ll_real = op[:, 16]
        r_ll_imag = op[:, 17]

        eig_state_1_real = op[:, 18]
        eig_state_1_imag = op[:, 19]
        eig_state_2_real = op[:, 20]
        eig_state_2_imag = op[:, 21]

        return {'pra': self.pra, 'fre': fre, 'wavelength': stru_lambda,
                'Exx_real': Exx_real, 'Exx_imag': Exx_imag, 'Exy_real': Exy_real, 'Exy_imag': Exy_imag,
                'Eyx_real': Eyx_real, 'Eyx_imag': Eyx_imag, 'Eyy_real': Eyy_real, 'Eyy_imag': Eyy_imag,
                'r_lr_real': r_lr_real, 'r_lr_imag': r_lr_imag, 'r_rl_real': r_rl_real, 'r_rl_imag': r_rl_imag,
                'r_rr_real': r_rr_real, 'r_rr_imag': r_rr_imag, 'r_ll_real': r_ll_real, 'r_ll_imag': r_ll_imag,
                'eig_state_1_real': eig_state_1_real, 'eig_state_1_imag': eig_state_1_imag,
                'eig_state_2_real': eig_state_2_real, 'eig_state_2_imag': eig_state_2_imag}

    def draw_pattern(self):
        fig, ax = plt.subplots()
        rect_pixels = self.get_pattern()
        ax.imshow(rect_pixels, origin='lower')
        plt.xticks([]), plt.yticks([])
        plt.show()
        return True


# 工具类
# 测试获取结构工具是否正常工作
def test_pattern_rects(para=[112, 49, 52, 148, 53, 50, 121, 94, 80, 77]):
    # pra = [40, 60, 50, 150, 300, 60, 50, 300, 60, 60]
    # pra = [100, 100, 50, 200, 50, 80, 50, 300, 60, 60]
    # pra = [112, 49, 52, 148, 53, 50, 121, 94, 80, 77]
    pra = para
    pr = PatternRects(pra)
    # pr.draw_pattern()
    data = pr.get_details()
    # print(data['wavelength'])
    return data


# 批量获得一批结构，满足预先设定约束的结构, [40*40], 每一格代表10nm
class GetSamples:
    # 参数初始化
    def __init__(self, unit=400, step=1):
        # 参数初始化
        self.unit = unit
        self.step = step

    # 随机生成一组参数[pra1-pra10], 满足三个约束条件 #原始使用
    def get_random_pra(self):
        # old version
        #         pra1 = random.randint(30, 160)
        #         pra3 = random.randint(40, 80)
        #         pra4 = random.randint(80, 370-pra1)

        #         pra7 = random.randint(30, 150)
        #         pra8 = random.randint(40, 340-pra7)
        # 随机生成一组参数[pra1-pra10], 满足三个约束条件,note: 左闭右开
        pra1 = random.randrange(30, 160, self.step)
        pra3 = random.randrange(40, 80, self.step)
        pra4 = random.randrange(80, self.unit - 30 - pra1, self.step)

        pra7 = random.randrange(30, 150, self.step)
        pra8 = random.randrange(40, self.unit - 30 - pra7, self.step)

        index = True

        while index:
            pra10 = random.randrange(30, 100, self.step)
            pra9 = random.randrange(40, 100, self.step)
            pra6 = random.randrange(40, 100, self.step)
            pra5 = random.randrange(40, 100, self.step)
            pra2 = random.randrange(40, 100, self.step)
            if pra10 + pra9 + pra6 + pra5 + pra2 <= self.unit:
                return [pra1, pra2, pra3, pra4, pra5, pra6, pra7, pra8, pra9, pra10]
            else:
                index = True

    def get_random_sample(self):
        pra = self.get_random_pra()
        pr = PatternRects(pra)
        # pr.draw_pattern()
        data = pr.get_details(op_show=False)
        return data

    @staticmethod
    def get_sample(pra, op_show=True, draw_pattern=False):
        pr = PatternRects(pra)
        # pr.draw_pattern()
        if draw_pattern:
            pr.draw_pattern()
        data = pr.get_details(op_show=op_show)
        return data


# 测试生成样本类是否正常工作
def test_get_samples():
    while True:
        gs = GetSamples()
        data = gs.get_random_sample()
        print(data)


# 对于给定的数据文件夹进行数据处理并分析
class AnalysisSamples:
    # 参数初始化

    def __init__(self, data_path):
        # 参数初始化: data_path, file_paths, target
        self.data_path = data_path
        self.file_paths = self.get_file_paths()
        self.target = 100

    # 获取文件路径
    def get_file_paths(self):
        file_paths = []
        file_names = os.listdir(self.data_path)
        for file_name in file_names:
            file_path = os.path.join(self.data_path, file_name)
            file_paths.append(file_path)
        return file_paths

    # 读取单个数据
    def read_data(self, file_path, loss_alpha=0):
        data = np.load(file_path, allow_pickle=True).item()
        # print(data['wavelength'][target])
        # 将[[11, 28, 13, 18], [24, 27, 3, 19], [34, 37, 23, 27]]
        # 转换为[11 28 13 18 24 27  3 19 34 37 23 27]
        # rects 变换为3维参数
        target = self.target
        pra = np.array(data['pra'])
        data['pra'] = pra
        delta = ((data['eig_state_1_real'] - data['eig_state_2_real']) ** 2 +
                 (data['eig_state_1_imag'] - data['eig_state_2_imag']) ** 2) ** (1 / 2)

        abs_ev1 = (data['eig_state_1_real'] ** 2 + data['eig_state_1_imag'] ** 2) ** (1 / 2)
        abs_ev2 = (data['eig_state_2_real'] ** 2 + data['eig_state_2_imag'] ** 2) ** (1 / 2)

        #     设置优化函数：L = alpha/(abs(Ev1-Ev2) + 1e-6) + 1/(Ev1+ 1e-6) + 1/(Ev1+1e-6)
        #     alpha 超参数，设置为10

        # goal = (alpha * r_lr) / (r_rl + alpha)  
        # hand-crafted loss function
        loss = loss_alpha / (delta + 1e-6) + 1 / (abs_ev1 + abs_ev2 + 1e-6)

        data['loss'] = loss
        target_goal = loss[target]
        # print(data['wavelength'][target])
        data['target_goal'] = target_goal
        return data   # 偏振转化率

    # 读取整个数据集的数据,准备数据集
    def read_datas(self):
        x_train = []
        y_train = []
        for file_path in self.file_paths:
            data = self.read_data(file_path)
            x_train.append(data['pra'])
            y_train.append(data['target_goal'])   
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        return x_train, y_train

    # 读取整个数据集参数，计算偏振转化率的最大值
    def get_max_loss(self):
        _, y_train = self.read_datas()
        # print(y_train)
        max_ep = np.max(np.abs(y_train))
        return max_ep
    # 读取整个数据集参数，计算偏振转化率的最小值
    def get_min_loss(self):
        _, y_train = self.read_datas()
        # print(y_train)
        min_ep = np.min(np.abs(y_train))
        return min_ep

    # 读取整个数据集参数，计算偏振转化率的平均值
    def get_mean(self):
        _, y_train = self.read_datas()
        mean_ep = np.mean(np.abs(y_train))
        return mean_ep

    # 选出目标goal的平均值最大的n个data
    def get_max_stru(self, num=6):
        datas = []
        for file_path in self.file_paths:
            if '.png' in file_path:
                os.remove(file_path)
                continue
            data = self.read_data(file_path)
            data['file name'] = file_path
            datas.append(data)

        # 将数据按照goal从大到小排序
        datas.sort(key=lambda x: np.mean(x['target_goal']), reverse=True)
        max_data = datas[:num]
        return max_data

    # 选出目标goal的平均值最小的n个data
    def get_min_ep_stru(self, num=6):
        datas = []
        for file_path in self.file_paths:
            if '.png' in file_path:
                os.remove(file_path)
                continue
            data = self.read_data(file_path)
            data['file name'] = file_path
            datas.append(data)

        # 将数据按照goal从小到大排序
        datas.sort(key=lambda x: np.mean(x['target_goal']), reverse=False)
        min_data = datas[:num]
        return min_data

    # 将gen_min - gen_max中的ce最大的n个结构绘图并保存到data-result/max-ce-num的文件夹中
    def save_max_stru(self, genmin, gen_max, num=6):
        gens = np.arange(genmin, gen_max + 1)
        for gen in gens:
            gen_path = self.data_path[:-1] + str(gen)
            gen_analysis = AnalysisSamples(gen_path)
            gen_save_path = gen_path.replace('data_save', r'data_result\max-ep' + str(num))
            gen_save_path = os.path.dirname(gen_save_path)
            # print(gen_save_path)
            if not os.path.exists(gen_save_path):
                os.makedirs(gen_save_path)
            max_ce_strus = gen_analysis.get_max_stru(num=num)
            for stru in max_ce_strus:
                file_name = os.path.basename(stru['file name']).replace('npy', 'png')
                file_name = os.path.join(gen_save_path, 'gen_' + str(gen) + file_name)
                # print(file_name)

                # self.draw_ep_stru(stru, savepath=file_name)  # needs debug
                self.draw_ep_stru_phase_amplitude(stru, savepath=file_name)
        return True
    
    #from 600 to 700
    def draw_ep_stru_phase_amplitude(self, stru, savepath='none'): 
    
        # if r_rl[self.target] > 0.05:
        #     return True

        pra = stru['pra']
        imag = PatternRects(pra).get_pattern()
        stru_lambda = np.array(stru['wavelength'])

        r_lr_real = np.array(stru['r_lr_real'])
        r_lr_imag = np.array(stru['r_lr_imag'])
        r_rl_real = np.array(stru['r_rl_real'])
        r_rl_imag = np.array(stru['r_rl_imag'])
        r_rr_real = np.array(stru['r_rr_real'])
        r_rr_imag = np.array(stru['r_rr_imag'])
        # r_ll_real = stru[:, 16]
        # r_ll_imag = stru[:, 17]
        #
        eig_state_1_real = np.array(stru['eig_state_1_real'])
        eig_state_1_imag = np.array(stru['eig_state_1_imag'])
        eig_state_2_real = np.array(stru['eig_state_2_real'])
        eig_state_2_imag = np.array(stru['eig_state_2_imag'])

        r_lr = r_lr_real ** 2 + r_lr_imag ** 2
        r_rl = r_rl_real ** 2 + r_rl_imag ** 2
        r_rr_ll = r_rr_real ** 2 + r_rr_imag ** 2

        delta_imag = abs(eig_state_1_imag[self.target] - eig_state_2_imag[self.target])
        delta_real = abs(eig_state_1_real[self.target] - eig_state_2_real[self.target])
        # if delta_imag + delta_real > 0.2:
        #     return True

        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'

        fig = plt.figure(dpi=300, figsize=(24, 12))
        ax1 = fig.add_subplot(241)
        ax1.imshow(imag, origin='lower')
        ax1.set_xticks([]), ax1.set_yticks([])

        ax2 = fig.add_subplot(242)

        font = {'family': 'Times New Roman', 'size': 12}
        font_legend = {'family': 'Times New Roman', 'size': 10}

        ax2.plot(stru_lambda, 10 * np.log10(r_lr), 'r', ms=10, zorder=1, label=u'r_lr')
        ax2.plot(stru_lambda, 10 * np.log10(r_rl), 'g', ms=10, zorder=1, label=u'r_rl')
        ax2.plot(stru_lambda, 10 * np.log10(r_rr_ll), 'b', ms=10, zorder=1, label=u'r_rr_ll')  # 原来的

        #         ax2.plot(stru_lambda, (r_lr), 'r', ms=10, zorder=1, label=u'r_lr')
        #         ax2.plot(stru_lambda, (r_rl), 'g', ms=10, zorder=1, label=u'r_rl')
        #         ax2.plot(stru_lambda, (r_rr_ll), 'b', ms=10, zorder=1, label=u'r_rr_ll')

        ep_min = min(min(10 * np.log10(r_lr)), min(10 * np.log10(r_rl)), min(10 * np.log10(r_rr_ll))) - 5
        # 绘制650nm 标注虚线
        ax2.plot([650, 650], [0, ep_min], color='gray', linestyle='--')
        # ax2.plot(stru_lambda, r_lr + r_rl, 'b', ms=10, zorder=1, label=u'total')

        ax2.set_ylabel('Spectrum dB', fontdict=font)
        ax2.set_xlabel('Wavelength (nm)', fontdict=font)
        ax2.set_xlim(600, 700)
        ax2.set_ylim(ep_min, 0)
        ax2.legend(prop=font_legend, loc='center right')
        ax2.set_title('target = 650 nm, r_rl = {:.4f}'.format(r_rl[self.target]), fontdict=font)

        ax3 = fig.add_subplot(243)

        ax3.plot(stru_lambda, (r_lr), 'r', ms=10, zorder=1, label=u'r_lr')
        ax3.plot(stru_lambda, (r_rl), 'g', ms=10, zorder=1, label=u'r_rl')
        ax3.plot(stru_lambda, (r_rr_ll), 'b', ms=10, zorder=1, label=u'r_rr_ll')

        ep_max = max(max(r_lr), max(r_rl), max(r_rr_ll)) + 0.01
        # 绘制650nm 标注虚线
        ax3.plot([650, 650], [0, ep_max], color='gray', linestyle='--')
        # ax2.plot(stru_lambda, r_lr + r_rl, 'b', ms=10, zorder=1, label=u'total')

        ax3.set_ylabel('Intensity (a.u.)', fontdict=font)
        ax3.set_xlabel('Wavelength (nm)', fontdict=font)
        ax3.set_xlim(600, 700)
        ax3.set_ylim(0, ep_max)
        ax3.legend(prop=font_legend, loc='center right')
        ax3.set_title('target = 650 nm, r_rl = {:.4f}'.format(r_rl[self.target]), fontdict=font)

        # 绘制CD图
        ax4 = fig.add_subplot(244)
        CD = abs(r_lr - r_rl) / (r_lr + r_rl + 2 * r_rr_ll)
        ax4.plot(stru_lambda, CD, 'r', ms=10, zorder=1, label=u'CD')

        ep_max = max(CD) + 0.01
        # 绘制650nm 标注虚线
        ax4.plot([650, 650], [0, ep_max], color='gray', linestyle='--')
        # ax2.plot(stru_lambda, r_lr + r_rl, 'b', ms=10, zorder=1, label=u'total')

        ax4.set_ylabel('CD', fontdict=font)
        ax4.set_xlabel('Wavelength (nm)', fontdict=font)
        ax4.set_xlim(600, 700)
        ax4.set_ylim(0, ep_max)
        ax4.legend(prop=font_legend, loc='center right')
        ax4.set_title('target = 650 nm, CD = {:.4f}'.format(CD[self.target]), fontdict=font)

        font = {'family': 'Times New Roman', 'size': 12}
        font_legend = {'family': 'Times New Roman', 'size': 10}

        # 绘制本征值实部图
        ax5 = fig.add_subplot(245)
        ax5.plot(stru_lambda, eig_state_1_real, color='royalblue', ms=10, zorder=1, label=u'eig_state_1_real')
        ax5.plot(stru_lambda, eig_state_2_real, color='darkviolet', ms=10, zorder=1, label=u'eig_state_2_real')
        # 绘制650nm 标注虚线
        ax5.plot([650, 650], [max(max(eig_state_1_real), max(eig_state_2_real)),
                              min(min(eig_state_1_real), min(eig_state_2_real))], color='gray', linestyle='--')
        # 绘制0虚线
        ax5.plot([600, 700], [0, 0], color='gray', linestyle='--')

        ax5.set_ylabel('Real eigenvalue', fontdict=font)
        ax5.set_xlabel('Wavelength (nm)', fontdict=font)
        ax5.set_xlim(600, 700)
        ax5.set_ylim(min(min(eig_state_1_real), min(eig_state_2_real)),
                     max(max(eig_state_1_real), max(eig_state_2_real)))
        ax5.legend(prop=font_legend)
        ax5.set_title('delta real = {:.4f}'.format(delta_real), fontdict=font)

        # 绘制本征值虚部图
        ax6 = fig.add_subplot(246)

        font = {'family': 'Times New Roman', 'size': 12}
        font_legend = {'family': 'Times New Roman', 'size': 10}

        ax6.plot(stru_lambda, eig_state_1_imag, color='royalblue', ms=10, zorder=1, label=u'eig_state_1_imag')
        ax6.plot(stru_lambda, eig_state_2_imag, color='darkviolet', ms=10, zorder=1, label=u'eig_state_2_imag')
        # 绘制650nm 标注虚线
        ax6.plot([650, 650], [max(max(eig_state_1_imag), max(eig_state_2_imag)),
                              min(min(eig_state_1_imag), min(eig_state_2_imag))], color='gray', linestyle='--')
        ax6.plot([600, 700], [0, 0], color='gray', linestyle='--')

        ax6.set_ylabel('imag eigenvalue', fontdict=font)
        ax6.set_xlabel('Wavelength (nm)', fontdict=font)
        ax6.set_xlim(600, 700)
        ax6.set_ylim(min(min(eig_state_1_imag), min(eig_state_2_imag)),
                     max(max(eig_state_1_imag), max(eig_state_2_imag)))
        ax6.legend(prop=font_legend)

        ax6.set_title('delta imag = {:.4f}'.format(delta_imag), fontdict=font)

        # 　绘制本征值振幅图
        eig_state_1 = eig_state_1_real + 1j * eig_state_1_imag
        eig_state_2 = eig_state_2_real + 1j * eig_state_2_imag
        delta_amplitude = abs(np.abs(eig_state_1[self.target]) - np.abs(eig_state_2[self.target]))
        delta_phase = abs(np.angle(eig_state_1[self.target]) - np.angle(eig_state_2[self.target]))
        ax7 = fig.add_subplot(247)

        font = {'family': 'Times New Roman', 'size': 12}
        font_legend = {'family': 'Times New Roman', 'size': 10}

        ax7.plot(stru_lambda, np.abs(eig_state_1), color='royalblue', ms=10, zorder=1, label=u'eig_state_1')
        ax7.plot(stru_lambda, np.abs(eig_state_2), color='darkviolet', ms=10, zorder=1, label=u'eig_state_2')
        # 绘制650nm 标注虚线
        ax7.plot([650, 650], [max(max(np.abs(eig_state_1)), max(np.abs(eig_state_2))),
                              0], color='gray', linestyle='--')

        ax7.set_ylabel('amplitude eigenvalue', fontdict=font)
        ax7.set_xlabel('Wavelength (nm)', fontdict=font)
        ax7.set_xlim(600, 700)
        ax7.set_ylim(0, max(max(np.abs(eig_state_1)), max(np.abs(eig_state_2))))
        ax7.legend(prop=font_legend)

        ax7.set_title('amplitude = {:.4f}'.format(delta_amplitude), fontdict=font)

        ax8 = fig.add_subplot(248)

        font = {'family': 'Times New Roman', 'size': 12}
        font_legend = {'family': 'Times New Roman', 'size': 10}

        ax8.plot(stru_lambda, np.angle(eig_state_1), color='royalblue', ms=10, zorder=1, label=u'eig_state_1')
        ax8.plot(stru_lambda, np.angle(eig_state_2), color='darkviolet', ms=10, zorder=1, label=u'eig_state_2')

        # 绘制650nm 标注虚线
        ax8.plot([650, 650], [max(max(np.angle(eig_state_1)), max(np.angle(eig_state_2))),
                              min(min(np.angle(eig_state_1)), min(np.angle(eig_state_2)))], color='gray',
                 linestyle='--')

        ax8.set_ylabel('phase eigenvalue', fontdict=font)
        ax8.set_xlabel('Wavelength (nm)', fontdict=font)
        ax8.set_xlim(600, 700)
        ax8.set_ylim(min(min(np.angle(eig_state_1)), min(np.angle(eig_state_2))),
                     max(max(np.angle(eig_state_1)), max(np.angle(eig_state_2))))
        ax8.legend(prop=font_legend)

        ax8.set_title('phase = {:.4f}'.format(delta_phase), fontdict=font)

        # plt.savefig('D:/sxy/optim-script/fdtd/ep_point.png', dpi = 600)
        if savepath != 'none':
            plt.savefig(savepath)
        # plt.show()
        plt.cla()
        plt.close("all")
        return True
    
    #from 630 to 650
    def draw_ep_stru_phase_amplitude_detail(self, stru, savepath='none'):
        # if r_rl[self.target] > 0.05:
        #     return True

        pra = stru['pra']
        imag = PatternRects(pra).get_pattern()
        stru_lambda = np.array(stru['wavelength'])

        r_lr_real = np.array(stru['r_lr_real'])
        r_lr_imag = np.array(stru['r_lr_imag'])
        r_rl_real = np.array(stru['r_rl_real'])
        r_rl_imag = np.array(stru['r_rl_imag'])
        r_rr_real = np.array(stru['r_rr_real'])
        r_rr_imag = np.array(stru['r_rr_imag'])
        # r_ll_real = stru[:, 16]
        # r_ll_imag = stru[:, 17]
        #
        eig_state_1_real = np.array(stru['eig_state_1_real'])
        eig_state_1_imag = np.array(stru['eig_state_1_imag'])
        eig_state_2_real = np.array(stru['eig_state_2_real'])
        eig_state_2_imag = np.array(stru['eig_state_2_imag'])

        r_lr = r_lr_real ** 2 + r_lr_imag ** 2
        r_rl = r_rl_real ** 2 + r_rl_imag ** 2
        r_rr_ll = r_rr_real ** 2 + r_rr_imag ** 2

        delta_imag = abs(eig_state_1_imag[self.target] - eig_state_2_imag[self.target])
        delta_real = abs(eig_state_1_real[self.target] - eig_state_2_real[self.target])
        # if delta_imag + delta_real > 0.2:
        #     return True

        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'

        fig = plt.figure(dpi=300, figsize=(24, 12))
        ax1 = fig.add_subplot(241)
        ax1.imshow(imag, origin='lower')
        ax1.set_xticks([]), ax1.set_yticks([])

        ax2 = fig.add_subplot(242)

        font = {'family': 'Times New Roman', 'size': 12}
        font_legend = {'family': 'Times New Roman', 'size': 10}

        ax2.plot(stru_lambda, 10 * np.log10(r_lr), 'r', ms=10, zorder=1, label=u'r_lr')
        ax2.plot(stru_lambda, 10 * np.log10(r_rl), 'g', ms=10, zorder=1, label=u'r_rl')
        ax2.plot(stru_lambda, 10 * np.log10(r_rr_ll), 'b', ms=10, zorder=1, label=u'r_rr_ll')  # 原来的

        #         ax2.plot(stru_lambda, (r_lr), 'r', ms=10, zorder=1, label=u'r_lr')
        #         ax2.plot(stru_lambda, (r_rl), 'g', ms=10, zorder=1, label=u'r_rl')
        #         ax2.plot(stru_lambda, (r_rr_ll), 'b', ms=10, zorder=1, label=u'r_rr_ll')

        ep_min = min(min(10 * np.log10(r_lr)), min(10 * np.log10(r_rl)), min(10 * np.log10(r_rr_ll))) - 5
        # 绘制650nm 标注虚线
        ax2.plot([650, 650], [0, ep_min], color='gray', linestyle='--')
        # ax2.plot(stru_lambda, r_lr + r_rl, 'b', ms=10, zorder=1, label=u'total')

        ax2.set_ylabel('Spectrum dB', fontdict=font)
        ax2.set_xlabel('Wavelength (nm)', fontdict=font)
        ax2.set_xlim(630, 650)
        ax2.set_ylim(ep_min, 0)
        ax2.legend(prop=font_legend, loc='center right')
        ax2.set_title('target = 650 nm, r_rl = {:.4f}'.format(r_rl[self.target]), fontdict=font)

        ax3 = fig.add_subplot(243)

        ax3.plot(stru_lambda, (r_lr), 'r', ms=10, zorder=1, label=u'r_lr')
        ax3.plot(stru_lambda, (r_rl), 'g', ms=10, zorder=1, label=u'r_rl')
        ax3.plot(stru_lambda, (r_rr_ll), 'b', ms=10, zorder=1, label=u'r_rr_ll')

        ep_max = max(max(r_lr), max(r_rl), max(r_rr_ll)) + 0.01
        # 绘制650nm 标注虚线
        ax3.plot([643.5, 643.5], [0, ep_max], color='gray', linestyle='--')
        # ax2.plot(stru_lambda, r_lr + r_rl, 'b', ms=10, zorder=1, label=u'total')

        ax3.set_ylabel('Intensity (a.u.)', fontdict=font)
        ax3.set_xlabel('Wavelength (nm)', fontdict=font)
        ax3.set_xlim(630, 650)
        ax3.set_ylim(0, ep_max)
        ax3.legend(prop=font_legend, loc='center right')
        ax3.set_title('target = 650 nm, r_rl = {:.4f}'.format(r_rl[self.target]), fontdict=font)

        # 绘制CD图
        ax4 = fig.add_subplot(244)
        CD = abs(r_lr - r_rl) / (r_lr + r_rl + 2 * r_rr_ll)
        ax4.plot(stru_lambda, CD, 'r', ms=10, zorder=1, label=u'CD')

        ep_max = max(CD) + 0.01
        # 绘制650nm 标注虚线
        ax4.plot([643.5, 643.5], [0, ep_max], color='gray', linestyle='--')
        # ax2.plot(stru_lambda, r_lr + r_rl, 'b', ms=10, zorder=1, label=u'total')

        ax4.set_ylabel('CD', fontdict=font)
        ax4.set_xlabel('Wavelength (nm)', fontdict=font)
        ax4.set_xlim(630, 650)
        ax4.set_ylim(0, ep_max)
        ax4.legend(prop=font_legend, loc='center right')
        ax4.set_title('target = 650 nm, CD = {:.4f}'.format(CD[self.target]), fontdict=font)

        font = {'family': 'Times New Roman', 'size': 12}
        font_legend = {'family': 'Times New Roman', 'size': 10}

        # 绘制本征值实部图
        ax5 = fig.add_subplot(245)
        ax5.plot(stru_lambda, eig_state_1_real, color='royalblue', ms=10, zorder=1, label=u'eig_state_1_real')
        ax5.plot(stru_lambda, eig_state_2_real, color='darkviolet', ms=10, zorder=1, label=u'eig_state_2_real')
        # 绘制650nm 标注虚线
        ax5.plot([643.5, 643.5], [max(max(eig_state_1_real), max(eig_state_2_real)),
                                  min(min(eig_state_1_real), min(eig_state_2_real))], color='gray', linestyle='--')
        # 绘制0虚线
        ax5.plot((630, 650), [0, 0], color='gray', linestyle='--')

        ax5.set_ylabel('Real eigenvalue', fontdict=font)
        ax5.set_xlabel('Wavelength (nm)', fontdict=font)
        ax5.set_xlim(630, 650)
        #         ax5.set_ylim(min(min(eig_state_1_real), min(eig_state_2_real)),
        #                      max(max(eig_state_1_real), max(eig_state_2_real)))
        ax5.set_ylim(-1, 1)
        ax5.legend(prop=font_legend)
        ax5.set_title('delta real = {:.4f}'.format(delta_real), fontdict=font)

        # 绘制本征值虚部图
        ax6 = fig.add_subplot(246)

        font = {'family': 'Times New Roman', 'size': 12}
        font_legend = {'family': 'Times New Roman', 'size': 10}

        ax6.plot(stru_lambda, eig_state_1_imag, color='royalblue', ms=10, zorder=1, label=u'eig_state_1_imag')
        ax6.plot(stru_lambda, eig_state_2_imag, color='darkviolet', ms=10, zorder=1, label=u'eig_state_2_imag')
        # 绘制650nm 标注虚线
        ax6.plot([643.5, 643.5], [max(max(eig_state_1_imag), max(eig_state_2_imag)),
                                  min(min(eig_state_1_imag), min(eig_state_2_imag))], color='gray', linestyle='--')
        ax6.plot((630, 650), [0, 0], color='gray', linestyle='--')

        ax6.set_ylabel('imag eigenvalue', fontdict=font)
        ax6.set_xlabel('Wavelength (nm)', fontdict=font)
        ax6.set_xlim(630, 650)
        ax6.set_ylim(min(min(eig_state_1_imag), min(eig_state_2_imag)),
                     max(max(eig_state_1_imag), max(eig_state_2_imag)))
        ax6.legend(prop=font_legend)

        ax6.set_title('delta imag = {:.4f}'.format(delta_imag), fontdict=font)

        # 　绘制本征值振幅图
        eig_state_1 = eig_state_1_real + 1j * eig_state_1_imag
        eig_state_2 = eig_state_2_real + 1j * eig_state_2_imag
        delta_amplitude = abs(np.abs(eig_state_1[self.target]) - np.abs(eig_state_2[self.target]))
        delta_phase = abs(np.angle(eig_state_1[self.target]) - np.angle(eig_state_2[self.target]))
        ax7 = fig.add_subplot(247)

        font = {'family': 'Times New Roman', 'size': 12}
        font_legend = {'family': 'Times New Roman', 'size': 10}

        ax7.plot(stru_lambda, np.abs(eig_state_1), color='royalblue', ms=10, zorder=1, label=u'eig_state_1')
        ax7.plot(stru_lambda, np.abs(eig_state_2), color='darkviolet', ms=10, zorder=1, label=u'eig_state_2')
        # 绘制650nm 标注虚线
        ax7.plot([643.5, 643.5], [max(max(np.abs(eig_state_1)), max(np.abs(eig_state_2))),
                                  min(min(np.abs(eig_state_1)), min(np.abs(eig_state_2)))],
                 color='gray', linestyle='--')

        ax7.set_ylabel('amplitude eigenvalue', fontdict=font)
        ax7.set_xlabel('Wavelength (nm)', fontdict=font)
        ax7.set_xlim(630, 650)
        ax7.set_ylim(0, max(max(np.abs(eig_state_1)), max(np.abs(eig_state_2))))
        ax7.legend(prop=font_legend)

        ax7.set_title('amplitude = {:.4f}'.format(delta_amplitude), fontdict=font)

        ax8 = fig.add_subplot(248)

        font = {'family': 'Times New Roman', 'size': 12}
        font_legend = {'family': 'Times New Roman', 'size': 10}

        ax8.plot(stru_lambda, np.angle(eig_state_1), color='royalblue', ms=10, zorder=1, label=u'eig_state_1')
        ax8.plot(stru_lambda, np.angle(eig_state_2), color='darkviolet', ms=10, zorder=1, label=u'eig_state_2')

        # 绘制650nm 标注虚线
        ax8.plot([643.5, 643.5], [max(max(np.angle(eig_state_1)), max(np.angle(eig_state_2))),
                                  min(min(np.angle(eig_state_1)), min(np.angle(eig_state_2)))], color='gray',
                 linestyle='--')

        ax8.set_ylabel('phase eigenvalue', fontdict=font)
        ax8.set_xlabel('Wavelength (nm)', fontdict=font)
        ax8.set_xlim(630, 650)
        ax8.set_ylim(min(min(np.angle(eig_state_1)), min(np.angle(eig_state_2))),
                     max(max(np.angle(eig_state_1)), max(np.angle(eig_state_2))))
        ax8.legend(prop=font_legend)

        ax8.set_title('phase = {:.4f}'.format(delta_phase), fontdict=font)

        # plt.savefig('D:/sxy/optim-script/fdtd/ep_point.png', dpi = 600)
        if savepath != 'none':
            plt.savefig(savepath)
        # plt.show()
        plt.cla()
        plt.close("all")
        return True

    # 绘制优化曲线
    def plot_optim_process(self, genmin, gen_max):
        gens = np.arange(genmin, gen_max + 1)
        mean_ep = []
        min_ep = []
        print(self.data_path)
        #每一个gen都代表着一类数据集
        for gen in gens:
            gen_path = self.data_path[:-1] + str(gen)
            gen_analysis = AnalysisSamples(gen_path)
            gen_mean_ce = gen_analysis.get_mean()
            gen_min_ep = gen_analysis.get_max_loss()
            mean_ep.append(gen_mean_ce)
            min_ep.append(gen_min_ep)
        self.draw_optim_ce_process(gens, mean_ep, min_ep)
        return True

    @staticmethod
    def draw_optim_ce_process(gens, mean_ep, min_ep):
        # fig = plt.figure(dpi=300, figsize=(6, 4.8))
        fig = plt.figure(dpi=300)
        ax = fig.add_subplot(111)

        font = {'family': 'Times New Roman', 'size': 14}
        font_legend = {'family': 'Times New Roman', 'size': 12}

        ax.plot(gens, min_ep, 'r', ms=10, zorder=1, label=u'min_ep')
        ax.plot(gens, mean_ep, 'b', ms=10, zorder=1, label=u'mean_ep')

        ax.set_ylabel('Goal', fontdict=font)
        ax.set_xlabel('Generation', fontdict=font)
        ax.set_xlim(min(gens) - 0.01, max(gens) + 0.01)
        # ax.set_ylim(0, 1.01)

        # ax.legend(prop=font_legend, loc='lower right')
        ax.legend(prop=font_legend)
        plt.show()
        return True

# test samples analysis
def test_analysis_samples():
    time_now = get_Now_YYMMDD()
    # data_root_path = r'D:\sxy' \
    #                  r'\optim-script\data\data_save\EP_data_650nm_220810_alpha0'
    data_root_path = "D:/subject/physics/AI4S/project1/test_data"
    gen_path = os.path.join(data_root_path, 'gen0')
    analysis = AnalysisSamples(gen_path)
    # min_ep = analysis.get_min_ep()
    # print(min_ep)
    # print(max_ep)
    # print(max_ce)
    # mean_ce = analysis.get_mean_ce()
    # print(mean_ce)
    # strus = analysis.get_min_ep_stru(num=1)
    # for stru in strus:
    #     print(stru['rects'], stru['file name'])
    #     print("matrix", stru['matrix'])
    #     print("judge", stru['judge'])
    #     mat = np.array([[stru['matrix'][0], stru['matrix'][1]],[stru['matrix'][1], stru['matrix'][2]]])
    #     # mat = np.array([[1+1j, 1j],[1j, 3+1j]])
    #     # mat = np.array([[1+1j, -1+1j],[-1+1j, -1-1j]])
    #     eigen_value, eigen_vector = np.linalg.eig(mat)
    #     print("eigen_value\n", eigen_value)
    #     print('eigen_vector\n', eigen_vector)
    analysis.save_max_stru(0, 30, num=3)
    # analysis.plot_optim_process(0, 26)


# what is this ?
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# 计算光谱 para是个列表 一维 对应10个参数具体值
# 后面为生成文件夹目录与文件名
def cal_spectrum(para,path = "D:/subject/physics/AI4S/project1/S2Smodel/fdtd",file_name = 'para_epoch'):
#import random
    #print('-----------------------\n-----------------------\n epoch ' + str(1) + ' start\n-----------------------\n-----------------------\n')
    para_dic = {}
    getSampleTest = GetSamples(step = 1)
    #para = getSampleTest.get_random_pra_test()
    print('-----------------------')
    print('structure parameters:',para)
    #FDTD 数值计算
    data_t = test_pattern_rects(para)
    #     para_dic['a'] = data_t['Exx_real'][100]
    #     para_dic['b'] = data_t['Exx_imag'][100]
    #     para_dic['c'] = data_t['Eyx_real'][100]
    #     para_dic['d'] = data_t['Eyx_imag'][100]
    #     para_dic['e'] = data_t['Eyy_real'][100]
    #     para_dic['f'] = data_t['Eyy_imag'][100]
    # 写文件
    write_path = os.path.join(path, file_name + '.json')
    if not os.path.exists(write_path):
        js = json.dumps(data_t,cls = NpEncoder)
        file = open(write_path,'w')
        file.write(js)
    # 画图
    asa = AnalysisSamples("D:/subject/physics/AI4S/project1/S2Smodel/fdtd")
    pic_path = os.path.join(path, file_name + 'png')
    asa.draw_ep_stru_phase_amplitude(data_t,pic_path)

def draw_pattern(data_t,path = "D:/subject/physics/AI4S/project1/S2Smodel/fdtd",file_name = 'para_epoch'):
#import random
    #print('-----------------------\n-----------------------\n epoch ' + str(1) + ' start\n-----------------------\n-----------------------\n')
    para_dic = {}
    getSampleTest = GetSamples(step = 1)
    asa = AnalysisSamples("D:/subject/physics/AI4S/project1/S2Smodel/fdtd")
    pic_path = path + file_name + '.png'
    #asa.draw_ep_stru_phase_amplitude_detail(data_t,pic_path)
    asa.draw_ep_stru_phase_amplitude(data_t,pic_path)


def get_para_random_pra_test(unit=400):
    # para 参数
    # step 每次遍历的跳数
    # random-idx 遍历的参数空间
    max_unit = unit - 10
    min_unit = 10
    # old version
    pra1 = random.randint(min_unit, max_unit - 40)
    pra3 = random.randint(min_unit, max_unit)
    pra4 = random.randint(min_unit, unit - pra1)

    pra7 = random.randint(min_unit, max_unit)
    pra8 = random.randint(min_unit, unit - pra7)
    # 随机生成一组参数[pra1-pra10], 满足三个约束条件,note: 左闭右开
    # 对于已经找到的相近结构+- 5 参数
    # [112, 48, 52, 152, 48, 48, 112, 92, 80, 76]
    #     pra1 = random.randrange(para[0]-random_idx*step,para[0]+(random_idx+1)*step,step)
    #     pra3 = random.randrange(para[2]-random_idx*step,para[2]+(random_idx+1)*step,step)
    #     pra4 = random.randrange(para[3]-random_idx*step,para[3]+(random_idx+1)*step,step)

    #     pra7 = random.randrange(para[6]-random_idx*step,para[6]+(random_idx+1)*step,step)
    #     pra8 = random.randrange(para[7]-random_idx*step,para[7]+(random_idx+1)*step,step)

    index = True

    while index:
        pra2 = random.randint(min_unit, max_unit)
        pra5 = random.randint(min_unit, max_unit)
        pra6 = random.randint(min_unit, max_unit)

        pra9 = random.randint(min_unit, max_unit)
        pra10 = random.randint(min_unit, 100)

        if pra10 + pra9 + pra6 + pra5 + pra2 <= max_unit:
            if pra2 and pra5 and pra6 and pra9 > 0:
                index = False
                return [pra1, pra2, pra3, pra4, pra5, pra6, pra7, pra8, pra9, pra10]
        else:
            index = True


# random swap para space
# 随机生成参数
import datetime
import os
from tkinter import _flatten

if __name__ == '__main__':
# 遍历文件夹的所有文件
    path = 'D:/subject/physics/AI4S/project1/S2Smodel/*'
    files = glob.glob(path)

    file_lst = []
    for file in files:  # 遍历文件夹
        if os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
            files_name = os.listdir(file)
            file_lst.append(files_name)

    list(_flatten(file_lst))
    para = [112, 48, 52, 152, 48, 48, 112, 92, 80, 76]
    for _ in range(1): 

        # 路径
        path = 'D:/subject/physics/AI4S/project1/S2Smodel/fdtd'
        # 添加当前时间构建文件夹
        today = datetime.date.today()
        dir_gs = os.path.join(path, "gen{}".format(today))

        mesh = 1
        # 随机参数
        para_ran = get_para_random_pra_test(400)
        # 构造文件名
        file_name = str(para_ran)
        file_dir = os.path.join(dir_gs, file_name + '.json')
        # 如果文件已经存在 跳过
        if file_dir in file_lst:
            continue
        file_lst.append(file_name + '.json')
        #     file_dir = path + file_name + '.json'

        if not os.path.exists(dir_gs):
            os.makedirs(dir_gs)
        if not os.path.exists(file_dir):
            cal_spectrum(para=para_ran, path=dir_gs, file_name=file_name)
        print('done')
        time.sleep(1)