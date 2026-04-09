from matplotlib import pyplot as plt
import numpy as np
import os
import time

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 6
plt.rcParams['axes.labelsize'] = 6
plt.rcParams['axes.titlesize'] = 6
plt.rcParams['xtick.labelsize'] = 5
plt.rcParams['ytick.labelsize'] = 5
plt.rcParams['legend.fontsize'] = 5

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
        pattern = np.zeros((300, 300), dtype=int)
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

    def draw_pattern(self):
        fig, ax = plt.subplots()
        rect_pixels = self.get_pattern()
        ax.imshow(rect_pixels, origin='lower')
        plt.xticks([]), plt.yticks([])
        plt.show()
        return True




def draw_ep_stru_phase_amplitude(stru,target=100, savepath='none'): 
    # if r_rl[self.target] > 0.05:
    #     return True

    # print(stru)

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

    stru_lambda = stru_lambda[:201]
    r_rl_real = r_rl_real[:201]
    r_lr_real = r_lr_real[:201]
    r_rl_imag = r_rl_imag[:201]
    r_lr_imag = r_lr_imag[:201]
    r_rr_real = r_rr_real[:201]
    r_rr_imag = r_rr_imag[:201]
    eig_state_1_real = eig_state_1_real[:201]
    eig_state_1_imag = eig_state_1_imag[:201]
    eig_state_2_real = eig_state_2_real[:201]
    eig_state_2_imag = eig_state_2_imag[:201]

    r_lr = r_lr_real ** 2 + r_lr_imag ** 2
    r_rl = r_rl_real ** 2 + r_rl_imag ** 2
    r_rr_ll = r_rr_real ** 2 + r_rr_imag ** 2

    delta_imag = abs(eig_state_1_imag[target] - eig_state_2_imag[target])
    delta_real = abs(eig_state_1_real[target] - eig_state_2_real[target])
    # if delta_imag + delta_real > 0.2:
    #     return True

    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    fig = plt.figure(dpi=300, figsize=(24, 12))
    ax1 = fig.add_subplot(241)
    ax1.imshow(imag, origin='lower')
    ax1.set_xticks([]), ax1.set_yticks([])

    ax2 = fig.add_subplot(242)


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

    ax2.set_ylabel('Spectrum dB')
    ax2.set_xlabel('Wavelength (nm)')
    ax2.set_xlim(600, 700)
    ax2.set_ylim(ep_min, 0)
    ax2.legend(loc='center right')
    # ax2.set_title('target = 650 nm, r_rl = {:.4f}'.format(r_rl[target]))

    ax3 = fig.add_subplot(243)

    ax3.plot(stru_lambda, (r_lr), 'r', ms=10, zorder=1, label=u'r_lr')
    ax3.plot(stru_lambda, (r_rl), 'g', ms=10, zorder=1, label=u'r_rl')
    ax3.plot(stru_lambda, (r_rr_ll), 'b', ms=10, zorder=1, label=u'r_rr_ll')

    ep_max = max(max(r_lr), max(r_rl), max(r_rr_ll)) + 0.01
    # 绘制650nm 标注虚线
    ax3.plot([650, 650], [0, ep_max], color='gray', linestyle='--')
    # ax2.plot(stru_lambda, r_lr + r_rl, 'b', ms=10, zorder=1, label=u'total')

    ax3.set_ylabel('Intensity (a.u.)')
    ax3.set_xlabel('Wavelength (nm)')
    ax3.set_xlim(600, 700)
    ax3.set_ylim(0, ep_max)
    ax3.legend(loc='center right')
    ax3.set_title('target = 650 nm, r_rl = {:.4f}'.format(r_rl[target]))

    # 绘制CD图
    ax4 = fig.add_subplot(244)
    CD = abs(r_lr - r_rl) / (r_lr + r_rl + 2 * r_rr_ll)
    ax4.plot(stru_lambda, CD, 'r', ms=10, zorder=1, label=u'CD')

    ep_max = max(CD) + 0.01
    # 绘制650nm 标注虚线
    ax4.plot([650, 650], [0, ep_max], color='gray', linestyle='--')
    # ax2.plot(stru_lambda, r_lr + r_rl, 'b', ms=10, zorder=1, label=u'total')

    ax4.set_ylabel('CD')
    ax4.set_xlabel('Wavelength (nm)')
    ax4.set_xlim(600, 700)
    ax4.set_ylim(0, ep_max)
    ax4.legend(loc='center right')
    ax4.set_title('target = 650 nm, CD = {:.4f}'.format(CD[target]))


    # 绘制本征值实部图
    ax5 = fig.add_subplot(245)
    ax5.plot(stru_lambda, eig_state_1_real, color='royalblue', ms=10, zorder=1, label=u'eig_state_1_real')
    ax5.plot(stru_lambda, eig_state_2_real, color='darkviolet', ms=10, zorder=1, label=u'eig_state_2_real')
    # 绘制650nm 标注虚线
    ax5.plot([650, 650], [max(max(eig_state_1_real), max(eig_state_2_real)),
                            min(min(eig_state_1_real), min(eig_state_2_real))], color='gray', linestyle='--')
    # 绘制0虚线
    ax5.plot([600, 700], [0, 0], color='gray', linestyle='--')

    ax5.set_ylabel('Real eigenvalue')
    ax5.set_xlabel('Wavelength (nm)')
    ax5.set_xlim(600, 700)
    ax5.set_ylim(min(min(eig_state_1_real), min(eig_state_2_real)),
                    max(max(eig_state_1_real), max(eig_state_2_real)))
    ax5.legend()
    ax5.set_title('target = 650 nm, delta real = {:.4f}'.format(delta_real))

    # 绘制本征值虚部图
    ax6 = fig.add_subplot(246)


    ax6.plot(stru_lambda, eig_state_1_imag, color='royalblue', ms=10, zorder=1, label=u'eig_state_1_imag')
    ax6.plot(stru_lambda, eig_state_2_imag, color='darkviolet', ms=10, zorder=1, label=u'eig_state_2_imag')
    # 绘制650nm 标注虚线
    ax6.plot([650, 650], [max(max(eig_state_1_imag), max(eig_state_2_imag)),
                            min(min(eig_state_1_imag), min(eig_state_2_imag))], color='gray', linestyle='--')
    ax6.plot([600, 700], [0, 0], color='gray', linestyle='--')

    ax6.set_ylabel('imag eigenvalue')
    ax6.set_xlabel('Wavelength (nm)')
    ax6.set_xlim(600, 700)
    ax6.set_ylim(min(min(eig_state_1_imag), min(eig_state_2_imag)),
                    max(max(eig_state_1_imag), max(eig_state_2_imag)))
    ax6.legend()

    ax6.set_title('target = 650 nm, delta imag = {:.4f}'.format(delta_imag))

    # 　绘制本征值振幅图
    eig_state_1 = eig_state_1_real + 1j * eig_state_1_imag
    eig_state_2 = eig_state_2_real + 1j * eig_state_2_imag
    delta_amplitude = abs(np.abs(eig_state_1[target]) - np.abs(eig_state_2[target]))
    delta_phase = abs(np.angle(eig_state_1[target]) - np.angle(eig_state_2[target]))
    ax7 = fig.add_subplot(247)


    ax7.plot(stru_lambda, np.abs(eig_state_1), color='royalblue', ms=10, zorder=1, label=u'eig_state_1')
    ax7.plot(stru_lambda, np.abs(eig_state_2), color='darkviolet', ms=10, zorder=1, label=u'eig_state_2')
    # 绘制650nm 标注虚线
    ax7.plot([650, 650], [max(max(np.abs(eig_state_1)), max(np.abs(eig_state_2))),
                            0], color='gray', linestyle='--')

    ax7.set_ylabel('amplitude eigenvalue')
    ax7.set_xlabel('Wavelength (nm)')
    ax7.set_xlim(600, 700)
    ax7.set_ylim(0, max(max(np.abs(eig_state_1)), max(np.abs(eig_state_2))))
    ax7.legend()

    ax7.set_title('amplitude = {:.4f}'.format(delta_amplitude))

    ax8 = fig.add_subplot(248)


    ax8.plot(stru_lambda, np.angle(eig_state_1), color='royalblue', ms=10, zorder=1, label=u'eig_state_1')
    ax8.plot(stru_lambda, np.angle(eig_state_2), color='darkviolet', ms=10, zorder=1, label=u'eig_state_2')

    # 绘制650nm 标注虚线
    ax8.plot([650, 650], [max(max(np.angle(eig_state_1)), max(np.angle(eig_state_2))),
                            min(min(np.angle(eig_state_1)), min(np.angle(eig_state_2)))], color='gray',
                linestyle='--')

    ax8.set_ylabel('phase eigenvalue')
    ax8.set_xlabel('Wavelength (nm)')
    ax8.set_xlim(600, 700)
    ax8.set_ylim(min(min(np.angle(eig_state_1)), min(np.angle(eig_state_2))),
                    max(max(np.angle(eig_state_1)), max(np.angle(eig_state_2))))
    ax8.legend()

    ax8.set_title('phase = {:.4f}'.format(delta_phase))

    # 手动调整子图间距
    plt.subplots_adjust(
        left=0.05,     # 左边距
        right=0.95,    # 右边距
        bottom=0.15,   # 下边距（用于防止标签被截断）
        top=0.90,      # 上边距
        wspace=0.4,    # 水平间距（左右子图之间）
        hspace=0.5     # 垂直间距（行之间）
    )
    # savepath = "C:\\data\\draw"
    # if savepath != 'none':
    #     plt.savefig(savepath)
    plt.show()
    plt.cla()
    plt.close("all")
    return True
    
def draw_ep_stru_phase_amplitude_detail(stru, target = 100, savepath='none'):
    # if r_rl[target] > 0.05:
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

    delta_imag = abs(eig_state_1_imag[target] - eig_state_2_imag[target])
    delta_real = abs(eig_state_1_real[target] - eig_state_2_real[target])
    # if delta_imag + delta_real > 0.2:
    #     return True

    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    fig = plt.figure(dpi=300, figsize=(24, 12))
    ax1 = fig.add_subplot(241)
    ax1.imshow(imag, origin='lower')
    ax1.set_xticks([]), ax1.set_yticks([])

    ax2 = fig.add_subplot(242)


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

    ax2.set_ylabel('Spectrum dB')
    ax2.set_xlabel('Wavelength (nm)')
    ax2.set_xlim(630, 650)
    ax2.set_ylim(ep_min, 0)
    ax2.legend(loc='center right')
    ax2.set_title('target = 650 nm, r_rl = {:.4f}'.format(r_rl[target]))

    ax3 = fig.add_subplot(243)

    ax3.plot(stru_lambda, (r_lr), 'r', ms=10, zorder=1, label=u'r_lr')
    ax3.plot(stru_lambda, (r_rl), 'g', ms=10, zorder=1, label=u'r_rl')
    ax3.plot(stru_lambda, (r_rr_ll), 'b', ms=10, zorder=1, label=u'r_rr_ll')

    ep_max = max(max(r_lr), max(r_rl), max(r_rr_ll)) + 0.01
    # 绘制650nm 标注虚线
    ax3.plot([643.5, 643.5], [0, ep_max], color='gray', linestyle='--')
    # ax2.plot(stru_lambda, r_lr + r_rl, 'b', ms=10, zorder=1, label=u'total')

    ax3.set_ylabel('Intensity (a.u.)')
    ax3.set_xlabel('Wavelength (nm)')
    ax3.set_xlim(630, 650)
    ax3.set_ylim(0, ep_max)
    ax3.legend(loc='center right')
    ax3.set_title('target = 650 nm, r_rl = {:.4f}'.format(r_rl[target]))

    # 绘制CD图
    ax4 = fig.add_subplot(244)
    CD = abs(r_lr - r_rl) / (r_lr + r_rl + 2 * r_rr_ll)
    ax4.plot(stru_lambda, CD, 'r', ms=10, zorder=1, label=u'CD')

    ep_max = max(CD) + 0.01
    # 绘制650nm 标注虚线
    ax4.plot([643.5, 643.5], [0, ep_max], color='gray', linestyle='--')
    # ax2.plot(stru_lambda, r_lr + r_rl, 'b', ms=10, zorder=1, label=u'total')

    ax4.set_ylabel('CD')
    ax4.set_xlabel('Wavelength (nm)')
    ax4.set_xlim(630, 650)
    ax4.set_ylim(0, ep_max)
    ax4.legend(loc='center right')
    ax4.set_title('target = 650 nm, CD = {:.4f}'.format(CD[target]))


    # 绘制本征值实部图
    ax5 = fig.add_subplot(245)
    ax5.plot(stru_lambda, eig_state_1_real, color='royalblue', ms=10, zorder=1, label=u'eig_state_1_real')
    ax5.plot(stru_lambda, eig_state_2_real, color='darkviolet', ms=10, zorder=1, label=u'eig_state_2_real')
    # 绘制650nm 标注虚线
    ax5.plot([643.5, 643.5], [max(max(eig_state_1_real), max(eig_state_2_real)),
                                min(min(eig_state_1_real), min(eig_state_2_real))], color='gray', linestyle='--')
    # 绘制0虚线
    ax5.plot((630, 650), [0, 0], color='gray', linestyle='--')

    ax5.set_ylabel('Real eigenvalue')
    ax5.set_xlabel('Wavelength (nm)')
    ax5.set_xlim(630, 650)
    #         ax5.set_ylim(min(min(eig_state_1_real), min(eig_state_2_real)),
    #                      max(max(eig_state_1_real), max(eig_state_2_real)))
    ax5.set_ylim(-1, 1)
    ax5.legend()
    ax5.set_title('delta real = {:.4f}'.format(delta_real))

    # 绘制本征值虚部图
    ax6 = fig.add_subplot(246)


    ax6.plot(stru_lambda, eig_state_1_imag, color='royalblue', ms=10, zorder=1, label=u'eig_state_1_imag')
    ax6.plot(stru_lambda, eig_state_2_imag, color='darkviolet', ms=10, zorder=1, label=u'eig_state_2_imag')
    # 绘制650nm 标注虚线
    ax6.plot([643.5, 643.5], [max(max(eig_state_1_imag), max(eig_state_2_imag)),
                                min(min(eig_state_1_imag), min(eig_state_2_imag))], color='gray', linestyle='--')
    ax6.plot((630, 650), [0, 0], color='gray', linestyle='--')

    ax6.set_ylabel('imag eigenvalue')
    ax6.set_xlabel('Wavelength (nm)')
    ax6.set_xlim(630, 650)
    ax6.set_ylim(min(min(eig_state_1_imag), min(eig_state_2_imag)),
                    max(max(eig_state_1_imag), max(eig_state_2_imag)))
    ax6.legend()

    ax6.set_title('delta imag = {:.4f}'.format(delta_imag))

    # 　绘制本征值振幅图
    eig_state_1 = eig_state_1_real + 1j * eig_state_1_imag
    eig_state_2 = eig_state_2_real + 1j * eig_state_2_imag
    delta_amplitude = abs(np.abs(eig_state_1[target]) - np.abs(eig_state_2[target]))
    delta_phase = abs(np.angle(eig_state_1[target]) - np.angle(eig_state_2[target]))
    ax7 = fig.add_subplot(247)


    ax7.plot(stru_lambda, np.abs(eig_state_1), color='royalblue', ms=10, zorder=1, label=u'eig_state_1')
    ax7.plot(stru_lambda, np.abs(eig_state_2), color='darkviolet', ms=10, zorder=1, label=u'eig_state_2')
    # 绘制650nm 标注虚线
    ax7.plot([643.5, 643.5], [max(max(np.abs(eig_state_1)), max(np.abs(eig_state_2))),
                                min(min(np.abs(eig_state_1)), min(np.abs(eig_state_2)))],
                color='gray', linestyle='--')

    ax7.set_ylabel('amplitude eigenvalue')
    ax7.set_xlabel('Wavelength (nm)')
    ax7.set_xlim(630, 650)
    ax7.set_ylim(0, max(max(np.abs(eig_state_1)), max(np.abs(eig_state_2))))
    ax7.legend()

    ax7.set_title('amplitude = {:.4f}'.format(delta_amplitude))

    ax8 = fig.add_subplot(248)


    ax8.plot(stru_lambda, np.angle(eig_state_1), color='royalblue', ms=10, zorder=1, label=u'eig_state_1')
    ax8.plot(stru_lambda, np.angle(eig_state_2), color='darkviolet', ms=10, zorder=1, label=u'eig_state_2')

    # 绘制650nm 标注虚线
    ax8.plot([643.5, 643.5], [max(max(np.angle(eig_state_1)), max(np.angle(eig_state_2))),
                                min(min(np.angle(eig_state_1)), min(np.angle(eig_state_2)))], color='gray',
                linestyle='--')

    ax8.set_ylabel('phase eigenvalue')
    ax8.set_xlabel('Wavelength (nm)')
    ax8.set_xlim(630, 650)
    ax8.set_ylim(min(min(np.angle(eig_state_1)), min(np.angle(eig_state_2))),
                    max(max(np.angle(eig_state_1)), max(np.angle(eig_state_2))))
    ax8.legend()

    ax8.set_title('phase = {:.4f}'.format(delta_phase))

    # plt.savefig('D:/sxy/optim-script/fdtd/ep_point.png', dpi = 600)
    if savepath != 'none':
        plt.savefig(savepath)
    # plt.show()
    plt.cla()
    plt.close("all")
    return True
