import os
from utils import *
import numpy as np

def run_lumerical(configfile, curmodel="bar3_nearfield_0430.fsp"):
    # Some system specific parameters.
    # Depends on the installation path of Lumerical FDTD
    fdtdsolutions = "\"C:\\Program Files\\Lumerical\\FDTD\\bin\\fdtd-solutions.exe\""
    fspfile = curmodel
    lsffile=configfile
    cmd = " ".join([fdtdsolutions, fspfile, " -nw -run ", lsffile])
    os.system(cmd)
    # with open("Traceback_p0.log", mode='r') as f:
    #     results = f.readlines()

    # os.remove("Traceback_p0.log")
    return 1


def copy_and_modify_lsf(configfile, inputfile, outputfile):
    # import codecs
    with open(configfile,"r", encoding='utf-8') as f:
        lines=f.readlines()
        for i,line in enumerate(lines):
            if "readdata" in line:
                lines[i]='sample100 = readdata("'+inputfile+'");\n'
            if "matlabsave(" in line:
                if "Tlcp" in line:
                    lines[i]='matlabsave("'+outputfile+'/'+pattern+'_Tlcp", T_lcp);\n'
                if "Trcp" in line:
                    lines[i]='matlabsave("'+outputfile+'/'+pattern+'_Trcp", T_rcp);\n'
                if "_CD_" in line:
                    lines[i]='matlabsave("'+outputfile+'/'+pattern+'_CD",CD_);\n'
                if "_pattern_" in line:
                    lines[i]='matlabsave("'+outputfile+'/'+pattern+'_pattern_record", pattern_record);\n'
                if "_E2_lcp_" in line:
                    lines[i]='matlabsave("'+outputfile+'/'+pattern+'_E2_lcp", E2_lcp);\n'
                if "_E2_rcp_" in line:
                    lines[i]='matlabsave("'+outputfile+'/'+pattern+'_E2_rcp",E2_rcp);\n'

        with open(outputfile+"/curtemplate.lsf", "w") as f1:
            f1.write("".join(lines))
                    
    return 1


def built_svm_model():
    from sklearn import svm
    x_best = [1000, 10 ** (-8), 10 ** (-1)]
    svr = svm.SVR(C=x_best[0], epsilon=x_best[1], gamma=x_best[2])
    
    X_train1,Y_train1= amplifyData(X_train, Y_train)
    print("The shape of train X: {}".format(X_train1.shape))
    print("The shape of train Y: {}".format(Y_train1.shape))

    svr.fit(X_train1, Y_train1)
    Y_train_pred = svr.predict(X_train1)
    # print(Y_train_pred)
    print("The default parameters obtained: C=" + str(svr.C) + ", epilson=" + str(svr.epsilon) + ", gamma=" + str(
        svr.gamma))
    print("RMSE = " + str(np.sqrt(np.square(Y_train_pred - Y_train1).mean())))
    print("MAE = " + str(np.mean(np.abs(Y_train_pred - Y_train1))))
    print("The range of pred Y: {}, {},{}".format(np.min(Y_train_pred), np.max(Y_train_pred), np.mean(Y_train_pred)))

    return svr


def countConsNum(x):

    xx1=np.sum((x[:, 0] + 1 - x[:, 3])<=0)
    xx2 = np.sum((x[:, 1] + 1 - x[:, 4]) <= 0)
    xx3 = np.sum((x[:, 2] + 1 - x[:, 5]) <= 0)
    xx4 = np.sum((x[:, 6] + 1 - x[:, 9]) <= 0)
    xx5 = np.sum((x[:, 7] + 1 - x[:, 10]) <= 0)
    xx6 = np.sum((x[:, 8] + 1 - x[:, 11]) <= 0)

    print("If Constrain is working,{},{},{},{},{},{}".format(xx1,xx2,xx3,xx4,xx5,xx6))


# Module 1 ##Generaion initial 400 samples##################################
if __name__ == '__main__':
    
    # 400samples#####################################
    datadir="D:\Liyu\Project1\\bars\LY_XYJ_Opt\Parall1\\"
    filetlcp, filetrcp, filecd, filepara= os.path.join(datadir,"Bars3_Tlcp_test1.txt"), os.path.join(datadir,"Bars3_Trcp_test1.txt"),\
                                        os.path.join(datadir,"Bars3_CD_test1.txt"),os.path.join(datadir,"Parameters_test1.txt")

    trainLCP, trainRCP, trainCD, trainInput = loadData(filetlcp, filetrcp, filecd, filepara)
    print("The shape: {},{},{},{}".format(trainLCP.shape, trainRCP.shape, trainCD.shape, trainInput.shape))

    # remove zero-column
    trainInput = np.delete(trainInput, [3, 7, 11, 15], axis=1)

    # normalization
    X_train = trainInput/4.0
    index_pick = 10
    print("The range of train X: {},{}".format(np.min(X_train), np.max(X_train)))
    print("The range of train all CD: {},{},{}".format(np.min(trainCD), np.max(trainCD),np.mean(np.abs(trainCD))))
    Y_train = trainCD[:, index_pick]
    print("The range of train Y: {}, {}, {}".format(np.min(Y_train), np.max(Y_train), np.mean(np.abs(Y_train))))

    ####################################################################

    # Module 2 ## Set the range ########################################
    vecmax = 10
    vecmin = 0.25
    domain = [{'name': 'var1', 'type': 'continuous', 'domain': (vecmin, vecmax)}]
    for i in range(2, 13):
        domainin = [{'name': 'var' + str(i), 'type': 'continuous', 'domain': (vecmin, vecmax)}]
        domain = domain + domainin

    constraints = [{'name': 'constr_1', 'constraint': 'x[:,0] + 1 - x[:,3]'},
                    {'name': 'constr_2', 'constraint': 'x[:,1] + 1 - x[:,4]'},
                    {'name': 'constr_3', 'constraint': 'x[:,2] + 1 - x[:,5]'},
                    {'name': 'constr_4', 'constraint': 'x[:,6] + 1 - x[:,9]'},
                    {'name': 'constr_5', 'constraint': 'x[:,7] + 1 - x[:,10]'},
                    {'name': 'constr_6', 'constraint': 'x[:,8] + 1 - x[:,11]'}]
    batch_size = 2
    num_cores = 2
    pattern="Bars3"
    ####################################################################


    # Module 3 ## The loop of Bayesian Optimization ##################################
    import GPyOpt
    generation = 10
    for i in range(1, generation):
        print("Producing samples of generation {}".format(i))


        print("The shape of train X: {}".format(X_train.shape))
        print("The shape of train Y: {}".format(Y_train.shape))

        dir_gs="gen{}_data".format(i)
        if not os.path.exists(dir_gs):
            os.makedirs(dir_gs)

        # model = built_svm_model()

        # feasible_region = GPyOpt.Design_space(space=domain)
        BO_CDmax = GPyOpt.methods.BayesianOptimization(f=None,
                                                    domain=domain,
                                                    constrains=constraints,
                                                    model_type='GP',
                                                    X=X_train,
                                                    Y=-np.abs(Y_train[:, None]),
                                                    acquisition_type='EI',
                                                    evaluator_type='local_penalization',
                                                    batch_size=batch_size,
                                                    num_cores=num_cores,
                                                    optimize_restarts=10,
                                                    de_duplication=True,
                                                    acquisition_jitter=0.15  # 越大跳的越远?还是越小？
                                                    )
        # print(Y_train[:16,None])
        x_next = BO_CDmax.suggest_next_locations()
        countConsNum(x_next)

        
        model=built_svm_model()
        y_next=model.predict(x_next)[:,None]

        gen1 = np.concatenate((x_next, y_next), axis=1)
        print("Finish to generate {} samples: {}".format(batch_size, gen1.shape))
        # np.savetxt("gen5_100samples_0418.npy", gen1)
        np.savetxt(dir_gs+"/gen{}_100samples.txt".format(i), gen1, delimiter=",")

        file_gs="gen{}_data/gen{}_100samples.txt".format(i,i)
        
        #modify config file and generate the curtemplate.lsf
        copy_and_modify_lsf("template.lsf", file_gs, dir_gs)
        
        #run FDTD and generate three files about Tlcp, Trcp, CD
        run_lumerical(dir_gs+"/curtemplate.lsf")

        
        gen1dataX, gen1dataY=load_gen_Data(file_gs, dir_gs+"/"+pattern+"_CD.mat", index_pick)

        print("*"*50)
        print("The range of gen {} true Y: {}, {},{}".format(i, np.min(gen1dataY), np.max(gen1dataY),np.mean(np.abs(gen1dataY))))
        print("*" * 50)

        # combined
        X_train=np.concatenate((X_train, gen1dataX),axis=0)
        Y_train = np.concatenate((Y_train, gen1dataY),axis=0)

        print("The shape of train X: {}".format(X_train.shape))
        print("The shape of train Y: {}".format(Y_train.shape))
        # break
####################################################################
