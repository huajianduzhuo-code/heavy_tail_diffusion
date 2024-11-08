import numpy as np
import seaborn
import json
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from scipy.stats import gaussian_kde


def drop_top_percentile(arr, percentile=0.5):
    arr = arr + np.random.uniform(0,1e-6,arr.shape) # add small noise to ensure that all values are different
    # Flatten the array to simplify the operation
    flattened_arr = arr.flatten()
    abs_values = np.abs(flattened_arr)
    threshold = np.percentile(abs_values, 100 - percentile)
    mask = abs_values < threshold
    filtered_flattened_arr = flattened_arr[mask]
    return filtered_flattened_arr


def print_and_save(text, file_path):
    print(text)
    with open(file_path, "a") as file:  # 'a' mode appends to the file without overwriting
        file.write(text + "\n")

def plot_density(config, all_target, samples, datafolder):
    # Create a figure and an array of subplots
    fig, axes = plt.subplots(config["data"]["target_L"], config["data"]["target_dim"], figsize = (4*config["data"]["target_dim"], 4*config["data"]["target_L"]))  # Adjust the figsize to fit your screen
    axes = np.array(axes).reshape(config["data"]["target_L"], config["data"]["target_dim"])
    for L in range(config["data"]["target_L"]):
        for dim in range(config["data"]["target_dim"]):
            ax = axes[L,dim]
            bins = 20
            real_data = all_target[:,config["data"]["condition_L"]+L,dim].squeeze().cpu().numpy().copy()
            # real_noised_data = all_noised_target[:,config["data"]["condition_L"]+L,dim].squeeze().cpu().numpy().copy()
            fake_data = samples[:,config["data"]["condition_L"]+L,dim].squeeze().cpu().numpy().copy()
            # print(np.var(real_data), np.var(real_noised_data), np.var(fake_data))

            real_data = real_data[real_data < np.percentile(real_data, 99.9)]
            fake_data = fake_data[fake_data < np.percentile(fake_data, 99.9)]


            ax.hist(real_data,alpha=0.5,bins=bins,density=True,label='real')
            # ax.hist(real_noised_data,alpha=0.5,bins=bins,density=True,label='noised real')
            ax.hist(fake_data,alpha=0.5,bins=bins,density=True,label='fake')
            line1=seaborn.kdeplot(real_data,label='real', ax=ax)
            # line1=seaborn.kdeplot(real_noised_data,label='noised real', ax=ax)
            line1=seaborn.kdeplot(fake_data,label='fake', ax=ax)
            ax.legend()

    plt.savefig('./save/'+datafolder+'/visualize/density_plot.png')
    plt.tight_layout()  # Adjusts subplot params so that subplots are nicely fit in the figure area
    plt.show()
    plt.close()


def plot_density(model_list, fake_samples, real_samples, datafolder, exp_name, target_dim = [0,1,2,3,4], target_L = [1], color_lis = ['yellowgreen','cornflowerblue'], shape = None, save_fig = True, drop_percentile = 0.5, bw_adjust=1):
    # Create a figure and an array of subplots
    # plt.rcParams["font.size"] = 16
    if shape is None:
        fig, axes = plt.subplots(len(target_L), len(target_dim), figsize = (4*len(target_dim), 4*len(target_L)))  # Adjust the figsize to fit your screen
        axes = np.array(axes).reshape(len(target_L), len(target_dim))
    else:
        fig, axes = plt.subplots(shape[0], shape[1], figsize = (4*shape[1], 4*shape[0]))  # Adjust the figsize to fit your screen
        axes = np.array(axes).reshape(shape[0], shape[1])
    loc_col = -1
    loc_row = 0
    for L in range(len(target_L)):
        for K in range(len(target_dim)):
            if shape is None:
                loc_row = L
                loc_col = K
            else:
                if loc_col<shape[1]-1:
                    loc_col+=1
                else:
                    loc_col=0
                    loc_row+=1
            ax = axes[loc_row,loc_col]
            dim = target_dim[K]
            time = target_L[L]

            bins = 20

            real_data = real_samples[:,time,dim].squeeze().copy()
            real_data.sort()
            real_data = drop_top_percentile(real_data, drop_percentile)
            # ax.hist(real_data,alpha=0.2,bins=bins,density=True,label='real')
            seaborn.kdeplot(real_data,label='real', ax=ax, bw_adjust=bw_adjust, color=color_lis[0])

            c=1
            for model_name in fake_samples.keys():
                if model_name not in model_list:
                    continue
                fake_data = fake_samples[model_name][:,time,dim].squeeze().copy()
                fake_data.sort()
                fake_data = drop_top_percentile(fake_data, drop_percentile)
                # ax.hist(fake_data,alpha=0.2,bins=bins,density=True,label=model_name)
                seaborn.kdeplot(fake_data,label=model_name, ax=ax, bw_adjust=bw_adjust, color=color_lis[c])
                c = c+1
                    
                
            ax.set_xlabel('Value')
            ax.set_ylabel('probability density')
            if exp_name == "pareto":
                ax.set_title('Density on dimension '+str(dim+11))
            elif exp_name == "queue":
                ax.set_title('Density Plot on station '+str(dim+1))
            elif exp_name == "vectorAR":
                ax.set_title('Density Plot on dimension '+str(dim+1)+ f', for t={target_L[L]+1}')
            ax.legend()
            
            

    plt.tight_layout()  # Adjusts subplot params so that subplots are nicely fit in the figure area
    plt.subplots_adjust(hspace=0.35)
    if save_fig:
        plt.savefig(f'./save/{datafolder}/{exp_name}_density_plot_{len(target_dim)*len(target_L)}.png')
    plt.show()



def plot_QQ(model_list, fake_samples, real_samples, datafolder, exp_name, target_dim = [0,1,2,3,4], target_L = [1], color_lis = ['yellowgreen','cornflowerblue'], shape = None, save_fig = True, drop_percentile = 0.5):
    # Create a figure and an array of subplots
    # plt.rcParams["font.size"] = 16
    if shape is None:
        fig, axes = plt.subplots(len(target_L), len(target_dim), figsize = (4*len(target_dim), 4*len(target_L)))  # Adjust the figsize to fit your screen
        axes = np.array(axes).reshape(len(target_L), len(target_dim))
    else:
        fig, axes = plt.subplots(shape[0], shape[1], figsize = (4*shape[1], 4*shape[0]))  # Adjust the figsize to fit your screen
        axes = np.array(axes).reshape(shape[0], shape[1])
    loc_col = -1
    loc_row = 0
    for L in range(len(target_L)):
        for K in range(len(target_dim)):
            if shape is None:
                loc_row = L
                loc_col = K
            else:
                if loc_col<shape[1]-1:
                    loc_col+=1
                else:
                    loc_col=0
                    loc_row+=1
            ax = axes[loc_row,loc_col]
            dim = target_dim[K]
            time = target_L[L]

            real_data = real_samples[:,time,dim].squeeze().copy()
            real_data.sort()
            real_data = drop_top_percentile(real_data, drop_percentile)
            max_val = real_data[-1]
            min_val = real_data[0]

            c=0
            for model_name in fake_samples.keys():
                if model_name not in model_list:
                    continue
                fake_data = fake_samples[model_name][:,time,dim].squeeze().copy()
                fake_data.sort()
                fake_data = drop_top_percentile(fake_data, drop_percentile)
                if "SHD" not in model_name:
                    alpha = 0.5
                else:
                    alpha = 1
                ax.plot(real_data, fake_data, 'o', markersize=4, mew=0.5, alpha = alpha, label = model_name, color=color_lis[c])
                c+=1

                max_val = max(max_val, fake_data[-1])
                min_val = min(min_val, fake_data[0])
            
            ax.set_xlabel('Quantiles of Real Data')
            ax.set_ylabel('Quantiles of Generated Data')
            if exp_name == "pareto":
                ax.set_title('Q-Q Plot on dimension '+str(dim+11))
            elif exp_name == "queue":
                ax.set_title('Q-Q Plot on station '+str(dim+1))
            elif exp_name == "vectorAR":
                ax.set_title('Q-Q Plot on dimension '+str(dim+1)+ f', for t={target_L[L]+1}')
            else:
                ax.set_title('Q-Q Plot')
            

            # Adding a 45-degree line
            ax.plot([min_val, max_val], [min_val, max_val], 'r--')
            # ax.vlines(x=[real_data[int(len(fake_data)*0.9)], real_data[int(len(fake_data)*0.95)], real_data[int(len(fake_data)*0.99)], real_data[int(len(fake_data)*0.995)]], ymin=min_val, ymax=max_val, color='orange', linestyle='-.')
            ax.legend(fontsize='x-small')
            

    # plt.tight_layout()  # Adjusts subplot params so that subplots are nicely fit in the figure area
    plt.subplots_adjust(hspace=0.35)
    if save_fig:
        plt.savefig(f'./save/{datafolder}/{exp_name}_QQ_plot_{len(target_dim)*len(target_L)}.png')
    plt.show()


def plot_log_tail_distn(model_list, fake_samples, real_samples, datafolder, exp_name, target_dim = [0,1,2,3,4], target_L = [1], color_lis = ['orange', 'yellowgreen','cornflowerblue'], shape = None, save_fig = True, drop_percentile = 0.5):
    # Create a figure and an array of subplots
    # plt.rcParams["font.size"] = 16
    if shape is None:
        fig, axes = plt.subplots(len(target_L), len(target_dim), figsize = (4*len(target_dim), 4*len(target_L)))  # Adjust the figsize to fit your screen
        axes = np.array(axes).reshape(len(target_L), len(target_dim))
    else:
        fig, axes = plt.subplots(shape[0], shape[1], figsize = (4*shape[1], 4*shape[0]))  # Adjust the figsize to fit your screen
        axes = np.array(axes).reshape(shape[0], shape[1])
    loc_row = 0
    loc_col = -1
    for L in range(len(target_L)):
        for K in range(len(target_dim)):
            if shape is None:
                loc_row = L
                loc_col = K
            else:
                if loc_col<shape[1]-1:
                    loc_col+=1
                else:
                    loc_col=0
                    loc_row+=1
            ax = axes[loc_row,loc_col]
            time = target_L[L]
            dim = target_dim[K]

            real_data = real_samples[:,time,dim].squeeze().copy()
            real_data.sort()
            real_data = drop_top_percentile(real_data, drop_percentile)
            real_kde = gaussian_kde(real_data)

            x = np.linspace(min(real_data), max(real_data), 1000)
            real_cdf = np.array([real_kde.integrate_box_1d(-np.inf, v) for v in x])
            real_log_cdf_complement = np.log(1 - real_cdf)
            ax.plot(x, real_log_cdf_complement, label='real data', color = color_lis[0])

            c=1
            for model_name in fake_samples.keys():
                if model_name not in model_list:
                    continue
                fake_data = fake_samples[model_name][:,time,dim].squeeze().copy()
                fake_data.sort()
                fake_data = drop_top_percentile(fake_data, drop_percentile)
                fake_kde = gaussian_kde(fake_data)

                x = np.linspace(max(min(real_data),min(fake_data)), min(max(real_data),max(fake_data)), 1000)
                fake_cdf = np.array([fake_kde.integrate_box_1d(-np.inf, v) for v in x])
                fake_log_cdf_complement = np.log(1 - fake_cdf)
                ax.plot(x, fake_log_cdf_complement, label=model_name, color=color_lis[c])
                c+=1


            ax.set_xlabel('x')
            ax.set_ylabel('log(1-CDF(x))')
            if exp_name == "pareto":
                ax.set_title('Plot of log(1-CDF(x)) on dimension '+str(dim+11))
            elif exp_name == "queue":
                ax.set_title('Plot of log(1-CDF(x)) on station '+str(dim+1))
            ax.legend(fontsize="x-small")
            ax.grid(True)

    # plt.tight_layout()  # Adjusts subplot params so that subplots are nicely fit in the figure area
    plt.subplots_adjust(hspace=0.35)
    if save_fig:
        plt.savefig(f'./save/{datafolder}/{exp_name}_log_tail_distn_{len(target_dim)}.png')
    plt.show()


def calc_and_print_VAR(fake_samples, real_samples, datafolder, level_list=[0.95, 0.99, 0.995], target_dim = [0,1,2,3,4], target_L = [1]):
    def value_at_risk(data, level):
        kde = gaussian_kde(data)

        # compute the CDF using the KDE
        def cdf_kde(kde, x):
            return kde.integrate_box_1d(-np.inf, x)

        x_vals = np.linspace(min(data), max(data), 10000)
        cdf_vals = np.array([cdf_kde(kde, x) for x in x_vals])
        x_level = x_vals[np.argmin(np.abs(cdf_vals - level))]
        return x_level
    
    VAR_dict = {"real":{}}
    for level in level_list:
        VAR_dict["real"][level] = []

        for L in range(len(target_L)):
            for K in range(len(target_dim)):
                dim = target_dim[K]
                time = target_L[L]
                real_data = abs(real_samples[:,time,dim].squeeze().copy())
                real_data.sort()
                # var = value_at_risk(real_data, level)
                var = real_data[int(level*len(real_data))]

                if K == 0:
                    VAR_dict["real"][level].append([var])
                else:
                    VAR_dict["real"][level][L].append(var)
        VAR_dict["real"][level] = np.array(VAR_dict["real"][level])

    for model_name in fake_samples.keys():
        VAR_dict[model_name] = {}
        VAR_dict[model_name+"_VAR_mean_abs_err"] = {}
        for level in level_list:
            VAR_dict[model_name][level] = []

            for L in range(len(target_L)):
                for K in range(len(target_dim)):
                    dim = target_dim[K]
                    time = target_L[L]
                    fake_data = abs(fake_samples[model_name][:,time,dim].squeeze().copy())
                    fake_data.sort()
                    # var = value_at_risk(fake_data, level)
                    var = fake_data[int(level*len(fake_data))]

                    if K == 0:
                        VAR_dict[model_name][level].append([var])
                    else:
                        VAR_dict[model_name][level][L].append(var)
            VAR_dict[model_name][level] = VAR_dict[model_name][level]
            VAR_dict[model_name+"_VAR_mean_abs_err"][level] = np.mean(abs(VAR_dict[model_name][level]/VAR_dict["real"][level]-1))

    return VAR_dict


def calc_and_print_W_dist(fake_samples, real_samples, datafolder, normalize = True, target_dim = [0,1,2,3,4], target_L = [1]):
    
    W_dist_dict = {}
    for model_name in fake_samples.keys():
        W_dist_dict[model_name] = {}
        W_dist_dict[model_name+"_VAR_mean_abs_err"] = {}
        W_dist_dict[model_name] = []

        for L in range(len(target_L)):
            for K in range(len(target_dim)):
                dim = target_dim[K]
                time = target_L[L]
                fake_data = fake_samples[model_name][:,time,dim].squeeze().copy()
                fake_data.sort()
                real_data = real_samples[:,time,dim].squeeze().copy()
                real_data.sort()

                if normalize:
                    real_mean = real_data.mean()
                    real_std = real_data.std()
                    normalized_real = (real_data-real_mean)/real_std
                    normalized_fake = (fake_data-real_mean)/real_std
                    w_dist = wasserstein_distance(normalized_real, normalized_fake)
                else:
                    w_dist = wasserstein_distance(real_data, fake_data)

                if K == 0:
                    W_dist_dict[model_name].append([w_dist])
                else:
                    W_dist_dict[model_name][L].append(w_dist)
        W_dist_dict[model_name] = np.array(W_dist_dict[model_name])
        W_dist_dict[model_name+"_mean_W_dist"] = np.mean(abs(W_dist_dict[model_name]))

    return W_dist_dict
    

            
