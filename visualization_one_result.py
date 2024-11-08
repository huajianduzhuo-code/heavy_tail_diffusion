import numpy as np
import seaborn
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.stats import wasserstein_distance


def drop_top_percentile(arr, percentile=0.5):
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


def plot_QQ(config, all_target, samples, datafolder, drop_percentile = 0.5):
    # Create a figure and an array of subplots
    fig, axes = plt.subplots(config["data"]["target_L"], config["data"]["target_dim"], figsize = (4*config["data"]["target_dim"], 4*config["data"]["target_L"]))  # Adjust the figsize to fit your screen
    axes = np.array(axes).reshape(config["data"]["target_L"], config["data"]["target_dim"])
    for L in range(config["data"]["target_L"]):
        for dim in range(config["data"]["target_dim"]):
            ax = axes[L,dim]
            real_data = all_target[:,config["data"]["condition_L"]+L,dim].squeeze().cpu().numpy().copy()
            fake_data = samples[:,config["data"]["condition_L"]+L,dim].squeeze().cpu().numpy().copy()
            
            real_data.sort()
            fake_data.sort()

            real_data = drop_top_percentile(real_data, drop_percentile)
            fake_data = drop_top_percentile(fake_data, drop_percentile)

            ax.plot(real_data, fake_data, 'o', markersize=5)
            ax.set_xlabel('Quantiles of Real Data')
            ax.set_ylabel('Quantiles of Generated Data')
            ax.set_title('Q-Q Plot on dimension '+str(dim+1))

            # Adding a 45-degree line
            max_val = max(real_data[-1], fake_data[-1])
            min_val = min(real_data[0], fake_data[0])
            ax.plot([min_val, max_val], [min_val, max_val], 'r--')

    plt.savefig('./save/'+datafolder+'/visualize/QQ_plot.png')
    plt.tight_layout()  # Adjusts subplot params so that subplots are nicely fit in the figure area
    plt.show()


def plot_log_tail_distn(config, all_target, samples, datafolder):
    # Create a figure and an array of subplots with 1 row and 5 columns
    fig, axes = plt.subplots(config["data"]["target_L"], config["data"]["target_dim"], figsize = (4*config["data"]["target_dim"], 4*config["data"]["target_L"]))  # Adjust the figsize to fit your screen
    axes = np.array(axes).reshape(config["data"]["target_L"], config["data"]["target_dim"])
    for L in range(config["data"]["target_L"]):
        for dim in range(config["data"]["target_dim"]):
            ax = axes[L,dim]
            real_data = all_target[:,config["data"]["condition_L"]+L,dim].squeeze().cpu().numpy().copy()
            # real_noised_data = all_noised_target[:,config["data"]["condition_L"]+L,dim].squeeze().cpu().numpy().copy()
            fake_data = samples[:,config["data"]["condition_L"]+L,dim].squeeze().cpu().numpy().copy()

            real_data = real_data[:int(len(real_data)*0.995)]
            fake_data = fake_data[:int(len(fake_data)*0.995)]

            # Perform Kernel Density Estimation
            real_kde = gaussian_kde(real_data)
            fake_kde = gaussian_kde(fake_data)

            x = np.linspace(min(np.concatenate((real_data,fake_data))), max(np.concatenate((real_data,fake_data))), 1000)

            real_cdf = np.array([real_kde.integrate_box_1d(-np.inf, v) for v in x])
            fake_cdf = np.array([fake_kde.integrate_box_1d(-np.inf, v) for v in x])

            real_log_cdf_complement = np.log(1 - real_cdf)
            fake_log_cdf_complement = np.log(1 - fake_cdf)

            # Plot
            ax.plot(x, real_log_cdf_complement, label='real data')
            # ax.plot(x, real_noised_log_cdf_complement, label='noised real data')
            ax.plot(x, fake_log_cdf_complement, label='fake data')
            ax.set_xlabel('x')
            ax.set_ylabel('log(1-CDF(x))')
            ax.set_title('Plot of log(1-CDF(x)) using KDE')
            ax.legend()
            ax.grid(True)

    plt.savefig('./save/'+datafolder+'/visualize/log_tail_distn.png')
    plt.tight_layout()  # Adjusts subplot params so that subplots are nicely fit in the figure area
    plt.show()


def calc_and_print_VAR(config, all_target, samples, datafolder, level_list = [0.99, 0.995, 0.999]):

    # Example dataset (replace with your actual data)
    def value_at_risk(data, level):
        kde = gaussian_kde(data)

        # compute the CDF using the KDE
        def cdf_kde(kde, x):
            return kde.integrate_box_1d(-np.inf, x)

        x_vals = np.linspace(min(data), max(data), 10000)
        cdf_vals = np.array([cdf_kde(kde, x) for x in x_vals])
        x_level = x_vals[np.argmin(np.abs(cdf_vals - level))]
        return x_level
    VAR_err = {}
    for level in level_list:
        VAR_err[level]=0
    for L in range(config["data"]["target_L"]):
        for dim in range(config["data"]["target_dim"]):
            real_data = all_target[:,config["data"]["condition_L"]+L,dim].squeeze().cpu().numpy().copy()
            real_data.sort()
            fake_data = samples[:,config["data"]["condition_L"]+L,dim].squeeze().cpu().numpy().copy()
            fake_data.sort()
            print_and_save("t = "+str(L+1)+", dim = "+str(dim+1), './save/'+datafolder+'/visualize/VAR_new.txt')
            for level in level_list:
                # print_and_save(f"{round((1-level)*100, 2)}% value at risk of real data is {value_at_risk(real_data, level)}", './save/'+datafolder+'/visualize/VAR.txt')
                print_and_save(f"{round((1-level)*100, 2)}% value at risk of real data 2 is {real_data[int(len(real_data)*level)]}", './save/'+datafolder+'/visualize/VAR_new.txt')
                # print_and_save(f"{round((1-level)*100, 2)}% value at risk of fake data is {value_at_risk(fake_data, level)}", './save/'+datafolder+'/visualize/VAR.txt')
                print_and_save(f"{round((1-level)*100, 2)}% value at risk of fake data 2 is {fake_data[int(len(fake_data)*level)]}", './save/'+datafolder+'/visualize/VAR_new.txt')
                VAR_err[level] +=abs(fake_data[int(len(fake_data)*level)]/real_data[int(len(real_data)*level)]-1)
    for level in level_list:
        VAR_err[level]=VAR_err[level]/(config["data"]["target_L"]*config["data"]["target_dim"])
    return VAR_err


def calc_and_print_W_dist(fake_samples, real_samples, datafolder, normalize = True, target_dim = [0,1,2,3,4], target_L = [1]):

    W_dist_list = []
    for L in range(len(target_L)):
        for K in range(len(target_dim)):
            dim = target_dim[K]
            time = target_L[L]
            fake_data = fake_samples[:,time,dim].squeeze().copy()
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
                    W_dist_list.append([w_dist])
            else:
                W_dist_list[L].append(w_dist)
    W_dist_list = np.array(W_dist_list)
    mean_W_dist = np.mean(abs(W_dist_list))

    return mean_W_dist, W_dist_list