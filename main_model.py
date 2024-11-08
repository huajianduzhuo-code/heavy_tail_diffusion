import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
from diff_models import diff_CSDI

def generate_noise(observed_data, device, noise_type = "gaussian", t_param = None):
    if noise_type == "gaussian":
        return torch.randn_like(observed_data)
    elif noise_type == "student_t":
        return torch.tensor(np.random.standard_t(t_param, observed_data.shape)/np.sqrt(t_param/(t_param-2))).float().to(device) # devide by standard deviation to normalize
    else:
        raise ValueError(f"{noise_type} is not a supported noise type")


def calc_sigma_scale(x, use_sigma_scale, sigma_scale_max):
    if use_sigma_scale:
        return (2*(sigma_scale_max-1)/(1+torch.exp(-1/5*x)) + 2 - sigma_scale_max)
    else:
        return torch.ones(x.shape)


class Model_base(nn.Module):
    def __init__(self, target_dim, config, device):
        super().__init__()
        self.config = config
        self.device = device
        self.target_dim = target_dim

        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.is_unconditional = config["model"]["is_unconditional"]
        self.target_strategy = config["model"]["target_strategy"]

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        if self.is_unconditional == False:
            self.emb_total_dim += 1  # for conditional mask
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )

        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim

        input_dim = 1 if self.is_unconditional == True else 2
        self.diffmodel = diff_CSDI(config_diff, input_dim)
        self.noise_type = config_diff["noise_type"]
        self.t_param = config_diff["t_param"]

        # parameters for diffusion models NOTE: begin with big noise, and end with small noise
        self.num_steps = config_diff["num_steps"]
        self.step_lr = config_diff["step_lr"]
        self.n_steps_each = config_diff["n_steps_each"]
        self.anneal_power = config_diff["anneal_power"]
        self.sigmas = torch.tensor(
            np.exp(np.linspace(np.log(config_diff["sigma_begin"]), np.log(config_diff["sigma_end"]),
                               self.num_steps))).float().to(self.device) 

    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe


    def get_test_pattern_mask(self, observed_mask, test_pattern_mask):
        return observed_mask * test_pattern_mask


    def calc_loss_valid(
        self, observed_data, cond_mask, observed_mask, side_info, is_train
    ):
        loss_sum = 0
        for t in range(self.num_steps):  # calculate loss for all t
            loss = self.calc_loss(
                observed_data, cond_mask, observed_mask, side_info, is_train, set_t=t
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    def calc_loss(
        self, observed_data, cond_mask, observed_mask, side_info, is_train, set_t=-1
    ):
        B, K, L = observed_data.shape
        if is_train != 1:  # for validation
            labels = torch.tensor([set_t]*B).to(self.device)
        else:
            labels = torch.randint(0, len(self.sigmas), (B,)).to(self.device)

        used_sigmas = self.sigmas[labels].view(B, *([1] * len(observed_data.shape[1:]))) # B, 1, 1

        noise = generate_noise(observed_data, self.device, self.noise_type, self.t_param)
        normalized_observed_data = torch.abs(observed_data-observed_data.mean((0,)))/observed_data.std((0,))
        sigmas_scale = calc_sigma_scale(normalized_observed_data, self.config["diffusion"]["sigma_scale"], self.config["diffusion"]["non_homo_sigma_max_scale"]).float().to(self.device)
        noisy_data = observed_data + noise * used_sigmas * sigmas_scale
        if self.is_unconditional == True:
            target_mask = observed_mask
        else:
            target_mask = observed_mask - cond_mask

        if self.noise_type == "gaussian":
            total_sigma = used_sigmas
            target = - 1 / (total_sigma ** 2) * (noisy_data - observed_data) # score function of conditional distribution
        elif self.noise_type == "student_t":
            total_sigma = (used_sigmas * sigmas_scale)/np.sqrt(self.t_param/(self.t_param-2))
            added_noise = noisy_data - observed_data
            target = - (self.t_param + 1) * added_noise / (total_sigma ** 2 * self.t_param + added_noise**2) # score function of conditional distribution

        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)
        predicted_score = self.diffmodel(total_input, side_info, labels)  # (B,K,L)

        num_eval = target_mask.sum()

        loss_all = 1 / 2. * (((predicted_score - target) * target_mask) ** 2) * total_sigma ** self.anneal_power
        loss = loss_all.sum() / (num_eval if num_eval > 0 else 1)
        return loss

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        if self.is_unconditional == True:
            total_input = noisy_data.unsqueeze(1)  # (B,1,K,L)
        else:
            cond_obs = (cond_mask * observed_data).unsqueeze(1)
            noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
            total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L), here the 2 contains "cond_obs" and "noisy_data"

        return total_input

    def generate(self, observed_data, cond_mask, side_info, n_samples):
        B, K, L = observed_data.shape

        generated_samples = torch.zeros(B, n_samples, K, L).to(self.device)

        for i in range(n_samples):
            
            if self.config["diffusion"]["start_noise"] == "uniform":
                current_sample = torch.rand_like(observed_data)
            elif self.config["diffusion"]["start_noise"] == "gaussian":
                current_sample = torch.randn_like(observed_data)
            elif self.config["diffusion"]["start_noise"] == "student_t":
                current_sample = generate_noise(observed_data, self.device, "student_t", self.config["diffusion"]["start_noise_t_param"])
            print(f"start from {self.config['diffusion']['start_noise']}")

            with torch.no_grad():
                for c, sigma in tqdm(enumerate(self.sigmas), total=len(self.sigmas), desc='annealed Langevin dynamics sampling'):
                    labels = torch.ones(current_sample.shape[0]).to(self.device) * c
                    labels = labels.long()
                    step_size = self.step_lr * (sigma / self.sigmas[-1]) ** 2
                    for s in range(self.n_steps_each):
                        if self.is_unconditional == True:
                            diff_input = current_sample.unsqueeze(1) # (B,1,K,L)
                        else:
                            cond_obs = (cond_mask * observed_data).unsqueeze(1)
                            noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                            diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
                        noise = torch.randn_like(current_sample) * np.sqrt(step_size.cpu() * 2)
                        grad_predicted = self.diffmodel(diff_input, side_info, labels)
                        current_sample = current_sample + step_size * grad_predicted + noise
                        if self.is_unconditional == True:
                            current_sample = cond_mask * observed_data + (1 - cond_mask) * current_sample

            generated_samples[:, i] = current_sample.detach()
        return generated_samples


class Score_Based_Diffusion(Model_base):
    def __init__(self, config, device, target_dim):
        super(Score_Based_Diffusion, self).__init__(target_dim, config, device)
        self.target_dim_base = target_dim
        self.num_sample_features = config["model"]["num_sample_features"]
        self.config = config

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()

        # if normalize method is reflect_normalize, randomly reflect the data in output dimensions
        if self.config["data"]["normalize_method"] == "reflect_normalize":
            output_mask = observed_mask-gt_mask
            # Generate random integers 0 or 1, then map them to {1, -1}, only operate on output dimensions
            random_tensor = (1 - 2*torch.randint(0, 2, size=observed_data.shape).to(self.device) * output_mask)
            observed_data = observed_data*random_tensor
        elif self.config["data"]["normalize_method"] != "normalize":
            raise ValueError("not valid normalization")
        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        feature_id=torch.arange(self.target_dim_base).unsqueeze(0).expand(observed_data.shape[0],-1).to(self.device)

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
            feature_id, 
        )        

    def sample_features(self,observed_data, observed_mask,feature_id,gt_mask):
        size = self.num_sample_features
        self.target_dim = size
        extracted_data = []
        extracted_mask = []
        extracted_feature_id = []
        extracted_gt_mask = []
        
        for k in range(len(observed_data)):
            ind = np.arange(self.target_dim_base)
            np.random.shuffle(ind)
            extracted_data.append(observed_data[k,ind[:size]])
            extracted_mask.append(observed_mask[k,ind[:size]])
            extracted_feature_id.append(feature_id[k,ind[:size]])
            extracted_gt_mask.append(gt_mask[k,ind[:size]])
        extracted_data = torch.stack(extracted_data,0)
        extracted_mask = torch.stack(extracted_mask,0)
        extracted_feature_id = torch.stack(extracted_feature_id,0)
        extracted_gt_mask = torch.stack(extracted_gt_mask,0)
        return extracted_data, extracted_mask,extracted_feature_id, extracted_gt_mask


    def get_side_info(self, observed_tp, cond_mask,feature_id=None):
        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, self.target_dim, -1)
        if self.target_dim == self.target_dim_base:
            feature_embed = self.embed_layer(
                torch.arange(self.target_dim).to(self.device)
            )  # (K,emb)
            feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        else:
            feature_embed = self.embed_layer(feature_id).unsqueeze(1).expand(-1,L,-1,-1)
        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)

        if self.is_unconditional == False:
            side_mask = cond_mask.unsqueeze(1)  # (B,1,K,L)
            side_info = torch.cat([side_info, side_mask], dim=1)

        return side_info

    def forward(self, batch, is_train=1):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            _,
            feature_id, 
        ) = self.process_data(batch)
        if is_train == 1 and (self.target_dim_base > self.num_sample_features):
            observed_data, observed_mask,feature_id,gt_mask = \
                    self.sample_features(observed_data, observed_mask,feature_id,gt_mask)
        else:
            self.target_dim = self.target_dim_base
            feature_id = None

        if is_train == 0:
            cond_mask = gt_mask
        else: #test pattern
            cond_mask = self.get_test_pattern_mask(
                observed_mask, gt_mask
            )

        side_info = self.get_side_info(observed_tp, cond_mask, feature_id)

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(observed_data, cond_mask, observed_mask, side_info, is_train)

    def evaluate(self, batch, n_samples):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            _,
            feature_id, 
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask * (1-gt_mask)

            side_info = self.get_side_info(observed_tp, cond_mask)

            samples = self.generate(observed_data, cond_mask, side_info, n_samples)

        return samples, observed_data, target_mask, observed_mask, observed_tp
