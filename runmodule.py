import torch
from torch.utils.data import DataLoader

from metric import valid
from datasets import get_dataset, MvDataset
from network import Network
from loss import Loss
from util import getMvKNNGraph
from alignment import get_alignment


class RunModule:
    def __init__(self, config, device):
        self.cfg = config
        self.device = device

        # get data
        self.dataset_aligned, self.dataset_shuffle, self.aligned_idx = get_dataset(self.cfg, self.device)

        self.model_path = "./pretrain/" + self.cfg['Dataset']["name"]

    def pretrain_ae(self):
        # get param
        num_views = self.cfg['Dataset']['num_views']
        epochs = self.cfg['training']['mse_epochs']
        lambda_graph = self.cfg['training']['lambda_graph']

        # get model
        model = Network(num_views,
                        self.cfg['Module']['in_dim'],
                        self.cfg['Module']['feature_dim'],
                        self.cfg['Module']['high_feature_dim'],
                        self.cfg['Dataset']['num_classes'], self.device).to(self.device)
        data_loader = torch.utils.data.DataLoader(
            self.dataset_shuffle,
            batch_size=self.cfg['Dataset']['batch_size'],
            shuffle=True,
            drop_last=True,
        )
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=self.cfg['training']['lr'],
                                     weight_decay=self.cfg['training']['weight_decay'])
        criterion = Loss(self.cfg['Dataset']['batch_size'],
                         self.cfg['Dataset']['num_classes'],
                         self.cfg['training']['temperature_f'],
                         self.cfg['training']['temperature_l'],
                         self.device).to(self.device)

        # training
        all_new_x = [torch.from_numpy(v_data) for v_data in self.dataset_shuffle.fea]
        all_new_z = torch.ones((self.cfg['Dataset']['num_views'], self.cfg['Dataset']['num_sample'],
                                self.cfg['Module']['feature_dim'])).to(self.device)
        all_graph = torch.tensor(getMvKNNGraph(all_new_x, k=self.cfg['training']['knn']), device=self.device,
                                 dtype=torch.float32)
        for epoch in range(epochs):
            for xs, _, _, idx in data_loader:
                _, _, xrs, zs = model(xs)

                loss_list = []
                for v in range(num_views):
                    loss_list.append(torch.nn.MSELoss()(xs[v], xrs[v]))
                loss = sum(loss_list)

                if epoch > 0:
                    loss = loss + lambda_graph * criterion.graph_loss(all_graph[:, idx], zs, all_new_z)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # fill all_new_z
                for v in range(num_views):
                    all_new_z[v][idx] = zs[v].detach().clone()

        # save model
        torch.save({'model': model.state_dict()}, self.model_path + '_ae.pth')

    def pretrain_cl(self):
        # get param
        num_views = self.cfg['Dataset']['num_views']
        num_classes = self.cfg['Dataset']['num_classes']
        epochs = self.cfg['training']['con_epochs']
        lambda_graph = self.cfg['training']['lambda_graph']

        # get model
        model = Network(num_views,
                        self.cfg['Module']['in_dim'],
                        self.cfg['Module']['feature_dim'],
                        self.cfg['Module']['high_feature_dim'],
                        num_classes, self.device).to(self.device)
        model.load_state_dict(torch.load(self.model_path + '_ae.pth')['model'])
        data_loader = torch.utils.data.DataLoader(
            self.dataset_shuffle,
            batch_size=self.cfg['Dataset']['batch_size'],
            shuffle=True,
            drop_last=True,
        )
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=self.cfg['training']['lr'],
                                     weight_decay=self.cfg['training']['weight_decay'])
        criterion = Loss(self.cfg['Dataset']['batch_size'],
                         self.cfg['Dataset']['num_classes'],
                         self.cfg['training']['temperature_f'],
                         self.cfg['training']['temperature_l'],
                         self.device).to(self.device)

        # training
        all_new_x = [torch.from_numpy(v_data) for v_data in self.dataset_shuffle.fea]
        all_new_z = torch.ones((self.cfg['Dataset']['num_views'], self.cfg['Dataset']['num_sample'],
                                self.cfg['Module']['feature_dim'])).to(self.device)
        all_graph = torch.tensor(getMvKNNGraph(all_new_x, k=self.cfg['training']['knn']), device=self.device,
                                 dtype=torch.float32)
        for epoch in range(epochs):
            for xs, _, aligned_idx, idx in data_loader:
                hs, qs, xrs, zs = model(xs)

                loss_list = []
                for v in range(num_views):
                    for w in range(v + 1, num_views):
                        loss_list.append(criterion.forward_feature(hs[v][aligned_idx == 1], hs[w][aligned_idx == 1]))
                        loss_list.append(criterion.forward_label(qs[v][aligned_idx == 1], qs[w][aligned_idx == 1]))
                    loss_list.append(torch.nn.MSELoss()(xs[v], xrs[v]))
                loss = sum(loss_list)

                if epoch > 0:
                    loss = loss + lambda_graph * criterion.graph_loss(all_graph[:, idx], zs, all_new_z)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # fill all_new_z
                for v in range(num_views):
                    all_new_z[v][idx] = zs[v].detach().clone()

            # evaluate
            if epoch % 5 == 0:
                print("epoch: " + str(epoch))
                valid(model, self.device, self.dataset_aligned, num_views, self.cfg['Dataset']['num_sample'], num_classes,
                      eval_h=False)

        # save model
        torch.save({'model': model.state_dict()}, self.model_path + '_cl.pth')

    def train1(self):
        # get param
        num_views = self.cfg['Dataset']['num_views']
        num_classes = self.cfg['Dataset']['num_classes']
        epochs = self.cfg['training']['tune_epochs']
        lambda_graph = self.cfg['training']['lambda_graph']

        # get model
        model = Network(num_views,
                        self.cfg['Module']['in_dim'],
                        self.cfg['Module']['feature_dim'],
                        self.cfg['Module']['high_feature_dim'],
                        num_classes, self.device).to(self.device)
        model.load_state_dict(torch.load(self.model_path + '_cl.pth')['model'])

        # valid aligned data and get info
        print("valid on aligned data")
        _, _, _, pre_labels, hs, labels, qs, zs = valid(model, self.device, self.dataset_aligned, num_views,
                                                        self.cfg['Dataset']['num_sample'], num_classes, eval_h=False)

        # performer alignment
        fea_realigned, labels_realigned, realigned_idx = get_alignment(self.dataset_aligned.fea, hs, qs, zs, pre_labels,
                                                                       self.dataset_aligned.labels, self.aligned_idx,
                                                                       self.device)
        dataset_realigned = MvDataset(fea_realigned, labels_realigned, realigned_idx, self.device)
        print("valid on realigned data")
        valid(model, self.device, dataset_realigned, num_views, self.cfg['Dataset']['num_sample'], num_classes,
              eval_h=False)

        data_loader = torch.utils.data.DataLoader(
            dataset_realigned,
            batch_size=self.cfg['Dataset']['batch_size'],
            shuffle=True,
            drop_last=True,
        )
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=self.cfg['training']['lr'],
                                     weight_decay=self.cfg['training']['weight_decay'])
        criterion = Loss(self.cfg['Dataset']['batch_size'],
                         self.cfg['Dataset']['num_classes'],
                         self.cfg['training']['temperature_f'],
                         self.cfg['training']['temperature_l'],
                         self.device).to(self.device)

        # training
        all_new_x = [torch.from_numpy(v_data) for v_data in dataset_realigned.fea]
        all_new_z = torch.ones((self.cfg['Dataset']['num_views'], self.cfg['Dataset']['num_sample'],
                                self.cfg['Module']['feature_dim'])).to(self.device)
        all_graph = torch.tensor(getMvKNNGraph(all_new_x, k=self.cfg['training']['knn']), device=self.device,
                                 dtype=torch.float32)
        for epoch in range(epochs):
            for xs, _, aligned_idx, idx in data_loader:
                hs, qs, xrs, zs = model(xs)

                loss_list = []
                for v in range(num_views):
                    for w in range(v + 1, num_views):
                        loss_list.append(criterion.forward_feature(hs[v][aligned_idx == 1], hs[w][aligned_idx == 1]))
                        loss_list.append(criterion.forward_label(qs[v][aligned_idx == 1], qs[w][aligned_idx == 1]))
                    loss_list.append(torch.nn.MSELoss()(xs[v], xrs[v]))
                loss = sum(loss_list)

                if epoch > 0:
                    loss = loss + lambda_graph * criterion.graph_loss(all_graph[:, idx], zs, all_new_z)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # fill all_new_z
                for v in range(num_views):
                    all_new_z[v][idx] = zs[v].detach().clone()

            # evaluate
            if epoch % 5 == 0:
                print("epoch: " + str(epoch))
                _, _, _, pre_labels, hs, labels, qs, zs = valid(model, self.device, dataset_realigned, num_views,
                                                                self.cfg['Dataset']['num_sample'], num_classes,
                                                                eval_h=False)

                fea_realigned, labels_realigned, realigned_idx = get_alignment(dataset_realigned.fea, hs, qs, zs,
                                                                               pre_labels,
                                                                               dataset_realigned.labels,
                                                                               dataset_realigned.aligned_idx,
                                                                               self.device)
                dataset_realigned = MvDataset(fea_realigned, labels_realigned, realigned_idx, self.device)
                print("valid on realigned data")
                valid(model, self.device, dataset_realigned, num_views, self.cfg['Dataset']['num_sample'], num_classes,
                      eval_con=False)

                data_loader = torch.utils.data.DataLoader(
                    dataset_realigned,
                    batch_size=self.cfg['Dataset']['batch_size'],
                    shuffle=True,
                    drop_last=True,
                )

                all_new_x = [torch.from_numpy(v_data) for v_data in dataset_realigned.fea]
                all_graph = torch.tensor(getMvKNNGraph(all_new_x, k=self.cfg['training']['knn']), device=self.device,
                                         dtype=torch.float32)
