import torch


class Regularization(torch.nn.Module):
    def __init__(self, model):
        super(Regularization, self).__init__()
        self.model = model
        self.conv_weight_list = []

    def get_weight(self, model):
        conv_weight_list = []
        for name, param in model.named_parameters():
            # only weights in conv layer
            if 'conv' in name and 'weight' in name:
                weight = (name, param)
                conv_weight_list.append(weight)
        return conv_weight_list

    def Filter_regularization_loss(self, conv_weight_list, penalty):
        filter_reg_loss = 0
        # GL1/2
        if penalty == 1:
            for name, w in conv_weight_list:
                ith_filter_reg_loss = torch.sqrt(torch.sum(torch.abs(w), dim=[1, 2, 3]))  # 1 -> 1/2   GL1/2
                filter_reg_loss += torch.sum(ith_filter_reg_loss)

        # GL
        if penalty == 2:
            for name, w in conv_weight_list:
                ith_filter_reg_loss = torch.sqrt(torch.sum(torch.pow(w, 2), dim=[1, 2, 3]))
                filter_reg_loss += torch.sum(ith_filter_reg_loss)

        # SGL1,2
        if penalty == 3:
            for name, w in conv_weight_list:
                ith_filter_reg_loss = torch.pow(torch.sum(self.smooth(w), dim=[1, 2, 3]), 2)
                filter_reg_loss += torch.sum(ith_filter_reg_loss)

        # GL1,2
        if penalty == 4:
            for name, w in conv_weight_list:
                ith_filter_reg_loss = torch.pow(torch.sum(torch.abs(w), dim=[1, 2, 3]), 2)
                filter_reg_loss += torch.sum(ith_filter_reg_loss)
        return filter_reg_loss

    def forward(self, model, penalty, net_name):
        self.conv_weight_list = self.get_weight(model)

        # no conv1.weight in ResNet20 and ResNet50
        if "resnet" in net_name:
            return self.Filter_regularization_loss(self.conv_weight_list[1:], penalty)
        else:
            return self.Filter_regularization_loss(self.conv_weight_list, penalty)

    def smooth(self, input_x):
        a = 0.008
        f1 = -244140.62499999997
        f2 = 93.75
        f3 = 0.003

        t = torch.abs(input_x)
        area = (f1 * torch.pow(t, 4)) + (f2 * torch.pow(t, 2)) + f3
        output = torch.where(t >= a, t, area)
        return output
