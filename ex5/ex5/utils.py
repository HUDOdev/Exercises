import torch
import torch.nn as nn

def init_weights(m):
    if isinstance(m, nn.Linear):
        # set the layer's weights as discussed in the lecture
        # gesucht ist die Initialisation nach Folie 6
        # wir müssen 2/ni nehmen weil wir ReLUs als activation layers haben
        # damit berechnet man die Varianz. Da die funktion .normal_ die standardabweichung braucht noch die wurzel
        # weight.normal packt in den weight tensor wieder einen Normalverteilten Datensatz mit der Größe die weight schon vorher hatte (ähnlich zu ones oder zeros)
        with torch.no_grad():
            var = torch.as_tensor(2/m.weight.shape[1])
            std = torch.sqrt(var)
            m.weight.normal_(0, std)
        pass
        

class BatchNorm(nn.Module):
    
    def __init__(self, num_channels):
        super().__init__()
        # set theta_mu and theta_sigma such that the output of
        # forward initially is zero centered and 
        # normalized to variance 1
        self.theta_mu = nn.Parameter(theta_mu)
        self.theta_sigma = nn.Parameter(theta_sigma)
        self.running_mean = None
        self.running_var = None
        self.eps = 1e-6
        
    def forward(self, x):
        if self.training:
            # specify behavior at training time
            if self.running_mean is None:
                # set the running stats to stats of x
                pass
            else:
                # update the running stats by setting them
                # to the weighted sum of 0.9 times the
                # current running stats and 0.1 times the
                # stats of x
                pass
        else:
            if self.running_mean is None:
                # normalized wrt to stats of
                # current batch x
                pass
            else:
                # use running stats for normalization
                pass
    