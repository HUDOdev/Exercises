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
        # initialisierung zu 0 bzw 1, da die Faktoren keinen Einfluss haben sollen und dies so in der Formel in Zeile 65 keinen Einfluss haben
        self.theta_mu = nn.Parameter(torch.zeros(num_channels))
        self.theta_sigma = nn.Parameter(torch.ones(num_channels))
        self.running_mean = None
        self.running_var = None
        self.eps = 1e-6
        
    def forward(self, x):
        # hier hat man zwei Modi in einer Forward Funktion. entweder man trainiert das netzwerk, dann updated man die ganze zeit die running werte
        # für den fall des testens nutzt man diese Werte nur noch und überschreibt die nicht mehr
        # deshalb ist unten drunter die formel mit mean und var und nicht mit self.runn..
        if self.training:
            # specify behavior at training time
            mean = x.mean(dim=0)
            var = x.var(dim=0)
            if self.running_mean is None:
                # set the running stats to stats of x
                self.running_mean = mean
                self.running_var = var
                pass
            else:
                # update the running stats by setting them
                # to the weighted sum of 0.9 times the
                # current running stats and 0.1 times the
                # stats of x
                self.running_mean = 0.9 * self.running_mean + 0.1 * mean
                self.running_var = 0.9 * self.running_var + 0.1 * var
                pass
        else:
            if self.running_mean is None:
                # normalized wrt to stats of
                # current batch x
                # der fall fängt ab, falls man nichts trainiert hat und direkt ins testen geht
                # deshalb könnte man entweder default werte setzen oder man schmeisst einfach einen Error.. siehe Video
                raise ValueError('Model must be trained first.')
            else:
                # use running stats for normalization
                mean = self.running_mean
                var = self.running_var
                pass
        x = self.theta_sigma* (x-mean)/torch.sqrt(var+self.eps) + self.theta_mu # Formel nach Folie 10
        return x
    