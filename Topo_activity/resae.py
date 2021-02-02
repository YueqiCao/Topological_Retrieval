from numpy import prod
import torch.nn as nn

class ResNetAEEncoder(nn.Module):

    def __init__(self, n_f=32, n_ResidualBlock=3, n_levels=3, z_dim=10, output_channels=1, **kwargs):
        super(ResNetAEEncoder, self).__init__()

        self.n_f = n_f

        self.n_ResidualBlock = n_ResidualBlock
        self.n_levels = n_levels
        self.max_filters = 2 ** (n_levels + 3)
        self.z_dim = z_dim
        self.output_channels = output_channels

        in_filters = 8
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv1 = nn.Conv2d(1, out_channels=in_filters, kernel_size=(3, 3), stride=(1, 1),
                            padding=1)
        #### ENCODER
        self.convs = nn.ModuleList([])
        self.outconvs = nn.ModuleList([])
        self.residuals = nn.ModuleList([])
        self.skip_convs = nn.ModuleList([])
        for i in range(self.n_levels):
            n_filters_1 = 2 ** (i + 3)
            n_filters_2 = 2 ** (i + 4)
            ks = 2 ** (self.n_levels - i)
            conv = nn.Conv2d(in_filters, out_channels=self.max_filters, kernel_size=3, stride=ks, padding=1)
            self.skip_convs.append(conv)
            res = nn.ModuleList([])
            for _ in range(self.n_ResidualBlock):
                res.append(self.ResidualBlock(in_filters, n_filters_1, 3, 1))
            self.residuals.append(res)

            self.outconvs.append(nn.Conv2d(in_filters, out_channels=n_filters_2, kernel_size=3, stride=2, padding=1))
            in_filters = n_filters_2
        self.out_conv = nn.Conv2d(self.max_filters, out_channels=self.z_dim, kernel_size=(3, 3), stride=1, padding=1)

    def ResidualBlock(self, input_size, filters=64, kernel_size=(3, 3), strides=(1, 1)):
        long_path = nn.Sequential(nn.BatchNorm2d(filters),
                                  nn.ReLU(),
                                  nn.Conv2d(input_size, filters, kernel_size, strides, 1),
                                  nn.BatchNorm2d(filters),
                                  nn.ReLU(),
                                  nn.Conv2d(input_size, filters, kernel_size, strides, 1)
                                  )
        return long_path

    def RunResidual(self, x, filters=64, kernel_size=(3, 3), strides=(1, 1)):
        longpath = self.ResidualBlock(x.shape[1], filters, kernel_size, strides)

        x_long = longpath(x)
        x = x + x_long
        return x

    def forward(self,inputs):
        x = self.conv1(inputs)
        x = self.relu(x)
        skips = []

        for i in range(self.n_levels):

            y = self.skip_convs[i](x)
            skips.append(self.relu(y))
            for j in range(self.n_ResidualBlock):
                long = self.residuals[i][j](x)
                x = x + long
            x = self.outconvs[i](x)
            x = self.relu(x)

        x = sum([x] + skips)

        x = self.out_conv(x)
        encoded = self.sigmoid(x)

        return encoded


class ResNetAEDecoder(nn.Module):

    def __init__(self, n_f=32, n_ResidualBlock=3, n_levels=3, z_dim=10, output_channels=1, **kwargs):
        super(ResNetAEDecoder, self).__init__()
        self.n_f = n_f

        self.n_ResidualBlock = n_ResidualBlock
        self.n_levels = n_levels
        self.max_filters = 2 ** (n_levels + 3)
        self.z_dim = z_dim
        self.output_channels = output_channels

        in_filters = 8

        self.relu = nn.LeakyReLU(inplace=True)

        self.decoder_conv1 = nn.Conv2d(self.z_dim, out_channels=self.max_filters, kernel_size=(3, 3), stride=1,
                                       padding=1)
        in_filters = in_filters_2 = self.max_filters
        self.decoder_convs = nn.ModuleList([])
        self.decoder_res = nn.ModuleList([])
        self.dec_outconvs = nn.ModuleList([])
        for i in range(self.n_levels):
            n_filters = 2 ** (self.n_levels - i + 2)
            ks = 2 ** (i + 1)
            self.decoder_convs.append(nn.ConvTranspose2d(in_filters, n_filters, kernel_size=2, stride=2, padding=0))
            dec_res = nn.ModuleList([])
            for _ in range(self.n_ResidualBlock):
                dec_res.append(self.ResidualBlock(n_filters, n_filters, 3, 1))
            self.decoder_res.append(dec_res)
            self.dec_outconvs.append(nn.ConvTranspose2d(in_filters_2, n_filters, kernel_size=ks, stride=ks, padding=0))
            in_filters = n_filters

        self.final_layer = nn.Conv2d(in_filters, self.output_channels, kernel_size=(3, 3), stride=1, padding=1)

        self.projector = nn.Sequential(
            nn.Linear(100, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, n_f * 31 + 3),
            nn.LeakyReLU(inplace=True)
        )

    def ResidualBlock(self, input_size, filters=64, kernel_size=(3, 3), strides=(1, 1)):
        long_path = nn.Sequential(nn.BatchNorm2d(filters),
                                  nn.ReLU(),
                                  nn.Conv2d(input_size, filters, kernel_size, strides, 1),
                                  nn.BatchNorm2d(filters),
                                  nn.ReLU(),
                                  nn.Conv2d(input_size, filters, kernel_size, strides, 1)
                                  )
        return long_path

    def RunResidual(self, x, filters=64, kernel_size=(3, 3), strides=(1, 1)):
        longpath = self.ResidualBlock(x.shape[1], filters, kernel_size, strides)

        x_long = longpath(x)
        x = x + x_long
        return x

    def forward(self, encoded):

        encoded = self.decoder_conv1(encoded)
        encoded = z_top = self.relu(encoded)

        for i in range(self.n_levels):

            encoded = self.decoder_convs[i](encoded)
            encoded = self.relu(encoded)

            for j in range(self.n_ResidualBlock):
                encoded = self.decoder_res[i][j](encoded)

            temp = self.dec_outconvs[i](z_top)
            encoded += self.relu(temp)

        decoded = self.final_layer(encoded)
        decoded = self.relu(decoded)

        return decoded


class ResAutoEncoder(nn.Module):
    def __init__(self, n_f=32, n_ResidualBlock=3, n_levels=3, z_dim=10, output_channels=1, **kwargs):
        super(ResAutoEncoder, self).__init__()

        self.n_f = n_f

        self.n_ResidualBlock = n_ResidualBlock
        self.n_levels = n_levels
        self.z_dim = z_dim
        self.output_channels = output_channels

        self.enc = ResNetAEEncoder(n_f=self.n_f,
                                   n_ResidualBlock=self.n_ResidualBlock,
                                   n_levels=self.n_levels,
                                   z_dim=self.z_dim,
                                   output_channels=self.output_channels)

        self.dec = ResNetAEDecoder(n_f=self.n_f,
                                   n_ResidualBlock=self.n_ResidualBlock,
                                   n_levels=self.n_levels,
                                   z_dim=self.z_dim,
                                   output_channels=self.output_channels)


    def forward(self, input):

        enc_x = self.enc(input)
        # print(enc_x.shape)
        output = self.dec(enc_x)
        return output


def extra_hidden_layer(hidden_dim, non_lin):
    return nn.Sequential(nn.Linear(hidden_dim, hidden_dim), non_lin)

class AutoEncoder(nn.Module):
    def __init__(self, n_f=32, n_ResidualBlock=3, n_levels=3, z_dim=10, output_channels=1, **kwargs):
        super(AutoEncoder, self).__init__()

        self.n_f = n_f
        self.data_size = kwargs.get('data_size',(64,64))
        self.n_ResidualBlock = n_ResidualBlock
        self.n_levels = n_levels
        self.z_dim = z_dim
        self.output_channels = output_channels
        non_lin = nn.ReLU()
        modules = []
        modules.append(nn.Sequential(nn.Linear(prod(self.data_size),n_f)))
        modules.extend([extra_hidden_layer(n_f, non_lin) for _ in range(n_levels - 1)])
        modules.append(nn.Sequential(nn.Linear(n_f, z_dim),non_lin))

        self.enc = nn.Sequential(*modules)

        modules = []
        modules.append(nn.Sequential(nn.Linear(z_dim, n_f),non_lin))
        modules.extend([extra_hidden_layer(n_f, non_lin) for _ in range(n_levels - 1)])
        modules.append(nn.Sequential(nn.Linear(n_f,prod(self.data_size))))
        self.dec = nn.Sequential(*modules)


    def forward(self, input):
        input = input.view(-1,prod(self.data_size))
        enc_x = self.enc(input)
        # print(enc_x.shape)
        output = self.dec(enc_x)
        output = output.view(-1,self.output_channels,self.data_size[1],self.data_size[0])
        return output