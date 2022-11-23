import torch
import torch.nn as nn
class Generator():
    def __init__(self,
                 start_channel_dim,
                 image_channels,
                 latent_size):
        super().__init__(start_channel_dim, image_channels)
        # Transition blockss
        self.latent_size = latent_size
        self.to_rgb_new = WSConv2d(start_channel_dim, self.image_channels, 1, 0)
        self.to_rgb_old = WSConv2d(start_channel_dim, self.image_channels, 1, 0)
        self.core_blocks = nn.Sequential(
            nn.Sequential(
                WSLinear(self.latent_size, self.latent_size*4*4),
                LatentReshape(self.latent_size),
                conv_bn_relu(self.latent_size, self.latent_size, 3, 1)
            ))
        self.new_blocks = nn.Sequential()
        self.upsampling = UpSamplingBlock()
    def conv_module_bn(dim_in, dim_out, kernel_size, padding):
        return nn.Sequential(
            WSConv2d(dim_in, dim_out, kernel_size, padding),
            nn.LeakyReLU(negative_slope=.2)
        )
    def conv_bn_relu(in_dim, out_dim, kernel_size, padding=0):
        return nn.Sequential(
            WSConv2d(in_dim, out_dim, kernel_size, padding),
            nn.LeakyReLU(negative_slope=.2),
            PixelwiseNormalization()
        )
    def extend(self):
        output_dim = self.transition_channels[self.transition_step]
        # Downsampling module
        if self.transition_step == 0:
            core_blocks = nn.Sequential(
                *self.core_blocks.children(),
                UpSamplingBlock()
            )
            self.core_blocks = core_blocks
        else:
            self.core_blocks = nn.Sequential(
                *self.core_blocks.children(),
                self.new_blocks,
                UpSamplingBlock()
            )
        self.to_rgb_old = self.to_rgb_new
        self.to_rgb_new = WSConv2d(output_dim, self.image_channels, 1, 0)

        self.new_blocks = nn.Sequential(
            conv_bn_relu(self.prev_channel_extension, output_dim, 3, 1),
            conv_bn_relu(output_dim, output_dim, 3, 1),
        )
        super().extend()
    def new_parameters(self):
        new_paramters = list(self.new_blocks.parameters()) + list(self.to_rgb_new.parameters())
        return new_paramters
    def forward(self, z):
        x = self.core_blocks(z)
        if self.transition_step == 0:
            x = self.to_rgb_new(x)
            return x
        x_old = self.to_rgb_old(x)
        x_new = self.new_blocks(x)
        x_new = self.to_rgb_new(x_new)

        x = get_transition_value(x_old, x_new, self.transition_value)
        return x
    def device(self):
        return next(self.parameters()).device
    def generate_latent_variable(self, batch_size):
        return torch.randn(batch_size, self.latent_size, device=self.device())

class Discriminator():
    def __init__(self,
                 image_channels,
                 start_channel_dim
                 ):
        super().__init__(start_channel_dim, image_channels)
        self.from_rgb_new = conv_module_bn(image_channels, start_channel_dim, 1, 0)
        self.from_rgb_old = conv_module_bn(image_channels, start_channel_dim, 1, 0)
        self.new_block = nn.Sequential()
        self.core_model = nn.Sequential(
            nn.Sequential(
                MinibatchStdLayer(),
                conv_module_bn(start_channel_dim + 1, start_channel_dim, 3, 1),
                conv_module_bn(start_channel_dim, start_channel_dim, 4, 0),
            )
        )
        self.output_layer = WSLinear(start_channel_dim, 1)
    def extend(self):
        input_dim = self.transition_channels[self.transition_step]
        output_dim = self.prev_channel_extension
        if self.transition_step != 0:
            self.core_model = nn.Sequential(
                self.new_block,
                *self.core_model.children()
            )
        self.from_rgb_old = nn.Sequential(
            nn.AvgPool2d([2, 2]),
            self.from_rgb_new
        )
        self.from_rgb_new = conv_module_bn(self.image_channels, input_dim, 1, 0)
        self.new_block = nn.Sequential(
            conv_module_bn(input_dim, input_dim, 3, 1),
            conv_module_bn(input_dim, output_dim, 3, 1),
            nn.AvgPool2d([2, 2])
        )
        self.new_block = self.new_block
        super().extend()
    def forward(self, x):
        x_old = self.from_rgb_old(x)
        x_new = self.from_rgb_new(x)
        x_new = self.new_block(x_new)
        x = get_transition_value(x_old, x_new, self.transition_value)
        x = self.core_model(x)
        x = x.view(x.shape[0], -1)
        x = self.output_layer(x)
        return x

class PROGAN:
    def convertToHighResolutionImage(src,dest):
        import time
        time.sleep(2)
