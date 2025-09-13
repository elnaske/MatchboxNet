from torch import nn

class MatchboxNet(nn.Module):
    def __init__(self, in_channels, n_classes, B=3, S=2):
        super().__init__()

        # Prolog
        self.C1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels,
                      out_channels=128,
                      kernel_size=11,
                      stride=2
                      ),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        # Blocks
        self.B = nn.ModuleList()
        self.S = S
        self.resB = nn.ModuleList()

        for i in range(B):
            n_channels = 128 if i == 0 else 64 # Accomodate output shape of C1

            # Residual Branch
            self.resB.append(
                nn.Sequential(
                    # Pointwise Convolution Only
                    nn.Conv1d(
                        in_channels=n_channels,
                        out_channels=64,
                        kernel_size=1
                    ),
                    nn.BatchNorm1d(64)
                )
            )

            # Block B_i
            block_layers = nn.ModuleList()

            for j in range(S):
                if j > 0:
                    n_channels = 64 # Match output size of first layer

                block_layers.append(
                    nn.Sequential(
                        # Depthwise Convolution (in_channels = out_channels = groups)
                        nn.Conv1d(
                            in_channels=n_channels,
                            out_channels=n_channels,
                            kernel_size=13 + 2 * i,
                            padding='same',
                            groups=n_channels,
                        ),
                        # Pointwise Convolution
                        nn.Conv1d(
                            in_channels=n_channels,
                            out_channels=64,
                            kernel_size=1
                        ),
                        nn.BatchNorm1d(64)
                    )
                )

            self.B.append(block_layers)

        # Epilog
        self.C2 = nn.Sequential(
            nn.Conv1d(in_channels=64,
                      out_channels=128,
                      kernel_size=29,
                      dilation=2
                      ),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.C3 = nn.Sequential(
            nn.Conv1d(in_channels=128,
                      out_channels=128,
                      kernel_size=1,
                      ),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.C4 = nn.Sequential(
            nn.Conv1d(in_channels=128,
                      out_channels=n_classes,
                      kernel_size=1,
                      )
        )

        self.ReLU = nn.ReLU()
        self.Dropout = nn.Dropout1d(p=0.1)

        self.Pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)

        # Prolog
        x = self.C1(x)

        # Blocks
        # for i in range(len(self.B)):
        #     res = self.resB[i](x)
        #     for j, layer in enumerate(self.B[i]):
        #         x = layer(x) # Convolution + Batch Norm
        #         if j + 1 == self.S: # Add residual before ReLU & Dropout on last layer
        #             x += res
        #         x = self.Dropout(self.ReLU(x))
        #         x = self.ReLU(x)

        for res_block, block in zip(self.resB, self.B):
            res = res_block(x)
            for j, layer in enumerate(block):
                x = layer(x) # Convolution + Batch Norm
                if j + 1 == self.S: # Add residual before ReLU & Dropout on last layer
                    x += res
                x = self.Dropout(self.ReLU(x))
                x = self.ReLU(x)

        # Epilog
        x = self.C2(x)
        x = self.C3(x)
        x = self.C4(x)

        return self.Pool(x).squeeze()