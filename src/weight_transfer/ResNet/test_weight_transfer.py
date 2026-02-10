"""
Test weight transfer and duplication functionality between narrow and wide ResNet models.
"""

import pytest
import torch
import torch.nn as nn
import mup
from weight_transfer.ResNet.models import create_ResNet
from weight_transfer.ResNet.data import get_data_loaders_CNN
from weight_transfer.transfer import transfer_weights, transfer_weights_sp, transfer_optimizer
from weight_transfer import DATA_DIR
import weight_transfer.optim as replaced_optim
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Disable torch.compile
torch._dynamo.config.disable = True
# Set higher precision for better numerical accuracy
torch.set_default_dtype(torch.float64)


class TestResNetWeightTransfer:
    """Test without adding noise, wider and narrower ResNets are equivalent."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up data, narrow and wide ResNet models."""
        # data
        self.train_loader, self.val_loader, self.output_dim = get_data_loaders_CNN("CIFAR100", data_dir=DATA_DIR, batch_size=1000)
        # model params
        self.hidden_dim_small = 0.125
        self.hidden_dim_large = 0.5
        self.bias = True
        # optimization params
        self.lr = 0.01
        self.momentum = 0.9
        self.dampening = 0.1
        self.weight_decay = 0.1
        self.loss_fn = nn.CrossEntropyLoss().double()
        self.error_tolerance = 1e-5
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.steps = 200

    def check_layer_outputs(self, narrow_model, wide_model):
        """Compare outputs of two models layer by layer, expanding narrow outputs to match wide shape"""
        narrow_model.eval()
        wide_model.eval()
        x, _ = next(iter(self.val_loader))
        x = x.to(self.device).double()

        max_diffs = []
        with torch.no_grad():
            # Initial conv and bn
            out1 = torch.relu(narrow_model.bn1(narrow_model.conv1(x)))
            out2 = torch.relu(wide_model.bn1(wide_model.conv1(x)))
            channels_ratio = out2.shape[1] // out1.shape[1]
            out1_expanded = out1.repeat_interleave(channels_ratio, dim=1)
            max_diffs.append(torch.max(torch.abs(out1_expanded - out2)).item())

            # Layer 1
            out1 = narrow_model.layer1(out1)
            out2 = wide_model.layer1(out2)
            channels_ratio = out2.shape[1] // out1.shape[1]
            out1_expanded = out1.repeat_interleave(channels_ratio, dim=1)
            max_diffs.append(torch.max(torch.abs(out1_expanded - out2)).item())

            # Layer 2
            out1 = narrow_model.layer2(out1)
            out2 = wide_model.layer2(out2)
            channels_ratio = out2.shape[1] // out1.shape[1]
            out1_expanded = out1.repeat_interleave(channels_ratio, dim=1)
            max_diffs.append(torch.max(torch.abs(out1_expanded - out2)).item())

            # Layer 3
            out1 = narrow_model.layer3(out1)
            out2 = wide_model.layer3(out2)
            channels_ratio = out2.shape[1] // out1.shape[1]
            out1_expanded = out1.repeat_interleave(channels_ratio, dim=1)
            max_diffs.append(torch.max(torch.abs(out1_expanded - out2)).item())

            # Layer 4
            out1 = narrow_model.layer4(out1)
            out2 = wide_model.layer4(out2)
            channels_ratio = out2.shape[1] // out1.shape[1]
            out1_expanded = out1.repeat_interleave(channels_ratio, dim=1)
            max_diffs.append(torch.max(torch.abs(out1_expanded - out2)).item())

            # Global average pooling
            out1 = torch.nn.functional.avg_pool2d(out1, 4)
            out2 = torch.nn.functional.avg_pool2d(out2, 4)
            channels_ratio = out2.shape[1] // out1.shape[1]
            out1_expanded = out1.repeat_interleave(channels_ratio, dim=1)
            max_diffs.append(torch.max(torch.abs(out1_expanded - out2)).item())

            # Flatten
            out1 = out1.view(out1.size(0), -1)
            out2 = out2.view(out2.size(0), -1)
            features_ratio = out2.shape[1] // out1.shape[1]
            out1_expanded = out1.repeat_interleave(features_ratio, dim=1)
            max_diffs.append(torch.max(torch.abs(out1_expanded - out2)).item())

            # Final linear layer
            final1 = narrow_model.linear(out1)
            final2 = wide_model.linear(out2)
            max_diffs.append(torch.max(torch.abs(final1 - final2)).item())

        return max_diffs

    def train_double_precision(self, model, optimizer, epochs=5):
        """Train model with double precision data handling."""
        model.train()
        for epoch in range(epochs):
            train_loss = 0
            for x_batch, y_batch in self.train_loader:
                x_batch, y_batch = x_batch.to(self.device).double(), y_batch.to(self.device)
                optimizer.zero_grad()
                outputs = model(x_batch)
                loss = self.loss_fn(outputs, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(x_batch)
            train_loss /= len(self.train_loader.dataset)
            print(f"Epoch {epoch}, Loss: {train_loss:.6f}")
        return model

    def check_equivalence(self, narrow_model, wide_model, narrow_optimizer, wide_optimizer, sp=False):
        """Test that weight transfer works correctly with provided models."""

        # Initialize wide model to zeros
        for param in wide_model.parameters():
            nn.init.zeros_(param)

        # Train narrower model for 5 epochs to set weights using our double precision trainer
        narrow_model = self.train_double_precision(narrow_model, narrow_optimizer, epochs=5)

        # Transfer weights
        if sp:
            transfer_weights_sp(narrow_model, wide_model)
        else:
            transfer_weights(narrow_model, wide_model)
            transfer_optimizer(narrow_optimizer, wide_optimizer, narrow_model, wide_model)

        # Test equivalence
        max_diffs = self.check_layer_outputs(narrow_model, wide_model)
        if not all(max_diff < self.error_tolerance for max_diff in max_diffs):
            print(f"Max differences too large before training: {max_diffs}")
            raise AssertionError(f"Max differences too large before training.")
        print(f"✓ Hidden outputs match before SGD step with max_diff={max(max_diffs):.17f}.")

        # Train both models and compare hidden outputs after each step
        narrow_model.train()
        wide_model.train()
        loader = iter(self.train_loader)
        for step in range(self.steps):  # check equivalence for specified number of steps
            try:
                x_batch, y_batch = next(loader)
            except StopIteration:
                loader = iter(self.train_loader)
                x_batch, y_batch = next(loader)
            x_batch, y_batch = x_batch.to(self.device).double(), y_batch.to(self.device)

            # Narrow model step
            narrow_optimizer.zero_grad()
            outputs_narrow = narrow_model(x_batch)
            loss_narrow = self.loss_fn(outputs_narrow, y_batch)
            loss_narrow.backward()
            narrow_optimizer.step()

            # Wide model step
            wide_optimizer.zero_grad()
            outputs_wide = wide_model(x_batch)
            loss_wide = self.loss_fn(outputs_wide, y_batch)
            loss_wide.backward()
            wide_optimizer.step()

            # Check equivalence after each batch
            max_diffs = self.check_layer_outputs(narrow_model, wide_model)
            if not sp and not all(max_diff < self.error_tolerance for max_diff in max_diffs):
                print(f"Max differences too large after step {step}: {max_diffs}")
                raise AssertionError(f"Max differences too large after step {step}.")
            elif sp:
                print(f"✓ Hidden outputs do not match after step {step} with max_diff={max(max_diffs):.17f}.")
            else:
                print(f"✓ Hidden outputs match after step {step} with max_diff={max(max_diffs):.17f}.")

    def test_weight_transfer_sgd_sp(self):
        """Test weight transfer with SGD optimizer."""
        print("\n=======Testing SGD=========")
        narrow_model = create_ResNet(self.output_dim, base_width=None, wm=self.hidden_dim_small).to(self.device).double()
        wide_model = create_ResNet(self.output_dim, base_width=None, wm=self.hidden_dim_large).to(self.device).double()

        narrow_optimizer = torch.optim.SGD(narrow_model.parameters(), lr=self.lr)
        wide_optimizer = torch.optim.SGD(wide_model.parameters(), lr=self.lr)
        self.check_equivalence(narrow_model, wide_model, narrow_optimizer, wide_optimizer, sp=True)

    def test_weight_transfer_sgd_momentum_decay(self):
        """Test weight transfer with SGD optimizer with momentum and dampening."""
        print("\n=======Testing SGD with momentum and dampening=========")
        narrow_model = create_ResNet(self.output_dim, base_width=1, wm=self.hidden_dim_small).to(self.device).double()
        wide_model = create_ResNet(self.output_dim, base_width=1, wm=self.hidden_dim_large).to(self.device).double()

        narrow_optimizer = mup.MuSGD(narrow_model.parameters(), lr=self.lr, momentum=self.momentum, dampening=self.dampening)
        wide_optimizer = mup.MuSGD(wide_model.parameters(), lr=self.lr, momentum=self.momentum, dampening=self.dampening)
        self.check_equivalence(narrow_model, wide_model, narrow_optimizer, wide_optimizer)

    def test_weight_transfer_sgd_nesterov(self):
        """Test weight transfer with SGD optimizer with Nesterov momentum."""
        print("\n=======Testing SGD with Nesterov momentum=========")
        narrow_model = create_ResNet(self.output_dim, base_width=1, wm=self.hidden_dim_small).to(self.device).double()
        wide_model = create_ResNet(self.output_dim, base_width=1, wm=self.hidden_dim_large).to(self.device).double()

        narrow_optimizer = mup.MuSGD(narrow_model.parameters(), lr=self.lr, momentum=self.momentum, nesterov=True)
        wide_optimizer = mup.MuSGD(wide_model.parameters(), lr=self.lr, momentum=self.momentum, nesterov=True)
        self.check_equivalence(narrow_model, wide_model, narrow_optimizer, wide_optimizer)

    def test_weight_transfer_sgd_wd(self):
        """Test weight transfer with SGD optimizer with weight decay."""
        print("\n=======Testing SGD with weight decay=========")
        narrow_model = create_ResNet(self.output_dim, base_width=1, wm=self.hidden_dim_small).to(self.device).double()
        wide_model = create_ResNet(self.output_dim, base_width=1, wm=self.hidden_dim_large).to(self.device).double()

        narrow_optimizer = mup.MuSGD(narrow_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        wide_optimizer = mup.MuSGD(wide_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.check_equivalence(narrow_model, wide_model, narrow_optimizer, wide_optimizer)

    def test_weight_transfer_adam(self):
        """Test weight transfer with Adam optimizer."""
        print("\n=======Testing Adam=========")
        narrow_model = create_ResNet(self.output_dim, base_width=1, wm=self.hidden_dim_small).to(self.device).double()
        wide_model = create_ResNet(self.output_dim, base_width=1, wm=self.hidden_dim_large).to(self.device).double()

        narrow_optimizer = replaced_optim.MuAdam(narrow_model.parameters(), lr=self.lr)
        wide_optimizer = replaced_optim.MuAdam(wide_model.parameters(), lr=self.lr)
        self.check_equivalence(narrow_model, wide_model, narrow_optimizer, wide_optimizer)

    def test_weight_transfer_adam_wd(self):
        """Test weight transfer with Adam optimizer with weight decay (not decoupled)."""
        print("\n=======Testing Adam with weight decay=========")
        narrow_model = create_ResNet(self.output_dim, base_width=1, wm=self.hidden_dim_small).to(self.device).double()
        wide_model = create_ResNet(self.output_dim, base_width=1, wm=self.hidden_dim_large).to(self.device).double()

        narrow_optimizer = replaced_optim.MuAdam(narrow_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        wide_optimizer = replaced_optim.MuAdam(wide_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.check_equivalence(narrow_model, wide_model, narrow_optimizer, wide_optimizer)

    def test_weight_transfer_adam_amsgrad(self):
        """Test weight transfer with Adam optimizer with amsgrad)."""
        print("\n=======Testing Adam with amsgrad=========")
        narrow_model = create_ResNet(self.output_dim, base_width=1, wm=self.hidden_dim_small).to(self.device).double()
        wide_model = create_ResNet(self.output_dim, base_width=1, wm=self.hidden_dim_large).to(self.device).double()

        narrow_optimizer = replaced_optim.MuAdam(narrow_model.parameters(), lr=self.lr, amsgrad=True)
        wide_optimizer = replaced_optim.MuAdam(wide_model.parameters(), lr=self.lr, amsgrad=True)
        self.check_equivalence(narrow_model, wide_model, narrow_optimizer, wide_optimizer)

    def test_weight_transfer_adamw(self):
        """Test weight transfer with AdamW optimizer."""
        print("\n=======Testing AdamW=========")
        narrow_model = create_ResNet(self.output_dim, base_width=1, wm=self.hidden_dim_small).to(self.device).double()
        wide_model = create_ResNet(self.output_dim, base_width=1, wm=self.hidden_dim_large).to(self.device).double()

        narrow_optimizer = replaced_optim.MuAdamW(narrow_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        wide_optimizer = replaced_optim.MuAdamW(wide_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.check_equivalence(narrow_model, wide_model, narrow_optimizer, wide_optimizer)
