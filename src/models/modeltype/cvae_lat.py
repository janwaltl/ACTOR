import torch
from .cae import CAE
from ..tools.losses import get_loss_function


class CVAETokLat(CAE):
    def reparameterize(self, batch, seed=None):
        mu, logvar = batch["mu"], batch["logvar"]
        std = torch.exp(logvar / 2)

        if seed is None:
            eps = std.data.new(std.size()).normal_()
        else:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)
            eps = std.data.new(std.size()).normal_(generator=generator)

        z = eps.mul(std).add_(mu)
        return z

    def forward(self, batch):

        if self.outputxyz:
            batch["x_xyz"] = self.rot2xyz(batch["x"], batch["mask"])
        elif self.pose_rep == "xyz":
            batch["x_xyz"] = batch["x"]
        # encode
        batch.update(self.encoder(batch))
        batch["z"] = self.reparameterize(batch)

        # decode

        batch.update(self.decoder(batch))

        # if we want to output xyz
        if self.outputxyz:
            batch["output_xyz"] = self.rot2xyz(batch["output"], batch["mask"])
        elif self.pose_rep == "xyz":
            batch["output_xyz"] = batch["output"]
        return batch

    def return_latent(self, batch, seed=None):
        distrib_param = self.encoder(batch)
        batch.update(distrib_param)
        return self.reparameterize(batch, seed=seed)

    def compute_loss(self, batch):
        """Additionally compute latent decoder loss."""
        super().compute_loss(batch)
        __import__('pdb').set_trace()

        # Add loss on latent
        x = batch["x"]
        self.decoder.auto.(x)



        batch["latent_dec"]
        return mixed_loss, losses
