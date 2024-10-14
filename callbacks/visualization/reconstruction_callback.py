import os

import einops
import torch
from kappadata import get_denorm_transform
from kappadata.wrappers import XTransformWrapper

from callbacks.base.periodic_callback import PeriodicCallback
from losses.mae_loss import MaeLoss
from trainers.ssl_trainer import SslTrainer
from utils.object_from_kwargs import objects_from_kwargs
from utils.save_image_utils import save_image_tensors, images_to_gif
from kappamodules.functional.patchify import patchify_as_1d, patchify_as_2d, unpatchify_from_2d, unpatchify


class ReconstructionCallback(PeriodicCallback):
    def __init__(
            self,
            dataset_key,
            reconstruct_kwargs=None,
            num_images=25,
            shuffle_seed=0,
            transpose_xy=False,
            normalize_pixels=None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.dataset_key = dataset_key
        self.reconstruct_kwargs = objects_from_kwargs(reconstruct_kwargs)
        self.num_images = num_images
        self.shuffle_seed = shuffle_seed
        self.images = None
        self.denormalize = None
        self.transpose_xy = transpose_xy
        self.normalize_pixels = normalize_pixels
        # create output folder
        self.out = self.path_provider.stage_output_path / "reconstructions"
        self.out.mkdir(exist_ok=True)
        # enforce seed for mask generators
        if "mask_generator" in self.reconstruct_kwargs:
            assert self.reconstruct_kwargs["mask_generator"].seed is not None

    def _before_training(self, model, trainer, **kwargs):
        self.out.mkdir(exist_ok=True)

        # load some images
        self.logger.info(f"loading {self.num_images} images from '{self.dataset_key}' dataset for reconstruction")
        ds, collator = self.data_container.get_dataset(
            self.dataset_key,
            mode="x",
            shuffle_seed=self.shuffle_seed,
            max_size=self.num_images,
        )
        if collator is not None:
            raise NotImplementedError
        xtransform_wrapper = ds.get_wrapper_of_type(XTransformWrapper)
        if xtransform_wrapper is not None:
            self.denormalize = get_denorm_transform(xtransform_wrapper.transform)
        else:
            self.denormalize = None
        self.images = torch.stack([ds[i][0] for i in range(len(ds))])

        # save original images
        self.logger.info(f"saving original images to {self.out}")
        save_image_tensors(
            tensors=self.images,
            out_uri=self.out / "original.png",
            denormalize=self.denormalize,
            transpose_xy=self.transpose_xy,
        )

        # store masked images if a mask_generator is used
        if "mask_generator" in self.reconstruct_kwargs:
            mask_generator = self.reconstruct_kwargs["mask_generator"]
            self.logger.info(f"saving masked images to {self.out} (mask_generator={mask_generator})")
            patches = patchify_as_2d(self.images, model.patch_size)
            _, _, h, w = patches.shape
            x, mask, _ = mask_generator.get_mask(patches, idx=torch.arange(len(patches), device=patches.device))
            if mask is None:
                # some mask generators only drop patches but generate no mask -> generate mask with iteration
                indices = einops.repeat(
                    torch.arange(h * w),
                    "(h w) -> num_images 1 h w",
                    h=h,
                    w=w,
                    num_images=self.num_images,
                )
                masked_indices, _, _ = mask_generator.get_mask(indices, idx=torch.arange(self.num_images))
                masked_indices = masked_indices.squeeze(-1)
                # not sure if there is a tensor operation for this -> for loop is okay as it is only done once
                # (index_fill only works with 0d/1d indices but not with 2d)
                mask = torch.ones((self.num_images, h * w))
                for i in range(self.num_images):
                    mask[i].index_fill_(dim=0, index=masked_indices[i], value=0.)
            mask = einops.rearrange(mask, "bs (h w) -> bs 1 h w", h=h, w=w)
            masked_imgs = unpatchify_from_2d(patches=patches * (1 - mask), patch_size=model.patch_size)
            save_image_tensors(
                tensors=masked_imgs,
                out_uri=self.out / f"original_{mask_generator}.png",
                denormalize=self.denormalize,
                transpose_xy=self.transpose_xy,
            )

    def _reconstruct(self, x, model, trainer):
        with trainer.autocast_context:
            x_hat = model.reconstruct(x, **self.reconstruct_kwargs, idx=torch.arange(len(x), device=x.device))

        # undo normalize_pixels
        eps = None
        if self.normalize_pixels is not None and self.normalize_pixels:
            eps = 1e-6
            normalize_pixels = True
        else:
            if isinstance(trainer, SslTrainer):
                loss_fns = [loss_fn for loss_fn in trainer.loss_fns.values() if isinstance(loss_fn, MaeLoss)]
                assert len(loss_fns) == 1
                loss_fn = loss_fns[0]
                normalize_pixels = loss_fn.normalize_pixels
                eps = loss_fn.eps
            else:
                normalize_pixels = False

        if normalize_pixels:
            assert x_hat.ndim == 3
            target = patchify_as_1d(x, model.patch_size)
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            x_hat *= (var + eps) ** 0.5
            x_hat += mean

        # unpatchify (internally calls from_2d or from_1d)
        imgs = unpatchify(
            patches=x_hat,
            patch_size=model.patch_size,
            img_shape=model.input_shape,
        )
        return imgs

    # noinspection PyMethodOverriding
    def _periodic_callback(self, model, trainer, **kwargs):
        x = self.images.to(model.device, non_blocking=True)
        with trainer.autocast_context:
            imgs_tensor = self._reconstruct(
                x=x,
                model=model,
                trainer=trainer,
            )
        mask_generator = self.reconstruct_kwargs["mask_generator"]
        # NOTE: to_pil_image handles transfer to cpu
        save_image_tensors(
            tensors=imgs_tensor,
            out_uri=self.out / f"all--{mask_generator}--{self.update_counter.cur_checkpoint}.png",
            denormalize=self.denormalize,
            transpose_xy=self.transpose_xy,
        )
        for i, img in enumerate(imgs_tensor):
            save_image_tensors(
                tensors=img.unsqueeze(0),
                out_uri=self.out / f"{i}--{mask_generator}--{self.update_counter.cur_checkpoint}.png",
                denormalize=self.denormalize,
                transpose_xy=self.transpose_xy,
            )
            # save tensors
            # torch.save(img, self.out / f"{i}--{mask_generator}--{self.update_counter.cur_checkpoint}.th")

    def _after_training(self, **kwargs):
        self.logger.info(f"creating reconstruction gif")
        fnames = [fname for fname in os.listdir(self.out) if "original" not in fname]
        # group by the initial name
        groups = set(fname.split("--")[0] for fname in fnames)
        mask_generator = self.reconstruct_kwargs["mask_generator"]
        for group in groups:
            fnames = [
                fname
                for fname in fnames
                if (
                        fname.split("--")[1] == str(mask_generator)
                        and fname.split("--")[0] == group
                        and fname.endswith(".png")
                )
            ]
            uris = [self.out / name for name in fnames]
            images_to_gif(uris, self.out / f"gif_{mask_generator}.gif")
